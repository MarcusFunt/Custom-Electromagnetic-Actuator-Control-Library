"""Combined HARDWARE Bayesian optimization x RL WAVEFORM control -- "how far can PM go?".

Outer loop: a GP + Expected-Improvement search (scikit-optimize) over the 7 PM GEOMETRY knobs
(the driver is fixed at the chosen H-bridge envelope, default 450 V / 100 A). Inner loop: for
each candidate gun, a PPO policy is WARM-STARTED from a shared champion and briefly fine-tuned
in that gun's `CoilgunEnv`, then evaluated -- the design's score is the RL-controlled exit speed
(the learned drive waveform, not a hand-tuned heuristic). The champion is promoted whenever a
candidate beats it, so the policy co-evolves with the hardware.

Why this is tractable for a ~9 h unattended run:
  - the RL inner loop uses the ANALYTIC PM force law (closed form, no FEMM) -- millions of env
    steps per candidate are cheap (it runs ~7% optimistic vs real FEMM; validate the winner in
    FEMM afterwards);
  - the observation/action are normalized + translation-invariant, so ONE policy transfers
    across every geometry -- warm-starting each candidate from the champion means a short
    fine-tune (not a from-scratch train) suffices.

Robust like run_study.py / bo_search.py: resumable (replay hw_eval_log.jsonl), wall-clock
budget, per-eval try/except (a bad candidate scores 0, never kills the run), atomic snapshot to
build/optimize_results/latest.json for the live GUI view.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
# .../tools/python/emac_sim/rl/hw_search.py -> repo root is parents[4] (rl, emac_sim, python, tools)
REPO_ROOT = HERE.parents[4]
sys.path.insert(0, str(REPO_ROOT / "tools" / "python"))

from emac_sim.rl.geometry import CoilgunSpec                    # noqa: E402
from emac_sim.rl.env import CoilgunEnv                          # noqa: E402

BUS_V_DEFAULT = 450.0
I_MAX_DEFAULT = 100.0
MAX_TUBE_M = 0.5                                                # tube-length budget

HW_KEYS = ["n_coils", "turns", "coil_length_m", "radial_thickness_m",
           "magnet_radius_m", "magnet_length_m", "remanence_t"]
HW_BOUNDS = {
    "n_coils":            (8, 24),
    "turns":              (150, 900),
    "coil_length_m":      (0.008, 0.030),
    "radial_thickness_m": (0.004, 0.020),
    "magnet_radius_m":    (0.003, 0.010),
    "magnet_length_m":    (0.008, 0.030),
    "remanence_t":        (0.90, 1.30),
}
# a known-good seed (the RL geometry that reached ~50 m/s) to warm-start the GP
SEED_X = [16, 450, 0.012, 0.006, 0.004, 0.014, 1.25]


def build_space():
    from skopt.space import Integer, Real
    dims = []
    for k in HW_KEYS:
        lo, hi = HW_BOUNDS[k]
        dims.append(Integer(int(lo), int(hi), name=k) if k in ("n_coils", "turns")
                    else Real(float(lo), float(hi), name=k))
    return dims


def spec_from_x(x, bus_v, i_max) -> CoilgunSpec:
    return CoilgunSpec(
        n_coils=int(round(x[0])), turns=int(round(x[1])), coil_length_m=float(x[2]),
        radial_thickness_m=float(x[3]), magnet_radius_m=float(x[4]),
        magnet_length_m=float(x[5]), remanence_t=float(x[6]),
        bus_voltage_v=bus_v, i_max_a=i_max)


def evaluate_hw(x, champion_path, timesteps, workdir, bus_v, i_max, n_envs=8, seed=0):
    """Fine-tune the champion on this gun and return {speed, efficiency, model_path}."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from emac_sim.rl.evaluate import evaluate_model

    spec = spec_from_x(x, bus_v, i_max)
    if spec.tube_length_m > MAX_TUBE_M:
        return {"speed": 0.0, "efficiency": 0.0, "error": "tube_too_long", "model_path": None}
    venv = make_vec_env(lambda: CoilgunEnv(spec=spec, lam=0.0, sensing="beam_break"),
                        n_envs=n_envs, seed=seed)
    kw = dict(n_steps=2048, batch_size=256, gae_lambda=0.95, gamma=0.99, ent_coef=0.01,
              learning_rate=3e-4, verbose=0, seed=seed)
    if champion_path and Path(champion_path).exists():
        model = PPO.load(champion_path, env=venv, **{k: kw[k] for k in ("learning_rate", "verbose")})
    else:
        model = PPO("MlpPolicy", venv, **kw)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    m = evaluate_model(model, spec, lam=0.0, sensing="beam_break", seeds=range(12))
    tmp = Path(workdir) / f"cand_{os.getpid()}.zip"
    model.save(str(tmp).replace(".zip", ""))
    del model, venv
    gc.collect()
    return {"speed": float(m["v"]), "v_std": float(m["v_std"]), "efficiency": float(m["efficiency"]),
            "exit_rate": float(m.get("exit_rate", 0.0)), "error": None, "model_path": str(tmp)}


def write_snapshot(path: Path, best, history, n_done, elapsed_s):
    path.parent.mkdir(parents=True, exist_ok=True)
    snap = {"source": "hw_rl_search", "objective": "speed",
            "best_speed_mps": best["speed"], "best_value": best["speed"],
            "best_knobs": best.get("knobs"), "history": history, "generation": n_done,
            "elapsed_s": round(elapsed_s, 1), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snap, indent=2))
    os.replace(tmp, path)


def run(hours, timesteps, outdir, bus_v, i_max, n_envs, seed, init_champion, snapshot, log=print):
    from skopt import Optimizer

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    workdir = outdir / "work"; workdir.mkdir(exist_ok=True)
    champion = outdir / "champion.zip"
    if init_champion and Path(init_champion).exists() and not champion.exists():
        shutil.copy(init_champion, champion)
    log_path = outdir / "hw_eval_log.jsonl"

    opt = Optimizer(build_space(), base_estimator="GP", acq_func="EI", n_initial_points=12,
                    random_state=seed)
    history, n_done = [], 0
    best = {"speed": -1.0, "knobs": None}

    def account(x, res):
        nonlocal n_done
        n_done += 1
        if res["speed"] > best["speed"]:
            best.update(speed=res["speed"], efficiency=res.get("efficiency"),
                        knobs={**dict(zip(HW_KEYS, x)), "bus_voltage_v": bus_v, "i_max_a": i_max})
            if res.get("model_path") and Path(res["model_path"]).exists():
                shutil.copy(res["model_path"], champion)         # promote the winning policy
        history.append({"eval": n_done, "speed": res["speed"], "best": best["speed"]})

    # ---- resume: replay the log ----
    resumed = 0
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            opt.tell(rec["x"], -rec["speed"])
            account(rec["x"], rec)
            resumed += 1
        if resumed:
            log(f"resumed {resumed} evals (best {best['speed']:.2f} m/s)")
    else:
        opt.tell(SEED_X, 0.0)   # weak prior: nudge the GP toward the known-good geometry region

    t0 = time.time()
    budget_s = hours * 3600.0
    while time.time() - t0 < budget_s:
        # coerce skopt's numpy scalars to native python (Integer dims come back as int64,
        # which json.dumps can't serialize) before logging/telling
        x = [int(round(v)) if HW_KEYS[i] in ("n_coils", "turns") else float(v)
             for i, v in enumerate(opt.ask())]
        champ = str(champion) if champion.exists() else None
        try:
            res = evaluate_hw(x, champ, timesteps, workdir, bus_v, i_max, n_envs=n_envs, seed=seed)
        except Exception as e:  # noqa: BLE001 -- a bad candidate must never kill a 9 h run
            res = {"speed": 0.0, "efficiency": 0.0, "error": f"{e!r}", "model_path": None}
        opt.tell(x, -float(res["speed"]))
        account(x, res)
        with log_path.open("a") as fh:
            fh.write(json.dumps({"x": x, "geom": dict(zip(HW_KEYS, x)), "speed": res["speed"],
                                 "efficiency": res.get("efficiency"), "error": res.get("error")}) + "\n")
        if snapshot:
            write_snapshot(Path(snapshot), best, history, n_done, time.time() - t0)
        el = time.time() - t0
        log(f"eval {n_done} ({el/3600:.2f}h): {res['speed']:.2f} m/s "
            f"eff={100*(res.get('efficiency') or 0):.1f}%  best={best['speed']:.2f}"
            f"{'  [' + res['error'] + ']' if res.get('error') else ''}", flush=True)
        gc.collect()

    log(f"DONE. best RL-controlled exit speed = {best['speed']:.2f} m/s", flush=True)
    log(f"best gun: {json.dumps(best.get('knobs'))}", flush=True)
    (outdir / "best.json").write_text(json.dumps(best, indent=2))
    return best


def build_arg_parser():
    p = argparse.ArgumentParser(description="Hardware BO x RL-waveform search for the PM coilgun.")
    p.add_argument("--hours", type=float, default=9.0)
    p.add_argument("--timesteps", type=int, default=120_000, help="RL fine-tune steps per candidate")
    p.add_argument("--bus-voltage", type=float, default=BUS_V_DEFAULT)
    p.add_argument("--i-max", type=float, default=I_MAX_DEFAULT)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="build/rl_hw")
    p.add_argument("--init-champion", default="build/rl_pareto/ppo_lam0.zip")
    p.add_argument("--snapshot", default=str(REPO_ROOT / "build" / "optimize_results" / "latest.json"))
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args.hours, args.timesteps, args.outdir, args.bus_voltage, args.i_max, args.n_envs,
        args.seed, args.init_champion, args.snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
