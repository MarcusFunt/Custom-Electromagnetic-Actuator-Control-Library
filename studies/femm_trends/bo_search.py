"""Bayesian optimization of the linear stepper over REAL-FEMM geometry (docs/DESIGN_OPTIMIZER.md).

  ############################################################################
  #  Uses REAL FEMM in the loop (a CORE requirement, like run_study.py):     #
  #  every geometry evaluation builds ONE real axisymmetric FEMM force LUT.   #
  #  Install the FEMM app (http://www.femm.info/) AND `pip install pyfemm`.   #
  #  Windows-only. The analytic path (--backend reference) needs neither and  #
  #  exists only for fast tests -- it is NOT the tool's purpose.              #
  #                                                                           #
  #  EXCLUSIVE FEMM: concurrent FEMM instances hang (observed, multi-hour).   #
  #  Run this ONLY when no other FEMM is running (i.e. stop the factorial     #
  #  sweep first). See run_study.py's header on pyfemm COM fragility.         #
  ############################################################################

Why BO and not the factorial sweep (run_study.py): the sweep spends equal, expensive FEMM
budget on hopeless-slow and fast designs alike -- most of its 972 cells land under ~3 m/s
while a thin tail reaches ~20. This concentrates FEMM effort in the fast end of the design
space using a Gaussian-process surrogate + Expected-Improvement acquisition (scikit-optimize),
and WARM-STARTS from whatever the sweep has already produced so that compute is reused, not
discarded.

The factoring that makes it cheap: a FEMM force table (ForceLUT) depends ONLY on the 6
geometry knobs (turns, coil_length_m, radial_thickness_m, magnet_radius_m, magnet_length_m,
remanence_t) -- it is position-independent and reused by every coil. The driver knobs
(bus_voltage_v, i_max_a, driver_bipolar, pump_envelope) and n_coils change only the CHEAP
simulation, not the LUT. So BO searches the 6 expensive geometry dimensions, and each
evaluation does a cheap inner maximization over the 32 driver combos on that single LUT.

Robustness (mirrors run_study.py, unattended for hours):
  - Each FEMM evaluation runs in a SUBPROCESS with a hard wall-clock timeout; a wedged FEMM
    COM call (which cannot be interrupted in-thread) is killed with the process and femm.exe
    is force-killed, so one stalled build can't freeze the whole search. That geometry scores
    0 and the search moves on.
  - The subprocess runs FEMM from a short cwd (C:\\femmwork\\bo) to dodge the Windows MAX_PATH
    solver hang (see run_study.py).
  - RESUMABLE: bo_eval_log.jsonl is the source of truth. On restart every logged
    (geometry -> objective) is replayed via tell() to reconstruct the optimizer, then the
    search continues.
  - A GUI-compatible build/optimize_results/latest.json snapshot is rewritten after each
    evaluation, so emac-gui -> Optimizer shows the run live.

Run:   python studies/femm_trends/bo_search.py --n-calls 250
Resume: just run it again with the same --outdir (replays the log).
Tests:  inject --backend reference (analytic, no FEMM) -- see tests/test_bo_search.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "python"))
sys.path.insert(0, str(HERE))

import study_lib  # noqa: E402  (path set above; also pulls in emac_sim)
from emac_sim import coil_design, optimize_design as od  # noqa: E402
from emac_sim.fem.reference_backend import AnalyticReferenceBackend  # noqa: E402

_THIS = str(Path(__file__).resolve())

# ----------------------------- search space ----------------------------------
# The 6 expensive geometry dimensions. Bounds are the study's proven FEMM-MESHABLE ranges
# (min/max of run_study.GEOM_FACTORS), NOT optimize_design.Bounds -- those are wider than
# FEMM can mesh (big coils balloon the air domain and the mesher fails; see run_study.py).
GEOM_KEYS = ["turns", "coil_length_m", "radial_thickness_m",
             "magnet_radius_m", "magnet_length_m", "remanence_t"]
GEOM_BOUNDS = {
    "turns":              (80, 900),      # Integer
    "coil_length_m":      (0.010, 0.032),
    "radial_thickness_m": (0.005, 0.028),
    "magnet_radius_m":    (0.004, 0.015),
    "magnet_length_m":    (0.010, 0.036),
    "remanence_t":        (0.5, 1.25),
}

# The 32 driver combos for the cheap inner sweep (same axes as run_study.DRIVER_FACTORS).
DRIVER_FACTORS = {
    "bus_voltage_v":  [12.0, 40.0, 110.0, 260.0],
    "driver_bipolar": [False, True],
    "pump_envelope":  ["rcos", "square"],
    "i_max_a":        [20.0, 70.0],
}
OBJECTIVES = ("speed", "energy", "momentum")


def driver_combos():
    keys = list(DRIVER_FACTORS)
    return [dict(zip(keys, vals)) for vals in itertools.product(*DRIVER_FACTORS.values())]


def build_space():
    """scikit-optimize dimension list, in GEOM_KEYS order (turns Integer, rest Real)."""
    from skopt.space import Integer, Real
    dims = []
    for k in GEOM_KEYS:
        lo, hi = GEOM_BOUNDS[k]
        dims.append(Integer(int(lo), int(hi), name=k) if k == "turns"
                    else Real(float(lo), float(hi), name=k))
    return dims


def geom_from_x(x):
    g = dict(zip(GEOM_KEYS, x))
    g["turns"] = int(round(g["turns"]))
    for k in GEOM_KEYS:
        if k != "turns":
            g[k] = float(g[k])
    return g


def x_from_geom(g):
    return [int(round(g["turns"]))] + [float(g[k]) for k in GEOM_KEYS if k != "turns"]


def objective_value(speed_mps: float, mass_kg: float, kind: str) -> float:
    if kind == "speed":
        return float(speed_mps)
    if kind == "energy":
        return 0.5 * mass_kg * speed_mps * speed_mps
    if kind == "momentum":
        return mass_kg * speed_mps
    raise ValueError(f"unknown objective {kind!r}")


# ----------------------------- one geometry evaluation ------------------------
def evaluate_geometry(geom: dict, n_coils: int, objective: str, backend,
                      max_tube_length_m: float = 1.0, sim_t_end: float = 3.0,
                      slug_type: str = "pm") -> dict:
    """Build ONE FEMM (or reference) LUT for `geom`, then cheaply maximize the objective over
    the 32 driver combos on that LUT. Returns the best objective and the winning full design.
    A LUT-build failure, or a tube longer than max_tube_length_m, scores 0 (like the sim).

    slug_type: "pm" (permanent-magnet, default) or "reluctance" (soft-iron). Selects the FEMM
    slug material + the reluctance current sampling, and the slug mass/force law.

    sim_t_end: horizon of the inner exit-speed sim (default 3.0, the study's value). The inner
    sweep is 32 ODE sims/geometry -- a shorter horizon speeds a long unattended run and only
    affects designs so slow they'd never clear the tube in ~1 s (which score ~0 either way)."""
    mass_kg = coil_design.slug_mass_kg(geom["magnet_radius_m"], geom["magnet_length_m"], slug_type)
    base = {"geom": geom, "n_coils": n_coils, "objective": objective, "mass_kg": mass_kg,
            "slug_type": slug_type, "value": 0.0, "speed_mps": 0.0, "driver": None,
            "n_evaluated": 0, "error": None}

    if n_coils * geom["coil_length_m"] > max_tube_length_m:
        base["error"] = "tube_too_long"
        return base

    try:
        lut = study_lib.build_femm_lut(
            geom["turns"], geom["coil_length_m"], geom["radial_thickness_m"],
            geom["magnet_radius_m"], geom["magnet_length_m"], geom["remanence_t"], backend,
            slug_type=slug_type)
    except Exception as e:  # noqa: BLE001 -- a LUT failure must not kill the search
        base["error"] = f"lut_failed: {e!r}"
        return base

    best_val, best_speed, best_driver, n_eval = 0.0, 0.0, None, 0
    for driver in driver_combos():
        knobs = od.DesignKnobs(
            n_coils=n_coils, turns=geom["turns"], coil_length_m=geom["coil_length_m"],
            radial_thickness_m=geom["radial_thickness_m"], magnet_radius_m=geom["magnet_radius_m"],
            magnet_length_m=geom["magnet_length_m"], remanence_t=geom["remanence_t"],
            slug_type=slug_type, **driver)
        try:
            speed = study_lib.simulate_exit_speed(knobs, "femm", lut, t_end=sim_t_end)
        except Exception:  # noqa: BLE001 -- a single bad driver combo is not fatal
            speed = 0.0
        n_eval += 1
        val = objective_value(speed, mass_kg, objective)
        if val > best_val:
            best_val, best_speed, best_driver = val, speed, driver

    base.update(value=best_val, speed_mps=best_speed, driver=best_driver, n_evaluated=n_eval)
    return base


def best_knobs_dict(geom: dict, driver: dict | None, n_coils: int,
                    slug_type: str = "pm") -> dict | None:
    if driver is None:
        return None
    knobs = od.DesignKnobs(
        n_coils=n_coils, turns=geom["turns"], coil_length_m=geom["coil_length_m"],
        radial_thickness_m=geom["radial_thickness_m"], magnet_radius_m=geom["magnet_radius_m"],
        magnet_length_m=geom["magnet_length_m"], remanence_t=geom["remanence_t"],
        slug_type=slug_type, **driver)
    return dataclasses.asdict(knobs)


# ----------------------------- evaluators (in-process / subprocess) -----------
def make_backend(name: str):
    if name == "femm":
        b = study_lib.CorrectedFemmBackend(keep_open=True)
        b._tmp_fem = f"_bo_{os.getpid()}.fem"
        return b
    if name == "reference":
        return AnalyticReferenceBackend()
    raise ValueError(f"unknown backend {name!r}")


def inprocess_evaluator(n_coils, objective, backend_name="reference", max_tube_length_m=1.0,
                        sim_t_end=3.0, slug_type="pm"):
    """Evaluate geometries in THIS process with a fresh/held backend. Used by the tests with
    the analytic reference backend (no FEMM, no subprocess). Not hang-isolated -- do not use
    with real FEMM for a long unattended run; use subprocess_evaluator for that."""
    backend = make_backend(backend_name)

    def _eval(geom):
        return evaluate_geometry(geom, n_coils, objective, backend, max_tube_length_m,
                                 sim_t_end, slug_type)

    return _eval


def _kill_femm():
    """Force-kill every femm.exe. Safe here because BO requires EXCLUSIVE FEMM (no other run
    should be using it). No-op on non-Windows / when femm.exe isn't running."""
    try:
        subprocess.run(["taskkill", "/F", "/IM", "femm.exe", "/T"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    except Exception:  # noqa: BLE001
        pass


def subprocess_evaluator(n_coils, objective, backend_name="femm", timeout_s=900.0,
                         workdir=None, max_tube_length_m=1.0, sim_t_end=3.0, slug_type="pm"):
    """Evaluate each geometry in a SUBPROCESS (this same file, --worker mode) with a hard
    timeout. On timeout/crash the worker + any femm.exe are killed and the geometry scores 0.
    This is what makes a wedged FEMM COM call survivable in a long unattended search."""
    workdir = Path(workdir) if workdir else Path(r"C:\femmwork\bo")
    workdir.mkdir(parents=True, exist_ok=True)

    def _eval(geom):
        tag = f"{os.getpid()}_{int(time.time()*1000) % 1_000_000}"
        in_json = workdir / f"in_{tag}.json"
        out_json = workdir / f"out_{tag}.json"
        payload = {"geom": geom, "n_coils": n_coils, "objective": objective,
                   "backend": backend_name, "max_tube_length_m": max_tube_length_m,
                   "sim_t_end": sim_t_end, "slug_type": slug_type}
        in_json.write_text(json.dumps(payload))
        mass_kg = coil_design.slug_mass_kg(geom["magnet_radius_m"], geom["magnet_length_m"], slug_type)
        fail = {"geom": geom, "n_coils": n_coils, "objective": objective, "mass_kg": mass_kg,
                "slug_type": slug_type, "value": 0.0, "speed_mps": 0.0, "driver": None,
                "n_evaluated": 0}
        try:
            proc = subprocess.Popen([sys.executable, _THIS, "--worker", str(in_json),
                                     str(out_json)], cwd=str(workdir))
        except Exception as e:  # noqa: BLE001
            return {**fail, "error": f"spawn_failed: {e!r}"}
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            if backend_name == "femm":
                _kill_femm()
            return {**fail, "error": "timeout"}
        finally:
            try:
                in_json.unlink()
            except OSError:
                pass
        if proc.returncode != 0 or not out_json.exists():
            if backend_name == "femm":
                _kill_femm()
            return {**fail, "error": f"worker_exit_{proc.returncode}"}
        try:
            res = json.loads(out_json.read_text())
        except Exception as e:  # noqa: BLE001
            res = {**fail, "error": f"bad_worker_output: {e!r}"}
        finally:
            try:
                out_json.unlink()
            except OSError:
                pass
        return res

    return _eval


def _worker_main(in_json: str, out_json: str) -> int:
    """Subprocess entry: read a geometry payload, evaluate it (building its own backend), write
    the result. Runs in a short cwd (Popen cwd=) so FEMM's solver stays under MAX_PATH."""
    try:
        payload = json.loads(Path(in_json).read_text())
        backend = make_backend(payload["backend"])
        if hasattr(backend, "_ensure_open"):
            backend._ensure_open()
        res = evaluate_geometry(payload["geom"], payload["n_coils"], payload["objective"],
                                backend, payload.get("max_tube_length_m", 1.0),
                                payload.get("sim_t_end", 3.0), payload.get("slug_type", "pm"))
        try:
            if hasattr(backend, "close"):
                backend.close()
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001 -- still emit a result so the parent isn't left guessing
        res = {"value": 0.0, "speed_mps": 0.0, "driver": None, "n_evaluated": 0,
               "error": f"worker_crash: {e!r}"}
    Path(out_json).write_text(json.dumps(res, default=_json_default))
    return 0


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


# ----------------------------- warm start -------------------------------------
def load_warmstart(dirs, objective: str, slug_type: str = "pm"):
    """Reduce existing sweep results to per-geometry prior RECORDS: for each distinct geometry,
    the BEST-scoring driver row under the CORRECTED 'femm' force law, as a dict
    {geom, value, speed_mps, driver, n_coils}. Both the GP priors AND an honest starting best
    (with full reconstructed knobs) come from these.

    Only force_law=='femm' rows of the SAME slug_type are used (rows without a slug_type field
    are the original PM sweep). The legacy committed results/ dir uses the PRE-FIX stress-tensor
    extraction and is excluded by default -- its objectives are distorted and would mislead the
    surrogate."""
    best = {}
    n_rows = 0
    driver_keys = list(DRIVER_FACTORS)
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in sorted(d.glob("cell_*.jsonl")):
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("force_law") != "femm" or r.get("exit_speed_mps") is None:
                    continue
                if r.get("slug_type", "pm") != slug_type:
                    continue
                if any(k not in r for k in GEOM_KEYS):
                    continue
                n_rows += 1
                g = tuple(r[k] for k in GEOM_KEYS)
                mass = coil_design.magnet_mass_kg(r["magnet_radius_m"], r["magnet_length_m"])
                val = objective_value(r["exit_speed_mps"], mass, objective)
                cur = best.get(g)
                if cur is None or val > cur["value"]:
                    best[g] = {
                        "geom": dict(zip(GEOM_KEYS, g)),
                        "value": val,
                        "speed_mps": r["exit_speed_mps"],
                        "driver": {k: r[k] for k in driver_keys if k in r},
                        "n_coils": r.get("n_coils", 5),
                    }
    return list(best.values()), n_rows


# ----------------------------- snapshot (GUI-compatible) ----------------------
def write_snapshot(path: Path, objective: str, best, history, n_done, elapsed_s, slug_type="pm"):
    """Write build/optimize_results/latest.json in the shape emac-gui's Optimizer tab reads:
    best_speed_mps / best_value / best_knobs / history[{best,value}] / generation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    snap = {
        "source": "bo_search",
        "objective": objective,
        "slug_type": slug_type,
        "best_value": best["value"],
        "best_speed_mps": best["speed_mps"],
        "best_knobs": best["knobs"],
        "history": history,
        "generation": n_done,
        "elapsed_s": round(elapsed_s, 1),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snap, default=_json_default, indent=2))
    os.replace(tmp, path)


# ----------------------------- the BO loop ------------------------------------
def run_bo(evaluate, *, n_calls, objective, n_coils, outdir, snapshot_path,
           warmstart_dirs=None, n_initial=10, seed=0, budget_s=0.0, slug_type="pm", log=print):
    """Core ask/tell loop. `evaluate(geom)->result` is injected (subprocess-FEMM for real runs,
    in-process-reference for tests). Returns the final best record.

    Resume: replays outdir/bo_eval_log.jsonl into the optimizer before continuing.
    Warm-start: tells per-geometry sweep bests (of the same slug_type)."""
    from skopt import Optimizer

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "bo_eval_log.jsonl"

    opt = Optimizer(build_space(), base_estimator="GP", acq_func="EI",
                    n_initial_points=n_initial, random_state=seed)

    history, n_done = [], 0
    best = {"value": -1.0, "speed_mps": 0.0, "knobs": None, "geom": None, "driver": None}

    def _account(rec):
        nonlocal n_done
        n_done += 1
        if rec["value"] > best["value"]:
            best.update(value=rec["value"], speed_mps=rec["speed_mps"], geom=rec["geom"],
                        driver=rec["driver"],
                        knobs=best_knobs_dict(rec["geom"], rec["driver"], n_coils, slug_type))
        history.append({"eval": n_done, "value": rec["value"], "speed_mps": rec["speed_mps"],
                        "best": best["value"]})

    # ---- warm-start (applied BEFORE the log replay, on fresh AND resumed runs) ----
    # Priors both (a) seed the GP surrogate and (b) set an honest starting best, so a BO run
    # that fails to beat the sweep still reports the sweep's best design, not a worse one.
    # Warm-start points are never written to the eval log (they come from the sweep, not from
    # BO's own asks), so re-applying them on resume can't double-count -- and doing so keeps a
    # resumed run's GP/best faithful to the original.
    if warmstart_dirs:
        recs, n_rows = load_warmstart(warmstart_dirs, objective, slug_type)
        if recs:
            opt.tell([x_from_geom(r["geom"]) for r in recs], [-r["value"] for r in recs])
            top = max(recs, key=lambda r: r["value"])
            if top["value"] > best["value"]:
                best.update(value=top["value"], speed_mps=top["speed_mps"], geom=top["geom"],
                            driver=top["driver"],
                            knobs=best_knobs_dict(top["geom"], top["driver"], top["n_coils"],
                                                  slug_type))
            log(f"warm-start: {len(recs)} distinct geometries from {n_rows} sweep rows "
                f"(best prior {objective} {top['value']:.3f})")

    # ---- resume: replay the log on top of the priors ----
    resumed = 0
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            opt.tell(x_from_geom(rec["geom"]), -rec["value"])
            _account(rec)
            resumed += 1
        if resumed:
            log(f"resumed: replayed {resumed} evaluations from {log_path.name} "
                f"(best {objective} so far {best['value']:.3f})")

    # ---- ask / evaluate / tell ----
    # n_calls is a TOTAL budget: a resumed run continues TOWARD it (not n_calls more). If the
    # log already has >= n_calls evaluations, there is nothing left to do.
    t0 = time.time()
    if resumed >= n_calls:
        log(f"already have {resumed} evaluations (>= n_calls={n_calls}); nothing to do")
    while n_done < n_calls:
        if budget_s and (time.time() - t0) >= budget_s:
            log(f"wall-clock budget {budget_s:.0f}s reached after {n_done - resumed} new evals")
            break
        x = opt.ask()
        geom = geom_from_x(x)
        rec = evaluate(geom)
        # evaluate() may return a geometry it clamped internally; tell the point we ASKED.
        opt.tell(x, -float(rec.get("value", 0.0)))
        rec.setdefault("geom", geom)
        _account(rec)
        with log_path.open("a") as fh:
            fh.write(json.dumps({"geom": geom, "value": rec["value"], "speed_mps": rec["speed_mps"],
                                 "driver": rec["driver"], "objective": objective,
                                 "slug_type": slug_type, "error": rec.get("error")},
                                default=_json_default) + "\n")
        if snapshot_path:
            write_snapshot(Path(snapshot_path), objective, best, history,
                           n_done, time.time() - t0, slug_type)
        tail = f"  [{rec['error']}]" if rec.get("error") else ""
        log(f"eval {n_done}/{n_calls}: {objective}={rec['value']:.3f} "
            f"(v={rec['speed_mps']:.2f} m/s)  best={best['value']:.3f}{tail}")

    log(f"done. best {objective}={best['value']:.3f}  "
        f"(exit speed {best['speed_mps']:.2f} m/s)")
    if best["knobs"]:
        log(f"best design: {json.dumps(best['knobs'], default=_json_default)}")
    return best


# ----------------------------- CLI -------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="Bayesian optimization of the linear stepper over "
                                            "real-FEMM geometry.")
    p.add_argument("--n-calls", type=int, default=250, help="FEMM geometry evaluations (default 250)")
    p.add_argument("--objective", choices=OBJECTIVES, default="speed",
                   help="what to maximize (default: exit speed m/s)")
    p.add_argument("--n-coils", type=int, default=5, help="coils in the inner sim (default 5, "
                   "matches the factorial sweep so warm-start stays consistent)")
    p.add_argument("--min-magnet-radius", type=float, default=None,
                   help="override the magnet_radius_m lower search bound (m). Default keeps the "
                   f"study's FEMM-safe {GEOM_BOUNDS['magnet_radius_m'][0]}. Smaller magnets mesh "
                   "finer/slower; LUT builds that fail or time out just score 0, so it's safe to try.")
    p.add_argument("--timeout", type=float, default=900.0, help="hard per-evaluation timeout, s")
    p.add_argument("--sim-t-end", type=float, default=3.0, help="inner exit-speed sim horizon, s "
                   "(default 3.0 = the study's value; lower is faster, only drops sub-~1 m/s designs)")
    p.add_argument("--n-initial", type=int, default=10, help="random points before the GP kicks in")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-s", type=float, default=0.0, help="optional wall-clock cap (0=none)")
    p.add_argument("--backend", choices=("femm", "reference"), default="femm",
                   help="femm (real, default) or reference (analytic, no FEMM -- testing only)")
    p.add_argument("--slug-type", choices=("pm", "reluctance"), default="pm",
                   help="pm (permanent-magnet, default) or reluctance (soft-iron slug). Selects "
                        "the FEMM slug material + reluctance current sampling; warm-start only "
                        "ingests sweep rows of the same slug_type.")
    p.add_argument("--no-warm-start", action="store_true", help="ignore existing sweep results")
    p.add_argument("--warmstart-dir", action="append", default=None,
                   help="extra results dir to warm-start from (repeatable)")
    p.add_argument("--outdir", default=str(HERE / "bo"), help="log + state dir")
    p.add_argument("--workdir", default=r"C:\femmwork\bo", help="short FEMM cwd (MAX_PATH)")
    p.add_argument("--snapshot", default=str(REPO_ROOT / "build" / "optimize_results" / "latest.json"),
                   help="GUI-compatible live snapshot path")
    return p


def main(argv):
    if argv and argv[0] == "--worker":
        return _worker_main(argv[1], argv[2])

    args = build_arg_parser().parse_args(argv)

    # Optional wider magnet-radius search (build_space reads GEOM_BOUNDS at call time inside
    # run_bo, so mutating the global here takes effect for the new Optimizer's space).
    if args.min_magnet_radius is not None:
        hi = GEOM_BOUNDS["magnet_radius_m"][1]
        GEOM_BOUNDS["magnet_radius_m"] = (float(args.min_magnet_radius), hi)
        print(f"magnet_radius_m search bound widened to "
              f"[{args.min_magnet_radius}, {hi}] m", flush=True)

    warmstart_dirs = None
    if not args.no_warm_start:
        # default: the CORRECTED current sweep only (study/results). The legacy committed
        # results/ dir is pre-fix and excluded on purpose; add it explicitly with
        # --warmstart-dir if you really want it.
        warmstart_dirs = [HERE / "study" / "results"]
        if args.warmstart_dir:
            warmstart_dirs += [Path(d) for d in args.warmstart_dir]

    if args.backend == "femm":
        evaluate = subprocess_evaluator(args.n_coils, args.objective, backend_name="femm",
                                        timeout_s=args.timeout, workdir=args.workdir,
                                        sim_t_end=args.sim_t_end, slug_type=args.slug_type)
    else:
        evaluate = inprocess_evaluator(args.n_coils, args.objective, backend_name="reference",
                                       sim_t_end=args.sim_t_end, slug_type=args.slug_type)

    run_bo(evaluate, n_calls=args.n_calls, objective=args.objective, n_coils=args.n_coils,
           outdir=args.outdir, snapshot_path=args.snapshot, warmstart_dirs=warmstart_dirs,
           n_initial=args.n_initial, seed=args.seed, budget_s=args.budget_s,
           slug_type=args.slug_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
