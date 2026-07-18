"""Train an RL policy to drive the many-stage PM coilgun (Stable-Baselines3 PPO / RecurrentPPO).

`emac-rl-train --lam 0.0 --timesteps 300000` trains one policy at one speed/efficiency weight;
`--pareto` sweeps several lambdas to trace the frontier. Models + a metrics JSON land in
`--outdir`. The env is CPU-cheap, so many parallel envs help. RecurrentPPO (`--recurrent`) is
the fallback for the partial-observability (beam-break) case if a feedforward policy plateaus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .env import CoilgunEnv
from .geometry import CoilgunSpec


def make_vec_env(spec, force_lut, lam, n_envs, sensing, seed):
    from stable_baselines3.common.env_util import make_vec_env as _mk
    return _mk(lambda: CoilgunEnv(spec=spec, force_lut=force_lut, lam=lam, sensing=sensing),
               n_envs=n_envs, seed=seed)


def train_one(spec: CoilgunSpec, lam: float, timesteps: int, outdir: Path, n_envs: int = 8,
              sensing: str = "beam_break", seed: int = 0, recurrent: bool = False,
              force_lut=None, verbose: int = 0):
    """Train a single PPO policy at weight `lam`; save it; return (model, eval-metrics)."""
    venv = make_vec_env(spec, force_lut, lam, n_envs, sensing, seed)
    # ent_coef/gamma tuned for this task: strong exploration (the agent must DISCOVER that
    # push-behind+pull-ahead accelerates -- a zero-current init just coasts and stalls), and a
    # ~100-step effective horizon matching the ~100-300-step episodes.
    kw = dict(n_steps=2048, batch_size=256, gae_lambda=0.95, gamma=0.99, ent_coef=0.01,
              learning_rate=3e-4, verbose=verbose, seed=seed)
    if recurrent:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO("MlpLstmPolicy", venv, **kw)
    else:
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", venv, **kw)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    outdir.mkdir(parents=True, exist_ok=True)
    # Sanitize the lambda in the filename: SB3's save/load treats the '.' in e.g. "ppo_lam0.05"
    # as a file extension, so it would NOT append/expect ".zip" and the saved/loaded names drift
    # apart. Use a dot-free tag (0.05 -> "0p05") so model.save reliably lands "<tag>.zip".
    tag = ("%g" % lam).replace(".", "p").replace("-", "m")
    path = outdir / f"ppo_lam{tag}{'_lstm' if recurrent else ''}"
    model.save(str(path))
    from .evaluate import evaluate_model
    metrics = evaluate_model(model, spec, force_lut=force_lut, lam=lam, sensing=sensing,
                             recurrent=recurrent)
    metrics.update(lam=lam, timesteps=timesteps, model_path=str(path) + ".zip")
    (outdir / f"metrics_lam{tag}.json").write_text(json.dumps(metrics, indent=2))
    return model, metrics


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train an RL controller for the many-stage PM coilgun.")
    p.add_argument("--lam", type=float, default=0.0, help="efficiency weight in reward (0=pure speed)")
    p.add_argument("--pareto", type=str, default=None,
                   help="comma-separated lambdas to sweep (e.g. 0,0.003,0.01,0.03,0.1)")
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--n-coils", type=int, default=16)
    p.add_argument("--sensing", choices=("beam_break", "perfect"), default="beam_break")
    p.add_argument("--recurrent", action="store_true", help="use RecurrentPPO (LSTM) for the POMDP")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="build/rl")
    p.add_argument("--verbose", type=int, default=1)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    spec = CoilgunSpec(n_coils=args.n_coils)
    outdir = Path(args.outdir)
    lams = [float(x) for x in args.pareto.split(",")] if args.pareto else [args.lam]
    results = []
    for lam in lams:
        print(f"[rl] training lam={lam} timesteps={args.timesteps} recurrent={args.recurrent}", flush=True)
        _, m = train_one(spec, lam, args.timesteps, outdir, n_envs=args.n_envs,
                         sensing=args.sensing, seed=args.seed, recurrent=args.recurrent,
                         verbose=args.verbose)
        print(f"[rl] lam={lam}: v={m['v']:.2f}+/-{m['v_std']:.2f} m/s  eff={m['efficiency']*100:.2f}%",
              flush=True)
        results.append(m)
    (outdir / "pareto.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
