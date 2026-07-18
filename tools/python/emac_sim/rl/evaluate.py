"""Evaluate + compare coilgun controllers: roll out a trained RL policy, the hand-tuned
supervisor, and the optimized schedule, and plot the speed/efficiency Pareto frontier.
"""
from __future__ import annotations

import numpy as np

from .env import CoilgunEnv
from .geometry import CoilgunSpec


def _rollout_model(model, env: CoilgunEnv, seed: int, recurrent: bool):
    obs, _ = env.reset(seed=seed)
    state, done = None, True
    while True:
        if recurrent:
            action, state = model.predict(obs, state=state,
                                          episode_start=np.array([done]), deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        if done:
            break
    return info


def evaluate_model(model, spec: CoilgunSpec = CoilgunSpec(), force_lut=None, lam: float = 0.0,
                   sensing: str = "beam_break", recurrent: bool = False, seeds=range(16)) -> dict:
    """Mean exit speed + efficiency of a trained model over several initial-velocity seeds."""
    env = CoilgunEnv(spec=spec, force_lut=force_lut, lam=lam, sensing=sensing)
    infos = [_rollout_model(model, env, s, recurrent) for s in seeds]
    v = np.array([i["v"] for i in infos]); eff = np.array([i["efficiency"] for i in infos])
    exited = np.mean([i["reason"] == "exit" for i in infos])
    return {"name": "rl_ppo", "v": float(v.mean()), "v_std": float(v.std()),
            "efficiency": float(eff.mean()), "e_in": float(np.mean([i["e_in"] for i in infos])),
            "exit_rate": float(exited)}


def compare_all(spec: CoilgunSpec = CoilgunSpec(), force_lut=None, model=None, lam: float = 0.0,
                sensing: str = "beam_break", recurrent: bool = False) -> list[dict]:
    """Supervisor + optimized-schedule baselines (+ the RL model if given), as a list of rows."""
    from . import baselines as B
    rows = [B.run_supervisor(spec, force_lut=force_lut)]
    rows.append(B.optimize_schedule(spec, force_lut=force_lut, lam=lam))
    if model is not None:
        rows.append(evaluate_model(model, spec, force_lut=force_lut, lam=lam, sensing=sensing,
                                   recurrent=recurrent))
    return rows


def plot_pareto(rl_rows: list[dict], baselines: list[dict], out_png: str,
                title: str = "Coilgun control: speed vs efficiency") -> None:
    """Scatter of exit speed (x) vs efficiency (y): the RL Pareto points + baseline references."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if rl_rows:
        v = [r["v"] for r in rl_rows]; e = [100 * r["efficiency"] for r in rl_rows]
        order = np.argsort(v)
        ax.plot(np.array(v)[order], np.array(e)[order], "-o", color="#4C72B0", lw=2,
                markersize=8, label="RL (PPO), swept λ", zorder=3)
        for r in rl_rows:
            ax.annotate(f"λ={r.get('lam','?')}", (r["v"], 100 * r["efficiency"]),
                        fontsize=8, color="#4C72B0", xytext=(4, 4), textcoords="offset points")
    marks = {"hand_tuned_supervisor": ("#C44E52", "s", "hand-tuned supervisor"),
             "optimized_schedule": ("#DD8452", "^", "optimized schedule")}
    for b in baselines:
        c, m, lab = marks.get(b["name"], ("#777", "x", b["name"]))
        ax.scatter([b["v"]], [100 * b["efficiency"]], c=c, marker=m, s=110, zorder=4, label=lab,
                   edgecolors="white", linewidths=1.2)
    ax.set_xlabel("exit speed (m/s)"); ax.set_ylabel("efficiency  KE_out / bus-J  (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
