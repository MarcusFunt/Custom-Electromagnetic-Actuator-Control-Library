"""Non-learned reference controllers for the coilgun, so the RL policy has an honest bar.

  1. `run_supervisor` -- the existing hand-tuned `linear_supervisor.StepperSupervisor` in its
     native `LinearSimulator` (now energy-metered), a genuine engineered baseline.
  2. `PhaseSchedulePolicy` + `optimize_schedule` -- a FIXED per-stage firing pattern (repel the
     coil behind / attract the coil ahead, each gated by the slug's phase within a pitch), whose
     handful of numbers are tuned with `scipy.differential_evolution` on the SAME env the RL
     agent trains in. This is the phase-scheduled feedforward analog of the paper's per-stage
     timing sweep -- the real thing RL must beat/robustly-match (RL additionally uses velocity
     feedback and a flexible policy).

All three report exit speed AND efficiency (KE_out / bus-source-J) via the identical
`v_applied*i*dt` metering, so the comparison is apples-to-apples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..linear_estimator import LinearStepperEstimator
from ..linear_sim import LinearSimulator
from ..linear_supervisor import FAULT, StepperSupervisor
from .env import CoilgunEnv
from .geometry import CoilgunSpec, build_params

V_TGT = 100.0     # unreachable governor target -> pure speed maximization (as in optimize_design)


def run_supervisor(spec: CoilgunSpec = CoilgunSpec(), force_lut=None, pm_envelope="square",
                   dt: float = 2e-4, t_end: float = 3.0) -> dict:
    """Exit speed + efficiency of the hand-tuned StepperSupervisor on this gun."""
    p = build_params(spec, force_lut=force_lut)
    est = LinearStepperEstimator([g.position_m for g in p.gates], [g.w_eff for g in p.gates])
    sup = StepperSupervisor(p, i_max=spec.i_max_a, pm_envelope=pm_envelope, bootstrap_timeout_s=0.20)
    sim = LinearSimulator(p, est, sup, dt=dt, sample_every=1_000_000)
    log = sim.run(x0=-0.5 * spec.pitch_m - 0.001, v0=0.0, v_tgt=V_TGT, t_end=t_end)
    v = log.gate_v[-1] if (sup.mode != FAULT and log.gate_v) else 0.0
    e_in = log.energy_in_j
    ke = 0.5 * p.mass_kg * v * v
    return {"name": "hand_tuned_supervisor", "v": v, "e_in": e_in,
            "efficiency": ke / e_in if e_in > 1e-9 else 0.0}


@dataclass
class PhaseSchedulePolicy:
    """Feedforward schedule keyed on the slug's phase within a pitch (obs[1] = `frac`): attract
    the AHEAD coil (window slot 2) when frac in [a0,a1], repel the BEHIND coil (slot 0) when frac
    in [b0,b1]. Same pattern at every stage. `params` = [a0,a1,b0,b1,i_a,i_r] (currents in 0..1
    of i_max). No velocity feedback -- that's what separates it from the RL policy."""
    params: np.ndarray
    window: int = 3

    def act(self, obs):
        a0, a1, b0, b1, i_a, i_r = self.params
        frac = float(obs[1])
        act = np.zeros(self.window, dtype=np.float32)
        if a0 <= frac <= a1:
            act[-1] = i_a               # attract ahead
        if b0 <= frac <= b1:
            act[0] = -i_r               # repel behind
        return act


def rollout(env: CoilgunEnv, policy, seed: int = 0) -> dict:
    obs, _ = env.reset(seed=seed)
    ret = 0.0
    while True:
        obs, r, term, trunc, info = env.step(policy.act(obs))
        ret += r
        if term or trunc:
            break
    ke = info["ke"]
    return {"v": info["v"], "e_in": info["e_in"], "efficiency": info["efficiency"],
            "return": ret, "reason": info["reason"], "ke": ke}


def evaluate(env: CoilgunEnv, policy, seeds=range(8)) -> dict:
    """Mean metrics over several initial-velocity seeds (robustness)."""
    rs = [rollout(env, policy, seed=s) for s in seeds]
    return {"v": float(np.mean([r["v"] for r in rs])),
            "v_std": float(np.std([r["v"] for r in rs])),
            "efficiency": float(np.mean([r["efficiency"] for r in rs])),
            "e_in": float(np.mean([r["e_in"] for r in rs])),
            "return": float(np.mean([r["return"] for r in rs]))}


def optimize_schedule(spec: CoilgunSpec = CoilgunSpec(), force_lut=None, lam: float = 0.0,
                      seeds=range(4), maxiter: int = 40, popsize: int = 12, seed: int = 0,
                      **env_kw) -> dict:
    """Tune the 6 PhaseSchedulePolicy numbers with differential evolution to maximize the mean
    env return at this lambda. Returns the best policy + its evaluated metrics."""
    from scipy.optimize import differential_evolution

    env = CoilgunEnv(spec=spec, force_lut=force_lut, lam=lam, **env_kw)
    bounds = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (0.0, 1.0), (0.0, 1.0)]

    def neg_return(x):
        pol = PhaseSchedulePolicy(np.asarray(x))
        return -float(np.mean([rollout(env, pol, seed=s)["return"] for s in seeds]))

    res = differential_evolution(neg_return, bounds, maxiter=maxiter, popsize=popsize,
                                 seed=seed, polish=True, tol=1e-3)
    pol = PhaseSchedulePolicy(res.x)
    metrics = evaluate(env, pol, seeds=range(8))
    return {"name": "optimized_schedule", "params": res.x.tolist(), **metrics}
