# RL control of the many-stage PM coilgun (`emac_sim.rl`)

A reinforcement-learning controller for a **fixed 15–20-stage permanent-magnet coilgun**, each
stage an independent **H-bridge**, sensed by **IR beam-breaks** between stages, optimizing a
tunable blend of **exit speed and energy efficiency**. The learned policy is a drop-in
replacement for the hand-tuned `linear_supervisor.StepperSupervisor`: it commands per-coil
**current targets** every control tick from the beam-break-estimated state, and the existing
electrical/mechanical/thermal model (`linear_plant`, `linear_sim`) executes them.

## Why this is (mostly) reuse
The plant already supports arbitrary stages, per-stage H-bridge current dynamics with motional
back-EMF (energy-conserving to ~1e-9) and per-coil thermal `i²R`, and the gates + estimator
already *are* the IR beam-break sensor. `coil_current_step` already turns a commanded current
target into rail-limited real dynamics. So only the **policy** and the **reward** are new.

## The pieces
| module | what |
|---|---|
| `rl/geometry.py` | `CoilgunSpec` + `build_params` — the fixed gun (default 16 stages, small coils, 450 V / 100 A H-bridge). One position-independent force law serves every identical stage; `build_reference_lut` gives a no-FEMM swept table (or pass a real-FEMM `ForceLUT`). |
| `rl/env.py` | `CoilgunEnv` (Gymnasium). **Per-tick control** made tractable by a **local, translation-invariant window**: the action is the current target for the `window` coils straddling the slug (behind/current/ahead); the observation is the **beam-break estimate** (dead-reckoned x, last-gate v, time-since-gate, confidence) + local coil currents/temps, all relative to the slug. Reward = `dKE − λ·dE_elec` (telescopes to `KE_final − λ·E_in`), normalized. |
| `rl/baselines.py` | `run_supervisor` (the hand-tuned controller, the real bar), and `PhaseSchedulePolicy` + `optimize_schedule` (a feedforward phase pattern tuned with `scipy.differential_evolution`). |
| `rl/train.py` | `emac-rl-train` — SB3 **PPO** (or `--recurrent` RecurrentPPO for the POMDP); `--pareto` sweeps λ. |
| `rl/evaluate.py` | roll out a trained policy over seeds, compare to baselines, and `plot_pareto`. |

## Install & run
```
pip install -e .[rl]                       # gymnasium + stable-baselines3 + sb3-contrib (torch already present)
emac-rl-train --lam 0.0 --timesteps 300000                 # one policy, pure speed
emac-rl-train --pareto 0,0.003,0.01,0.03,0.1 --timesteps 300000   # the speed/efficiency frontier
```
Models + `metrics_lam*.json` + `pareto.json` land in `--outdir` (default `build/rl`).

## What "efficiency" means
`efficiency = ½·m·v_exit² / E_in`, where `E_in = Σ v_applied·i·dt` is the **bus source energy**
(signed — an H-bridge regenerates on a hard cut). The same metering is used for the RL policy,
the supervisor, and the schedule (`LinearSimulator` now logs `energy_in_j`), so the comparison
is apples-to-apples.

## Honest expectations
- The **hand-tuned supervisor is a strong baseline** — on the 16-stage / 450 V / 100 A gun it
  already reaches ~37 m/s at ~14% efficiency (well above the single-config sweep's ~26 m/s, and
  ~10× the reluctance papers' ~1.3% — at far lower energy, so not a like-for-like speed claim).
  RL's job is to **beat/robustly-match that**, chiefly on efficiency and on robustness to a
  randomized initial velocity the open-loop schedule can't adapt to.
- **This is a genuine POMDP** (beam-break-only position + per-tick control): a feedforward MLP
  may plateau; `--recurrent` (LSTM) is the fallback.
- **450 V / 100 A caps speed** far below the papers' ~3600 A regime — the headline to chase here
  is **efficiency** and beating our own PM results, not the papers' absolute velocity.

Later phases: co-optimize the geometry with the control, a fast-cutoff/energy-recycling driver,
and higher-current regimes.
