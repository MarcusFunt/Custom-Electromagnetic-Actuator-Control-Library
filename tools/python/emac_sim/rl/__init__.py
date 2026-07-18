"""Reinforcement-learning control of the many-stage PM coilgun (docs/RL_CONTROL.md).

A learned controller drop-in for the hand-tuned `linear_supervisor.StepperSupervisor`: it
commands per-coil CURRENT TARGETS every control tick from the IR-beam-break-sensed state, to
maximize a tunable blend of exit speed and energy efficiency. Reuses the whole physical
substrate (linear_plant / linear_estimator / linear_sim electrical + thermal model); only the
control policy and the reward are new.

Submodules:
  - geometry: build the fixed 15-20-stage LinearActuatorParams (+ optional shared force LUT).
  - env:      CoilgunEnv, a Gymnasium environment wrapping the stepwise sim.
  - baselines: hand-tuned-supervisor and optimized-open-loop-schedule reference controllers.
  - train / evaluate: Stable-Baselines3 training, the speed/efficiency Pareto sweep, plots.

Optional deps (install with `pip install -e .[rl]`): gymnasium, stable-baselines3, sb3-contrib.
The env itself needs only gymnasium + numpy; torch/SB3 are needed only to train.
"""
