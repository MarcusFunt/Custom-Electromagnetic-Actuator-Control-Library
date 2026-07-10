"""emac_sim — Phase 0 host simulator for the electromagnetic actuator control library.

Pure-Python mirror of the C++ core, so you can watch an estimator reconstruct position
and a supervisor pump / hold / brake against synthetic photogate events, before any
hardware exists. Two geometries share the same primitives (q_shape/f_current, PulseCmd/
current_at) but have distinct concrete estimator/supervisor/simulator implementations,
selected via config (see config.parse_config's `[sim] kind`) -- not a forced common class
hierarchy, since their control laws are genuinely different (see docs/DESIGN_LINEAR.md).

Modules (pendulum -- bounded oscillator, one bottom coil)
----------------------------------------------------------
plant       : the physical pendulum + soft-iron (attract-only, F proportional i^2) torque map
estimator   : Tier-1 decaying-sinusoid estimator driven by Crossing events
supervisor  : energy-shaping supervisor (pump before bottom, brake after, cut at bottom)
sim         : closed-loop simulator that generates Crossing events from the true plant

Modules (linear stepper -- finite one-way N-coil actuator)
------------------------------------------------------------
linear_plant       : translational reluctance plant, N coil stations, no restoring term
linear_estimator   : position/velocity dead-reckoning from an ordered gate-crossing sequence
linear_supervisor  : forward-only commutation supervisor with a startup bootstrap FSM
linear_sim         : closed-loop simulator, linear analog of sim.py
"""

from .plant import (
    PendulumParams,
    tau_mag,
    q_shape,
    f_current,
    f_current_pm,
    current_for,
    conservative_alpha,
    rl_current_step,
)
from .estimator import Tier1Estimator
from .numerics import hermite_event_fraction
from .supervisor import EnergySupervisor, PulseCmd, current_at, envelope_average_linear
from .sim import Simulator, SimLog

from .linear_plant import LinearActuatorParams, CoilStation, GateStation, coil_current_step, undamped_accel
from .linear_estimator import LinearStepperEstimator
from .linear_supervisor import StepperSupervisor, StepperOutput
from .linear_sim import LinearSimulator, LinearSimLog

__all__ = [
    "PendulumParams", "tau_mag", "q_shape", "f_current", "f_current_pm", "current_for",
    "conservative_alpha", "rl_current_step", "hermite_event_fraction",
    "Tier1Estimator", "EnergySupervisor", "PulseCmd", "current_at", "envelope_average_linear",
    "Simulator", "SimLog",
    "LinearActuatorParams", "CoilStation", "GateStation", "coil_current_step", "undamped_accel",
    "LinearStepperEstimator", "StepperSupervisor", "StepperOutput",
    "LinearSimulator", "LinearSimLog",
]
