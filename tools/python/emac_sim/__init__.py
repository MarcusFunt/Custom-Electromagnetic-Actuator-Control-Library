"""emac_sim — Phase 0 host simulator for the magnetic-pendulum control library.

Pure-Python mirror of the C++ core, so you can watch the estimator reconstruct
position and the energy-shaping supervisor pump / hold / brake against synthetic
photogate events, before any hardware exists.

Modules
-------
plant       : the physical pendulum + soft-iron (attract-only, F proportional i^2) torque map
estimator   : Tier-1 decaying-sinusoid estimator driven by Crossing events
supervisor  : energy-shaping supervisor (pump before bottom, brake after, cut at bottom)
sim         : closed-loop simulator that generates Crossing events from the true plant
"""

from .plant import PendulumParams, tau_mag, q_shape, f_current, current_for
from .estimator import Tier1Estimator
from .supervisor import EnergySupervisor, PulseCmd
from .sim import Simulator, SimLog

__all__ = [
    "PendulumParams", "tau_mag", "q_shape", "f_current", "current_for",
    "Tier1Estimator", "EnergySupervisor", "PulseCmd", "Simulator", "SimLog",
]
