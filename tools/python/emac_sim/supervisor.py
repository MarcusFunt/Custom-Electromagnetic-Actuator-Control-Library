"""Energy-shaping supervisor: decide how hard and WHEN to fire the coil.

Control law (docs/DESIGN.md section 4):
    dE_cmd = dE_damp_ff + k_E * (E_tgt - E)          # replace loss + close a fraction of error

Timing:
    dE_cmd > 0  -> PUMP : attract pulse in [t* - T_p, t*]  (approach half), cut at bottom t*
    dE_cmd < 0  -> BRAKE: attract pulse in [t*, t* + T_p]  (departure half)  [soft-iron: timing only]

Because the torque shape q(theta) is odd (zero at the bottom), current lingering past t*
subtracts the work -- hence the hard cut. The force is shaped as a raised cosine (so
F ~ i^2 is band-limited) to avoid exciting structural / string vibration.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from .plant import PendulumParams, q_shape
from .estimator import Tier1Estimator


@dataclass
class PulseCmd:
    active: bool          # False -> coast this swing (HOLD dead-band)
    kind: str             # "pump" | "brake" | "coast"
    t0: float             # window start (s)
    t1: float             # window end   (s); for a pump this equals t* (hard cut)
    T_p: float            # window width (s)
    i_peak: float         # A, peak coil current
    dE_cmd: float         # J, commanded energy delta (signed)


def _q_window_integral(p: PendulumParams, a: float, b: float, n: int = 200) -> float:
    """Integral of |q(theta)| dtheta over an angular window [a, b] (trapezoid)."""
    h = (b - a) / n
    s = 0.5 * (abs(q_shape(a, p.theta_c)) + abs(q_shape(b, p.theta_c)))
    for k in range(1, n):
        s += abs(q_shape(a + k * h, p.theta_c))
    return abs(s * h)


class EnergySupervisor:
    def __init__(self, p: PendulumParams, k_E: float = 0.30,
                 T_p_frac: float = 0.30, i_max: float = 8.0,
                 eps_frac: float = 0.02):
        self.p = p
        self.k_E = k_E
        self.T_p_frac = T_p_frac          # window width as a fraction of a half-period
        self.i_max = i_max
        self.eps_frac = eps_frac          # HOLD dead-band as a fraction of E_tgt
        # Energy delivered per A^2 for a full approach-lobe pulse:  dE ~ Cmag * i^2 * Qwin.
        # The raised-cosine force envelope delivers ~0.5 of a flat pulse, folded in here.
        Qwin = _q_window_integral(p, -p.theta_c, 0.0)
        self.K_pump = 0.5 * p.Cmag * Qwin     # J per A^2

    def plan(self, est: Tier1Estimator, E_tgt: float) -> PulseCmd:
        E = est.energy()
        T_half = est.T_half
        T_p = self.T_p_frac * T_half
        t_star = est.t_next_bottom()   # next crossing -> approach window ends here (PUMP)
        t_dep = est.t_last             # crossing just seen -> departure window starts here (BRAKE)

        dE_ff = est.damping_loss_per_half_swing()          # feed-forward the loss (kills droop)
        diff = E_tgt - E
        eps = self.eps_frac * max(E_tgt, 1e-9)
        if abs(diff) < eps:
            dE_cmd = dE_ff                                  # holding: just replace the loss
        else:
            dE_cmd = dE_ff + self.k_E * diff

        if abs(dE_cmd) < 1e-6:
            return PulseCmd(False, "coast", 0.0, 0.0, T_p, 0.0, 0.0)

        i_peak = min(math.sqrt(abs(dE_cmd) / max(self.K_pump, 1e-12)), self.i_max)

        if dE_cmd >= 0.0:
            # PUMP: attract on the approach to the next bottom, cut AT the bottom.
            return PulseCmd(True, "pump", t_star - T_p, t_star, T_p, i_peak, dE_cmd)
        else:
            # BRAKE: attract on the departure from the bottom we just crossed. Scheduling
            # it on the current departure (not after the next crossing) is essential --
            # otherwise the next crossing's re-plan overwrites it before it ever fires.
            return PulseCmd(True, "brake", t_dep, t_dep + T_p, T_p, i_peak, dE_cmd)


def current_at(t: float, cmd: PulseCmd) -> float:
    """Instantaneous coil current from a scheduled pulse, with a raised-cosine FORCE
    envelope: i(t) = i_peak * sqrt(0.5 (1 - cos(2*pi*(t-t0)/T_p))) so that F ~ i^2 is smooth."""
    if not cmd.active or t < cmd.t0 or t > cmd.t1 or cmd.T_p <= 0.0:
        return 0.0
    phase = (t - cmd.t0) / cmd.T_p
    f_env = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
    return cmd.i_peak * math.sqrt(max(0.0, f_env))
