"""Tier-1 estimator: reconstruct continuous (theta, theta_dot) from sparse gate events.

One bottom gate gives two scalars per half-swing: peak speed (from the beam-block pulse
width) and a timestamp. A decaying oscillator has ~3 slowly-varying parameters
(amplitude, frequency, damping) -- so this tracker is exactly matched to the information
available, with far less code than a Kalman filter would need.

Key relations (docs/DESIGN.md section 2):
    theta_dot_bottom = dalpha / pulse_width          # NOTE: dalpha is an ANGLE, no /L
    E                = 1/2 * m * L^2 * theta_dot^2
    omega            = pi / T_half                    # measures the amplitude-dependent slowdown
    zeta*omega0      = ln(v_prev / v) / T_half        # log-decrement damping
"""

from __future__ import annotations
import math
from .plant import PendulumParams, amplitude_from_energy


class Tier1Estimator:
    def __init__(self, p: PendulumParams):
        self.p = p
        self.reset()

    def reset(self):
        self.have = False
        self.t_last = 0.0        # time of last bottom crossing
        self.v_last = 0.0        # |theta_dot| at that crossing
        self.E = 0.0             # J, total energy (measured at bottom)
        self.A = 0.0             # rad, amplitude
        self.omega = self.p.omega0
        self.T_half = math.pi / self.p.omega0
        self.zeta_w0 = self.p.omega0 / (2.0 * self.p.Q)   # initial damping guess
        self.direction = +1      # sign of motion just after the last crossing
        self.n = 0

    def on_crossing(self, t: float, pulse_width: float, pulsed: bool = False):
        """Fold in one photogate crossing at time t with beam-block duration pulse_width.

        `pulsed` = True if a control pulse acted during the half-swing that just ended.
        The log-decrement damping estimate is updated ONLY on coasting swings -- otherwise
        controlled braking looks like enormous natural damping and corrupts the feed-forward.
        """
        v = self.p.dalpha / pulse_width          # |theta_dot| at the bottom
        self.E = 0.5 * self.p.I * v * v
        self.A = amplitude_from_energy(self.E, self.p)

        if self.have:
            dt = t - self.t_last
            if dt > 1e-6:
                self.T_half = dt
                self.omega = math.pi / dt
                if (not pulsed) and v < self.v_last and self.v_last > 0.0 and v > 0.0:
                    self.zeta_w0 = max(0.0, math.log(self.v_last / v) / dt)
            self.direction = -self.direction     # free pendulum alternates

        self.t_last = t
        self.v_last = v
        self.have = True
        self.n += 1

    # --- continuous reconstruction between events -------------------------------------
    def predict(self, t: float):
        """Dead-reckon (theta, theta_dot) at time t via the decaying-sinusoid model,
        anchored at the last bottom crossing (theta=0, |theta_dot|=v_last)."""
        dt = t - self.t_last
        env = math.exp(-self.zeta_w0 * dt)
        s = math.sin(self.omega * dt)
        c = math.cos(self.omega * dt)
        theta = self.direction * (self.v_last / self.omega) * env * s
        omega = self.direction * self.v_last * env * (c - (self.zeta_w0 / self.omega) * s)
        return theta, omega

    def t_next_bottom(self) -> float:
        """Predicted time of the next bottom crossing (what the controller schedules on)."""
        return self.t_last + self.T_half

    # --- accessors --------------------------------------------------------------------
    def amplitude(self) -> float: return self.A
    def energy(self) -> float:    return self.E

    def damping_loss_per_half_swing(self) -> float:
        """Estimated energy lost to damping over the next half swing (feed-forward)."""
        return self.E * (1.0 - math.exp(-2.0 * self.zeta_w0 * self.T_half))
