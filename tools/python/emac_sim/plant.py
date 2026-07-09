"""Physical plant: pendulum dynamics + the separable soft-iron magnetic torque map.

Torque model (see docs/DESIGN.md section 3):

    tau_mag(theta, i) = q(theta) * f(i)

  q(theta)  -- ODD, zero at bottom-center, one lobe each side peaking at +/- theta_c,
               decaying outside.  This is the crucial fact: energizing the coil over a
               symmetric window does ZERO net work; you pump by using the approach half
               and cutting at the bottom.
  f(i)      -- soft-iron / reluctance: ATTRACT-ONLY and quadratic, f = Cmag * i^2 with
               saturation.  Reversing current does nothing; energy is managed by timing.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class PendulumParams:
    g: float = 9.81          # m/s^2
    L: float = 0.30          # m, effective pendulum length
    m: float = 0.05          # kg, effective bob mass
    Q: float = 200.0         # quality factor -> sets viscous damping b
    theta_c: float = 0.05    # rad, torque-lobe location (coil coupling half-width)
    Cmag: float = 0.010      # N*m per A^2 at the lobe peak (combined q*k_r force scale)
    i_sat: float = 8.0       # A, soft-iron saturation current
    dalpha: float = 0.060    # rad, effective angular width of the bob at the gate

    @property
    def I(self) -> float:            # moment of inertia (point-mass bob)
        return self.m * self.L * self.L

    @property
    def omega0(self) -> float:       # small-amplitude natural frequency
        return math.sqrt(self.g / self.L)

    @property
    def b(self) -> float:            # viscous damping coeff (torque per rad/s)
        return self.I * self.omega0 / self.Q


def q_shape(theta: float, theta_c: float) -> float:
    """Odd, RESTORING torque shape: 0 at theta=0, magnitude 1 at +/- theta_c, decays outside.

    The attractive coil pulls the bob toward the bottom, so the torque is restoring:
    NEGATIVE for theta>0, POSITIVE for theta<0. With i>0 the torque tau=q*f is therefore
    always toward center -> it PUMPS when the bob approaches and BRAKES when it departs,
    which is what the timing logic exploits.
    """
    u = theta / theta_c
    return -u * math.exp(0.5) * math.exp(-0.5 * u * u)


def f_current(i: float, p: PendulumParams) -> float:
    """Attract-only, quadratic-with-saturation current law. Returns a force scale >= 0."""
    if i <= 0.0:
        return 0.0
    return p.Cmag * (i * i) / (1.0 + (i / p.i_sat) ** 2)


def tau_mag(theta: float, i: float, p: PendulumParams) -> float:
    """Magnetic torque about the pivot. Always RESTORING (toward center) when i>0."""
    return q_shape(theta, p.theta_c) * f_current(i, p)


def current_for(theta: float, tau_desired: float, p: PendulumParams) -> float:
    """Inverse map: current needed to produce tau_desired at angle theta.

    Soft-iron is attract-only, so this is only feasible when tau_desired is RESTORING
    (same sign as q(theta)). Otherwise the coil cannot deliver it -> return math.inf.
    This exactly inverts f_current's saturating law for finite feasible currents.
    """
    if tau_desired == 0.0:
        return 0.0

    q = q_shape(theta, p.theta_c)
    if q == 0.0:
        return math.inf

    f_desired = tau_desired / q
    if f_desired <= 0.0 or p.Cmag <= 0.0 or p.i_sat <= 0.0:
        return math.inf

    max_force_scale = p.Cmag * p.i_sat * p.i_sat
    if f_desired >= max_force_scale:
        return math.inf

    y = f_desired / p.Cmag
    denom = 1.0 - (y / (p.i_sat * p.i_sat))
    return math.sqrt(y / denom)


def alpha(theta: float, omega: float, i: float, p: PendulumParams) -> float:
    """Angular acceleration = net torque / inertia."""
    tau = -p.m * p.g * p.L * math.sin(theta) - p.b * omega + tau_mag(theta, i, p)
    return tau / p.I


def step(theta: float, omega: float, i: float, dt: float, p: PendulumParams):
    """One semi-implicit (symplectic) Euler step. Conserves energy far better than
    explicit Euler, which is essential for a lightly-damped oscillator."""
    a = alpha(theta, omega, i, p)
    omega_new = omega + a * dt
    theta_new = theta + omega_new * dt
    return theta_new, omega_new


def energy(theta: float, omega: float, p: PendulumParams) -> float:
    """Total mechanical energy: kinetic + gravitational potential (zero at bottom)."""
    return 0.5 * p.I * omega * omega + p.m * p.g * p.L * (1.0 - math.cos(theta))


def amplitude_from_energy(E: float, p: PendulumParams) -> float:
    """Turning-point amplitude for a given total energy (large-angle exact)."""
    c = 1.0 - E / (p.m * p.g * p.L)
    c = max(-1.0, min(1.0, c))
    return math.acos(c)


def energy_for_amplitude(A: float, p: PendulumParams) -> float:
    return p.m * p.g * p.L * (1.0 - math.cos(A))
