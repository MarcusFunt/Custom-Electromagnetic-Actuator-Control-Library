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



def f_current_pm(i: float, k_a: float) -> float:
    """Permanent-magnet branch current law (docs/DESIGN.md section 3.3): linear and
    SIGNED, unlike the soft-iron branch above -- i>0 attracts, i<0 repels. Shared/reusable
    wherever a coil couples to a magnetized (not just reluctance) target; see
    linear_plant.py's hybrid reluctance+PM slug for the linear stepper's use of it."""
    return k_a * i



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



def conservative_alpha(theta: float, i: float, p: PendulumParams) -> float:
    """Angular acceleration from position-dependent torques only.

    This excludes viscous damping so the integrator can apply the linear damping part as
    an exact exponential split.  Keeping the conservative term separate also makes energy
    checks clearer: with i=0 and Q=inf/very large, this is the nonlinear pendulum ODE.
    """
    tau = -p.m * p.g * p.L * math.sin(theta) + tau_mag(theta, i, p)
    return tau / p.I



def alpha(theta: float, omega: float, i: float, p: PendulumParams) -> float:
    """Angular acceleration = net torque / inertia, including linear viscous damping."""
    return conservative_alpha(theta, i, p) - (p.b / p.I) * omega



def step(theta: float, omega: float, i: float, dt: float, p: PendulumParams):
    """One damped velocity-Verlet step with exact linear-damping splitting.

    The previous integrator was semi-implicit Euler: robust and symplectic for the
    undamped oscillator, but only first-order accurate.  This update uses a kick-drift-kick
    velocity-Verlet core (second-order for the conservative pendulum+magnet torques) and
    wraps it in half-step exact exponential damping.  Current is still treated as
    piecewise-constant over the tick, matching the supervisor's sampled-current model.
    """
    if dt == 0.0:
        return theta, omega

    gamma = p.b / p.I if p.I > 0.0 else 0.0
    damp_half = math.exp(-0.5 * gamma * dt) if gamma > 0.0 else 1.0

    omega_d = omega * damp_half
    a0 = conservative_alpha(theta, i, p)
    omega_half = omega_d + 0.5 * a0 * dt
    theta_new = theta + omega_half * dt
    a1 = conservative_alpha(theta_new, i, p)
    omega_new = (omega_half + 0.5 * a1 * dt) * damp_half
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



def rl_current_step(i: float, v_applied: float, r: float, l: float, dt: float) -> float:
    """Exact update for a first-order RL circuit (L di/dt = v_applied - i*r), assuming
    v_applied is piecewise-constant over dt. Coordinate/geometry-agnostic (not pendulum- or
    linear-stepper-specific) -- shared here so either model can use it once it needs real
    electrical dynamics instead of an idealized instantaneous current source.

    Uses the closed-form exponential solution rather than explicit Euler: unconditionally
    stable regardless of how dt compares to the L/R time constant, which matters once
    inductance is small relative to the simulation step.
    """
    tau = l / r
    i_ss = v_applied / r        # steady-state current this voltage would settle to
    return i_ss + (i - i_ss) * math.exp(-dt / tau)
