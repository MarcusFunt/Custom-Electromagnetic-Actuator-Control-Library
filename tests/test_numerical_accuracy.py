import math

import pytest

from emac_sim.linear_plant import LinearActuatorParams, step as linear_step
from emac_sim.numerics import hermite_event_fraction
from emac_sim.plant import PendulumParams, energy, step as pendulum_step



def test_hermite_event_fraction_is_exact_for_constant_acceleration_crossing():
    """A constant-acceleration path is quadratic, so cubic Hermite interpolation should
    locate its event time exactly from endpoint positions and velocities.
    """
    y0, v0, accel, dt = -1.0, 1.0, 2.0, 1.0
    y1 = y0 + v0 * dt + 0.5 * accel * dt * dt
    v1 = v0 + accel * dt

    frac, v_cross = hermite_event_fraction(y0, v0, y1, v1, dt)
    expected_t = (-v0 + math.sqrt(v0 * v0 - 2.0 * accel * y0)) / accel

    assert frac == pytest.approx(expected_t / dt, rel=1e-13, abs=1e-13)
    assert v_cross == pytest.approx(v0 + accel * expected_t, rel=1e-13, abs=1e-13)



def test_linear_step_applies_damping_as_exact_exponential_when_no_force_is_present():
    p = LinearActuatorParams(
        mass_kg=0.10,
        damping_n_per_mps=0.25,
        coils=(),
        gates=(),
        pressure_bias_n=0.0,
    )
    x0, v0, dt = 0.0, 1.5, 0.04

    _, v = linear_step(x0, v0, [], dt, p)

    gamma = p.damping_n_per_mps / p.mass_kg
    assert v == pytest.approx(v0 * math.exp(-gamma * dt), rel=1e-14)



def test_pendulum_step_has_small_energy_error_for_undriven_undamped_swing():
    """The host plant should be a useful physics reference, not just a controller smoke
    test.  With no coil current and negligible damping, the nonlinear pendulum's mechanical
    energy should stay nearly constant over several periods.
    """
    p = PendulumParams(Q=1e30)
    theta, omega = 0.35, 0.0
    dt = 5e-4
    period = 2.0 * math.pi / p.omega0
    n_steps = int(5.0 * period / dt)
    e0 = energy(theta, omega, p)

    max_rel_error = 0.0
    for _ in range(n_steps):
        theta, omega = pendulum_step(theta, omega, 0.0, dt, p)
        max_rel_error = max(max_rel_error, abs(energy(theta, omega, p) - e0) / e0)

    assert max_rel_error < 1e-4
