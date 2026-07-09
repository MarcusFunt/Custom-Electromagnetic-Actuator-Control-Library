import math

import pytest

from emac_sim.plant import (
    PendulumParams,
    amplitude_from_energy,
    current_for,
    energy_for_amplitude,
    f_current,
    f_current_pm,
    q_shape,
    rl_current_step,
    tau_mag,
)


def test_q_shape_is_odd_and_zero_at_bottom():
    p = PendulumParams()

    assert q_shape(0.0, p.theta_c) == pytest.approx(0.0)
    for theta in [0.01, 0.03, 0.05, 0.09]:
        assert q_shape(-theta, p.theta_c) == pytest.approx(-q_shape(theta, p.theta_c))


def test_energy_amplitude_round_trip():
    p = PendulumParams()

    for amplitude in [0.01, 0.1, 0.35, 0.8]:
        energy = energy_for_amplitude(amplitude, p)
        assert amplitude_from_energy(energy, p) == pytest.approx(amplitude)


def test_saturated_current_law_is_monotonic():
    p = PendulumParams()
    currents = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0]
    values = [f_current(i, p) for i in currents]

    assert values == sorted(values)
    assert values[-1] < p.Cmag * p.i_sat * p.i_sat


def test_current_for_exactly_inverts_feasible_saturated_force():
    p = PendulumParams()
    theta = -p.theta_c

    for current in [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0]:
        tau = tau_mag(theta, current, p)
        assert current_for(theta, tau, p) == pytest.approx(current)


def test_current_for_returns_inf_for_unattainable_force():
    p = PendulumParams()
    theta = -p.theta_c

    non_restoring_tau = -abs(tau_mag(theta, 1.0, p))
    assert math.isinf(current_for(theta, non_restoring_tau, p))
    assert math.isinf(current_for(0.0, 1e-6, p))


def test_f_current_and_q_shape_are_reused_by_any_duck_typed_coil_station():
    """Pins the reuse mechanism linear_plant.py relies on: f_current only reads
    .Cmag/.i_sat off whatever it's given, so a differently-named params object with the
    same field names works unchanged -- no signature change needed for the linear stepper."""
    from emac_sim.linear_plant import CoilStation

    coil = CoilStation(position_m=0.05, x_c=0.05, Cmag=0.010, i_sat=8.0)
    p = PendulumParams(theta_c=0.05, Cmag=0.010, i_sat=8.0)

    for current in [0.0, 0.5, 2.0, 8.0]:
        assert f_current(current, coil) == pytest.approx(f_current(current, p))
    for u in [-0.05, -0.01, 0.0, 0.02, 0.05]:
        assert q_shape(u, coil.x_c) == pytest.approx(q_shape(u, p.theta_c))


def test_rl_current_step_converges_to_v_over_r():
    """Repeatedly stepping toward a constant applied voltage must settle at the circuit's
    steady-state current V/R, regardless of dt (the exact exponential update is
    unconditionally stable, unlike explicit Euler)."""
    r, l, v = 1.2, 0.004, 12.0
    i = 0.0
    for _ in range(5000):
        i = rl_current_step(i, v, r, l, dt=1e-4)
    assert i == pytest.approx(v / r, rel=1e-6)


def test_rl_current_step_matches_the_analytic_single_step_response():
    """One big step should match the closed-form i(t) = (V/R)(1 - exp(-t/tau)) exactly --
    this IS the closed-form solution, evaluated in one shot instead of many small steps."""
    r, l, v, dt = 2.0, 0.01, 24.0, 0.003
    tau = l / r
    expected = (v / r) * (1.0 - math.exp(-dt / tau))
    assert rl_current_step(0.0, v, r, l, dt) == pytest.approx(expected)


def test_rl_current_step_decays_toward_zero_with_no_applied_voltage():
    """The freewheel/decay case: zero applied voltage lets existing current bleed off
    through the winding's own resistance, never overshooting past zero."""
    r, l = 1.2, 0.004
    i = 5.0
    prev = i
    for _ in range(50):
        i = rl_current_step(i, 0.0, r, l, dt=1e-4)
        assert 0.0 <= i <= prev
        prev = i
    assert i < 5.0


def test_rl_current_step_reaches_1_minus_1_over_e_after_one_time_constant():
    """A defining property of a first-order RL step response: after exactly tau = L/R
    seconds of constant applied voltage from rest, current should be at (1 - 1/e) of its
    final steady-state value -- checked here by taking many small steps that sum to
    exactly one tau, not by asserting a single large step (already covered by the
    closed-form single-step test above)."""
    r, l, v = 2.0, 0.01, 10.0
    tau = l / r
    n = 10_000
    dt = tau / n
    i = 0.0
    for _ in range(n):
        i = rl_current_step(i, v, r, l, dt)
    i_ss = v / r
    assert i == pytest.approx(i_ss * (1.0 - math.exp(-1.0)), rel=1e-6)


def test_f_current_pm_is_linear_and_odd():
    """The whole point of the PM branch: unlike f_current (zero for i<=0), f_current_pm
    is proportional to i for ANY sign -- attract for i>0, repel for i<0, with matching
    magnitude either way."""
    k_a = 0.35
    for i in (0.5, 1.0, 2.0, 7.0):
        assert f_current_pm(i, k_a) == pytest.approx(k_a * i)
        assert f_current_pm(-i, k_a) == pytest.approx(-f_current_pm(i, k_a))
    assert f_current_pm(2.0, k_a) == pytest.approx(2.0 * f_current_pm(1.0, k_a))
    assert f_current_pm(0.0, k_a) == 0.0
