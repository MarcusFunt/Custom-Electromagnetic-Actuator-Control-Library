import math

import pytest

from emac_sim.plant import (
    PendulumParams,
    amplitude_from_energy,
    current_for,
    energy_for_amplitude,
    f_current,
    q_shape,
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
