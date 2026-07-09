import pytest

from emac_sim import PendulumParams, Tier1Estimator


def test_crossing_updates_speed_energy_and_amplitude():
    p = PendulumParams()
    est = Tier1Estimator(p)
    pulse_width = 0.02

    est.on_crossing(1.0, pulse_width)

    expected_v = p.dalpha / pulse_width
    assert est.v_last == pytest.approx(expected_v)
    assert est.energy() == pytest.approx(0.5 * p.I * expected_v * expected_v)
    assert est.amplitude() > 0.0


def test_half_period_and_direction_update_on_consecutive_crossings():
    p = PendulumParams()
    est = Tier1Estimator(p)

    est.on_crossing(1.0, 0.02)
    first_direction = est.direction
    est.on_crossing(1.6, 0.018)

    assert est.T_half == pytest.approx(0.6)
    assert est.omega == pytest.approx(3.141592653589793 / 0.6)
    assert est.direction == -first_direction


def test_damping_updates_only_on_coasting_crossings():
    p = PendulumParams()
    est = Tier1Estimator(p)

    est.on_crossing(1.0, 0.02)
    initial = est.zeta_w0
    est.on_crossing(1.6, 0.03, pulsed=True)
    assert est.zeta_w0 == pytest.approx(initial)

    est.on_crossing(2.2, 0.04, pulsed=False)
    assert est.zeta_w0 > initial


def test_prediction_after_reset_is_stable():
    p = PendulumParams()
    est = Tier1Estimator(p)

    assert est.predict(0.1) == pytest.approx((0.0, 0.0))
    est.on_crossing(1.0, 0.02)
    theta, omega = est.predict(1.1)
    assert theta != 0.0
    assert omega != 0.0
