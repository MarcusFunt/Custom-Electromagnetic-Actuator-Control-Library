import math

import pytest

from emac_sim.linear_estimator import STALL_SUSPECT, TRACKING, LinearStepperEstimator


GATE_POSITIONS = [-0.025, 0.025, 0.075, 0.125, 0.175]
GATE_WIDTHS = [0.004] * 5


def make_estimator(**overrides) -> LinearStepperEstimator:
    return LinearStepperEstimator(GATE_POSITIONS, GATE_WIDTHS, **overrides)


def test_on_gate_computes_velocity_from_pulse_width():
    est = make_estimator()
    pulse_width = 0.008
    accepted = est.on_gate(0, t=1.0, pulse_width=pulse_width)

    assert accepted
    assert est.status == TRACKING
    assert est.v_last == pytest.approx(GATE_WIDTHS[0] / pulse_width)
    assert est.x_last == pytest.approx(GATE_POSITIONS[0])
    assert est.next_expected == 1


def test_out_of_order_gate_is_flagged_not_accepted():
    est = make_estimator()
    est.on_gate(0, t=1.0, pulse_width=0.008)

    accepted = est.on_gate(2, t=1.05, pulse_width=0.006)   # skipped gate 1

    assert not accepted
    assert est.status == STALL_SUSPECT
    assert est.next_expected == 1     # state from the last ACCEPTED gate is unchanged


def test_repeated_gate_index_is_rejected():
    est = make_estimator()
    est.on_gate(0, t=1.0, pulse_width=0.008)

    accepted = est.on_gate(0, t=1.05, pulse_width=0.008)   # same gate again

    assert not accepted
    assert est.status == STALL_SUSPECT


def test_predict_coasts_at_constant_velocity_between_gates():
    est = make_estimator()
    est.on_gate(0, t=1.0, pulse_width=0.008)   # v = 0.5 m/s
    x, v = est.predict(1.1)

    assert v == pytest.approx(0.5)
    assert x == pytest.approx(GATE_POSITIONS[0] + 0.5 * 0.1)


def test_cleared_last_gate_after_final_crossing():
    est = make_estimator()
    for idx in range(5):
        est.on_gate(idx, t=1.0 + idx * 0.1, pulse_width=0.008)

    assert est.cleared_last_gate()
    assert est.t_next_gate() is None


def test_confidence_decays_with_staleness():
    est = make_estimator(tau_confidence=0.05)
    est.on_gate(0, t=1.0, pulse_width=0.008)

    assert est.confidence(1.0) == pytest.approx(1.0)
    assert est.confidence(1.5) < est.confidence(1.0)


def test_update_status_flags_stall_when_next_gate_is_overdue():
    est = make_estimator(stall_factor=2.0)
    est.on_gate(0, t=1.0, pulse_width=0.008)   # v = 0.5 m/s, expects gate 1 at x=0.025

    t_next = est.t_next_gate()
    assert t_next is not None

    est.update_status(t_next + 2.0 * (t_next - 1.0) + 1e-3)
    assert est.status == STALL_SUSPECT
