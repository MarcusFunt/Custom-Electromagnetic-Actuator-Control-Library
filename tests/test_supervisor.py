import pytest

from emac_sim import EnergySupervisor, PendulumParams, Tier1Estimator
from emac_sim.supervisor import PulseCmd, _envelope_shape, current_at, envelope_average_linear


def seeded_estimator(p: PendulumParams) -> Tier1Estimator:
    est = Tier1Estimator(p)
    est.on_crossing(1.0, 0.02)
    est.on_crossing(1.5, 0.021)
    return est


def test_pump_window_ends_at_next_bottom():
    p = PendulumParams()
    est = seeded_estimator(p)
    sup = EnergySupervisor(p, k_E=0.35, T_p_frac=0.30, i_max=p.i_sat)

    cmd = sup.plan(est, est.energy() * 1.2)

    assert cmd.kind == "pump"
    assert cmd.t1 == pytest.approx(est.t_next_bottom())
    assert cmd.t0 == pytest.approx(cmd.t1 - cmd.T_p)


def test_brake_window_starts_at_last_bottom():
    p = PendulumParams()
    est = seeded_estimator(p)
    sup = EnergySupervisor(p, k_E=0.35, T_p_frac=0.30, i_max=p.i_sat)

    cmd = sup.plan(est, est.energy() * 0.2)

    assert cmd.kind == "brake"
    assert cmd.t0 == pytest.approx(est.t_last)
    assert cmd.t1 == pytest.approx(cmd.t0 + cmd.T_p)


def test_current_profile_is_zero_outside_and_at_endpoints():
    p = PendulumParams()
    est = seeded_estimator(p)
    cmd = EnergySupervisor(p).plan(est, est.energy() * 1.2)

    assert current_at(cmd.t0 - 1e-6, cmd) == 0.0
    assert current_at(cmd.t0, cmd) == pytest.approx(0.0)
    assert current_at((cmd.t0 + cmd.t1) / 2.0, cmd) > 0.0
    assert current_at(cmd.t1, cmd) == pytest.approx(0.0)
    assert current_at(cmd.t1 + 1e-6, cmd) == 0.0


def test_i_peak_clamps_to_i_max():
    p = PendulumParams()
    est = seeded_estimator(p)
    sup = EnergySupervisor(p, k_E=100.0, T_p_frac=0.30, i_max=1.25)

    cmd = sup.plan(est, est.energy() * 100.0)

    assert cmd.i_peak == pytest.approx(1.25)


def test_rcos_and_sqrt_rcos_envelopes_are_zero_at_both_edges():
    """The whole reason for these two shapes: a smooth (zero-derivative-at-the-edges)
    FORCE profile to avoid exciting structural vibration -- both must start and end at
    exactly zero current, not just approximately."""
    for envelope in ("rcos", "sqrt_rcos"):
        assert _envelope_shape(envelope, 0.0) == pytest.approx(0.0)
        assert _envelope_shape(envelope, 1.0) == pytest.approx(0.0, abs=1e-9)
        assert _envelope_shape(envelope, 0.5) == pytest.approx(1.0)   # peak at midpoint


def test_square_envelope_is_flat_at_full_current_throughout():
    for phase in (0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.0):
        assert _envelope_shape("square", phase) == pytest.approx(1.0)


def test_trapezoid_envelope_ramps_linearly_then_holds_flat():
    r = 0.2   # TRAPEZOID_RAMP_FRACTION
    assert _envelope_shape("trapezoid", 0.0) == pytest.approx(0.0)
    assert _envelope_shape("trapezoid", r / 2) == pytest.approx(0.5)     # halfway up the ramp
    assert _envelope_shape("trapezoid", r) == pytest.approx(1.0)         # ramp complete
    assert _envelope_shape("trapezoid", 0.5) == pytest.approx(1.0)       # flat top
    assert _envelope_shape("trapezoid", 1.0 - r) == pytest.approx(1.0)   # ramp-down starts
    assert _envelope_shape("trapezoid", 1.0) == pytest.approx(0.0)


def test_unknown_envelope_raises():
    with pytest.raises(ValueError):
        _envelope_shape("not_a_real_envelope", 0.5)


def test_envelope_average_linear_matches_numeric_integration_of_envelope_shape():
    """envelope_average_linear's closed-form constants must match what _envelope_shape
    actually integrates to -- get this wrong and PM-branch energy sizing silently
    mis-commands current instead of just trading off smoothness (see linear_supervisor.py)."""
    n = 20_000
    for envelope in ("rcos", "sqrt_rcos", "trapezoid", "square"):
        numeric_avg = sum(_envelope_shape(envelope, k / n) for k in range(n + 1)) / (n + 1)
        assert numeric_avg == pytest.approx(envelope_average_linear(envelope), abs=1e-3)


def test_current_at_repel_is_the_negative_of_attract_for_the_same_pulse():
    cmd_attract = PulseCmd(True, "pump", 0.0, 0.02, 0.02, 4.0, 0.0, "rcos", "attract")
    cmd_repel = PulseCmd(True, "pump", 0.0, 0.02, 0.02, 4.0, 0.0, "rcos", "repel")

    for t in (0.0, 0.005, 0.01, 0.015, 0.02):
        assert current_at(t, cmd_repel) == pytest.approx(-current_at(t, cmd_attract))
    # and away from a flat zero-crossing: the midpoint current must be genuinely nonzero
    assert current_at(0.01, cmd_attract) > 0.0
