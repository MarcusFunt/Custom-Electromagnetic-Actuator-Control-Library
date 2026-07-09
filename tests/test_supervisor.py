import pytest

from emac_sim import EnergySupervisor, PendulumParams, Tier1Estimator
from emac_sim.supervisor import current_at


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
