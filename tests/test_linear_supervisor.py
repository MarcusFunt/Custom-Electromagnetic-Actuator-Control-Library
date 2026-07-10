import math

import pytest

from emac_sim.linear_estimator import LinearStepperEstimator
from emac_sim.linear_plant import CoilStation, LinearActuatorParams
from emac_sim.linear_supervisor import (
    BOOTSTRAP,
    DONE,
    FAULT,
    RUN,
    StepperSupervisor,
    _i_peak_for_energy,
    _station_k_pump,
)
from emac_sim.supervisor import current_at, envelope_average_linear


def make_estimator(p: LinearActuatorParams) -> LinearStepperEstimator:
    return LinearStepperEstimator(
        gate_positions=[g.position_m for g in p.gates],
        gate_widths=[g.w_eff for g in p.gates],
    )


def test_bootstrap_walks_forward_without_repeating_a_station():
    p = LinearActuatorParams()
    sup = StepperSupervisor(p, bootstrap_dwell_s=0.01, bootstrap_timeout_s=0.02)
    sup.start(0.0)

    seen = [sup.active.coil_index]
    t = 0.0
    for _ in range(len(p.coils) - 1):
        t += 0.03
        sup.tick(t)
        seen.append(sup.active.coil_index)

    assert seen == list(range(len(p.coils)))
    for a, b in zip(seen, seen[1:]):
        assert a != b


def test_bootstrap_exhausted_reaches_fault():
    p = LinearActuatorParams()
    sup = StepperSupervisor(p, bootstrap_dwell_s=0.01, bootstrap_timeout_s=0.02)
    sup.start(0.0)

    t = 0.0
    for _ in range(len(p.coils) + 1):
        t += 0.03
        sup.tick(t)

    assert sup.mode == FAULT
    assert current_at(t, sup.active.cmd) == 0.0


def test_first_gate_transitions_bootstrap_to_run():
    p = LinearActuatorParams()
    sup = StepperSupervisor(p)
    sup.start(0.0)
    est = make_estimator(p)
    est.on_gate(0, t=0.05, pulse_width=0.008)

    sup.on_gate(0, est, 0.05, v_tgt=0.5)

    assert sup.mode == RUN


def test_run_step_schedules_pump_ending_at_predicted_coil_arrival():
    p = LinearActuatorParams()
    sup = StepperSupervisor(p)
    sup.start(0.0)
    est = make_estimator(p)
    est.on_gate(0, t=0.0, pulse_width=0.008)   # v = 0.5 m/s

    # target above the current speed so the supervisor actually schedules a pump
    # (rather than "coast", which it would return if v_hat already met v_tgt)
    out = sup.on_gate(0, est, 0.0, v_tgt=1.0)

    # Arrival time is corrected for the accel THIS pulse itself delivers (see
    # StepperSupervisor._predict_arrival) -- not plain constant-velocity dead reckoning --
    # so recompute it from the pulse's own actually-deliverable energy, the same way
    # _run_step does, rather than asserting against est.time_to_reach() directly.
    k_quad, k_lin = sup.K_pump[0]
    dE_deliverable = k_quad * out.cmd.i_peak ** 2 + k_lin * out.cmd.i_peak
    t_arrival = sup._predict_arrival(est, p.coils[0].position_m, dE_deliverable)
    assert out.coil_index == 0
    assert out.cmd.kind == "pump"
    assert out.cmd.t1 == pytest.approx(max(0.0, t_arrival - sup.phase_advance_s))


def test_predict_arrival_caps_the_assumed_energy_at_the_slugs_own_kinetic_energy():
    """A second, independent dt-instability failure caught after the deliverable-energy
    fix above: for a light slug, even the actually-deliverable energy can still be many
    times the slug's current kinetic energy (K_pump's calibration assumes a full lobe-
    spanning pulse, which the resulting short T_p may not leave enough time to actually
    deliver -- see the module docstring). Predicting arrival from an unbounded energy
    assumption can put it absurdly early, firing the departure-repel kick while the slug
    is still well before the coil. Capping the energy used in the correction at the
    slug's own current KE (at most doubling it in one lobe pass) should make the
    prediction insensitive to further increases in the commanded energy beyond that cap."""
    p = LinearActuatorParams(mass_kg=0.01563)
    sup = StepperSupervisor(p)
    est = make_estimator(p)
    est.on_gate(0, t=0.0, pulse_width=p.gates[0].w_eff / 4.334)   # v0 = 4.334 m/s

    e0 = 0.5 * p.mass_kg * 4.334 * 4.334
    t_at_cap = sup._predict_arrival(est, p.coils[0].position_m, e0)
    t_far_beyond_cap = sup._predict_arrival(est, p.coils[0].position_m, 16.0 * e0)
    t_naive = est.time_to_reach(p.coils[0].position_m)

    assert t_at_cap == pytest.approx(t_far_beyond_cap)     # capped: no further effect
    assert t_at_cap > 0.0
    assert t_at_cap < t_naive     # still corrects toward an earlier arrival than naive


def test_coast_end_of_travel_stops_after_last_coil():
    p = LinearActuatorParams(end_of_travel="coast")
    sup = StepperSupervisor(p)
    sup.start(0.0)
    est = make_estimator(p)

    t = 0.0
    for idx in range(len(p.gates)):
        t += 0.05
        est.on_gate(idx, t=t, pulse_width=0.008)
        sup.on_gate(idx, est, t, v_tgt=0.5)

    # dead-reckon far enough past the last coil for tick() to fire end-of-travel
    last_coil_x = p.coils[-1].position_m
    t_far = est.time_to_reach(last_coil_x) + 1.0
    sup.tick(t_far)

    assert sup.mode == DONE
    assert current_at(t_far, sup.active.cmd) == 0.0


def test_brake_hold_end_of_travel_schedules_a_brake_pulse():
    p = LinearActuatorParams(end_of_travel="brake_hold")
    sup = StepperSupervisor(p)
    sup.start(0.0)
    est = make_estimator(p)

    t = 0.0
    for idx in range(len(p.gates)):
        t += 0.05
        est.on_gate(idx, t=t, pulse_width=0.008)
        sup.on_gate(idx, est, t, v_tgt=0.5)

    t_arrival = est.time_to_reach(p.coils[-1].position_m)
    sup.tick(t_arrival + 1e-6)

    assert sup.mode == DONE
    assert sup.active.cmd.kind == "brake"


def test_i_peak_for_energy_reduces_to_sqrt_for_pure_reluctance():
    dE, k_quad = 0.02, 0.05
    assert _i_peak_for_energy(dE, k_quad, 0.0, i_max=100.0) == pytest.approx(math.sqrt(dE / k_quad))


def test_i_peak_for_energy_reduces_to_linear_divide_for_pure_pm():
    dE, k_lin = 0.02, 0.05
    assert _i_peak_for_energy(dE, 0.0, k_lin, i_max=100.0) == pytest.approx(dE / k_lin)


def test_i_peak_for_energy_solves_the_combined_quadratic_for_a_hybrid_coil():
    """When both branches are active, i_peak must satisfy k_quad*i^2 + k_lin*i == dE
    (the equation being inverted), not just be some plausible-looking number."""
    dE, k_quad, k_lin = 0.03, 0.10, 0.20
    i_peak = _i_peak_for_energy(dE, k_quad, k_lin, i_max=100.0)
    assert k_quad * i_peak**2 + k_lin * i_peak == pytest.approx(dE)


def test_i_peak_for_energy_clamps_to_i_max():
    assert _i_peak_for_energy(1000.0, 0.0, 0.01, i_max=5.0) == pytest.approx(5.0)
    assert _i_peak_for_energy(1000.0, 0.05, 0.0, i_max=5.0) == pytest.approx(5.0)


def test_i_peak_for_energy_is_zero_for_nonpositive_energy_or_no_branches():
    assert _i_peak_for_energy(-1.0, 0.05, 0.05, i_max=10.0) == 0.0
    assert _i_peak_for_energy(1.0, 0.0, 0.0, i_max=10.0) == 0.0


def test_station_k_pump_quadratic_term_is_independent_of_pm_envelope():
    coil = CoilStation(position_m=0.0, x_c=0.02, Cmag=0.4, k_a=0.0)
    k_quad_rcos, _ = _station_k_pump(coil, pm_envelope="rcos")
    k_quad_square, _ = _station_k_pump(coil, pm_envelope="square")
    assert k_quad_rcos == pytest.approx(k_quad_square)


def test_station_k_pump_linear_term_scales_with_envelope_average():
    """A 'square' pump envelope has a higher time-average current than 'rcos' for the
    same i_peak (1.0 vs 0.5), so it must deliver proportionally more energy per amp --
    i.e. a bigger k_lin -- exactly matching the ratio of the two envelope averages."""
    coil = CoilStation(position_m=0.0, x_c=0.02, Cmag=0.0, k_a=0.3)
    _, k_lin_rcos = _station_k_pump(coil, pm_envelope="rcos")
    _, k_lin_square = _station_k_pump(coil, pm_envelope="square")
    expected_ratio = envelope_average_linear("square") / envelope_average_linear("rcos")
    assert k_lin_square / k_lin_rcos == pytest.approx(expected_ratio)


def test_departure_repel_is_scheduled_and_fires_after_the_approach_pump_cuts():
    """The core repel-pumping mechanism (docs/DESIGN_LINEAR.md): after gate[0] fires and
    schedules the approach-attract pump for coil[0], once dead-reckoning says coil[0]'s
    center has been reached, tick() should switch to a REPEL pulse for that SAME coil --
    not silently do nothing."""
    p = LinearActuatorParams()   # default k_a=0.20 > 0, so the PM branch is active
    sup = StepperSupervisor(p)
    sup.start(0.0)
    est = make_estimator(p)
    est.on_gate(0, t=0.0, pulse_width=0.008)   # v = 0.5 m/s

    sup.on_gate(0, est, 0.0, v_tgt=1.0)   # v_tgt above v_hat so a real pump gets scheduled
    assert sup._pending_departure is not None
    coil_index, t_arrival = sup._pending_departure
    assert coil_index == 0

    out = sup.tick(t_arrival + 1e-9)

    assert out.coil_index == 0
    assert out.cmd.polarity == "repel"
    assert out.cmd.t0 == pytest.approx(t_arrival)


def test_departure_repel_is_not_scheduled_for_a_pure_reluctance_coil():
    """Reluctance-only stations (k_a=0) can't repel at all -- attract-only regardless of
    current sign -- so no departure kick should ever be scheduled for one."""
    from emac_sim.linear_plant import GateStation

    coil = CoilStation(position_m=0.0, x_c=0.02, Cmag=0.4, k_a=0.0)
    p = LinearActuatorParams(
        coils=(coil, CoilStation(position_m=0.05, x_c=0.02, Cmag=0.4, k_a=0.0)),
        gates=(GateStation(position_m=-0.025), GateStation(position_m=0.025)),
    )
    sup = StepperSupervisor(p)
    sup.start(0.0)
    est = make_estimator(p)
    est.on_gate(0, t=0.0, pulse_width=0.008)

    sup.on_gate(0, est, 0.0, v_tgt=1.0)

    assert sup._pending_departure is None
