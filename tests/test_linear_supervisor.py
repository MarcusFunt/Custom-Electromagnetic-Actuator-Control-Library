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
    # so recompute it from the pulse's own actually-deliverable energy and the coil's own
    # electrical time constant, the same way _run_step does, rather than asserting
    # against est.time_to_reach() directly.
    k_quad, k_lin = sup.K_pump[0]
    dE_deliverable = k_quad * out.cmd.i_peak ** 2 + k_lin * out.cmd.i_peak
    coil = p.coils[0]
    tau_elec = coil.inductance_h / coil.resistance_ohm
    t_arrival = sup._predict_arrival(est, p.coils[0].position_m, dE_deliverable, tau_elec)
    assert out.coil_index == 0
    assert out.cmd.kind == "pump"
    assert out.cmd.t1 == pytest.approx(max(0.0, t_arrival - sup.phase_advance_s))


def test_predict_arrival_without_tau_elec_applies_the_full_energy_correction():
    """With no tau_elec given (the caller doesn't know the coil's electrical time
    constant), the correction should behave exactly as the plain SUVAT estimate --
    no throttling at all, regardless of how large dE_cmd is relative to current KE.
    (A KE-based cap used to sit here; it was removed after it broke the bootstrap-
    acceleration case below -- see _predict_arrival's docstring.)"""
    p = LinearActuatorParams(mass_kg=0.01563)
    sup = StepperSupervisor(p)
    est = make_estimator(p)
    est.on_gate(0, t=0.0, pulse_width=p.gates[0].w_eff / 4.334)   # v0 = 4.334 m/s

    e0 = 0.5 * p.mass_kg * 4.334 * 4.334
    dE = 16.0 * e0
    v0 = 4.334
    v1 = math.sqrt(v0 * v0 + 2.0 * dE / p.mass_kg)
    v_eff = 0.5 * (v0 + v1)
    expected = est.t_last + (p.coils[0].position_m - est.x_last) / v_eff

    t_uncorrected = sup._predict_arrival(est, p.coils[0].position_m, dE)
    assert t_uncorrected == pytest.approx(expected)


def test_predict_arrival_with_tau_elec_barely_throttles_a_bootstrap_acceleration_pump():
    """The case that broke the old KE-based cap: a slug near rest (tiny current KE)
    about to receive a strong first pump that legitimately injects many multiples of
    that KE. Because the resulting transit is long relative to a normal coil's
    electrical time constant, the coil has plenty of time to actually reach the
    assumed current -- so the correction should barely be throttled at all, staying
    close to the naive dead-reckoned prediction's opposite extreme (much earlier
    than plain constant-velocity dead reckoning, not clamped back toward it)."""
    p = LinearActuatorParams(mass_kg=0.01563)
    sup = StepperSupervisor(p)
    est = make_estimator(p)
    est.on_gate(0, t=0.0, pulse_width=p.gates[0].w_eff / 1.19)   # v0 = 1.19 m/s, near rest

    e0 = 0.5 * p.mass_kg * 1.19 * 1.19
    dE = 205.0 * e0    # matches the ratio observed in the design that exposed this
    tau_elec = 0.0005036452313737064   # a real coil's L/R from this session's diagnosis

    t_corrected = sup._predict_arrival(est, p.coils[0].position_m, dE, tau_elec)
    t_naive = est.time_to_reach(p.coils[0].position_m)
    assert t_corrected < 0.7 * t_naive     # a real, large correction -- not clamped away


def test_predict_arrival_with_tau_elec_throttles_an_electrically_infeasible_pump():
    """The opposite regime: an assumed energy injection so large, over so short a
    transit, that the coil's own electrical time constant genuinely can't keep up
    (T_p ends up only a fraction of tau_elec). The correction should throttle back
    toward the naive prediction rather than trusting energy that could never really
    be delivered that fast."""
    p = LinearActuatorParams(mass_kg=0.01563)
    sup = StepperSupervisor(p)
    est = make_estimator(p)
    est.on_gate(0, t=0.0, pulse_width=p.gates[0].w_eff / 4.334)   # v0 = 4.334 m/s

    e0 = 0.5 * p.mass_kg * 4.334 * 4.334
    dE = 16.0 * e0
    tau_elec_huge = 1.0   # an implausibly slow coil -- T_p can never keep up

    t_throttled = sup._predict_arrival(est, p.coils[0].position_m, dE, tau_elec_huge)
    t_unthrottled = sup._predict_arrival(est, p.coils[0].position_m, dE)   # tau_elec=None
    t_naive = est.time_to_reach(p.coils[0].position_m)
    assert t_naive > t_throttled > t_unthrottled


def test_final_arrival_uses_naive_dead_reckoning_not_the_aggressive_correction():
    """Same architectural guard as the departure kick's, for the last coil: end-of-travel
    must not fire before the slug truly reaches it, so _final_arrival is naive
    est.time_to_reach(), not _run_step's aggressive _predict_arrival."""
    p = LinearActuatorParams()
    sup = StepperSupervisor(p)
    sup.start(0.0)
    est = make_estimator(p)

    t = 0.0
    for idx in range(len(p.gates) - 1):
        t += 0.01
        est.on_gate(idx, t=t, pulse_width=0.008)
        sup.on_gate(idx, est, t, v_tgt=1.0)

    last = len(p.gates) - 1
    t += 0.01
    est.on_gate(last, t=t, pulse_width=0.008)
    sup.on_gate(last, est, t, v_tgt=1.0)

    t_naive = est.time_to_reach(p.coils[-1].position_m)
    assert sup._final_arrival == pytest.approx(t_naive)


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


def test_pending_departure_uses_naive_dead_reckoning_not_the_aggressive_correction():
    """Architectural regression guard, not just a numeric one: the departure kick's
    schedule must come from est.time_to_reach() (naive, systematically LATE), never from
    _run_step's own accel-corrected _predict_arrival (aggressive, systematically EARLY) --
    reusing the latter for both was the root cause of four separate incidents this session
    (see docs/DESIGN_OPTIMIZER.md section 1.3). A late-biased schedule can only ever fire
    the departure kick after true arrival, never before; an early-biased one can fire it
    while the slug is still genuinely approaching, turning "repel" into a brake."""
    p = LinearActuatorParams()
    sup = StepperSupervisor(p)
    sup.start(0.0)
    est = make_estimator(p)
    est.on_gate(0, t=0.0, pulse_width=0.008)   # v = 0.5 m/s

    sup.on_gate(0, est, 0.0, v_tgt=1.0)
    assert sup._pending_departure is not None
    _, t_arrival_scheduled = sup._pending_departure

    t_naive = est.time_to_reach(p.coils[0].position_m)
    assert t_arrival_scheduled == pytest.approx(t_naive)


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
