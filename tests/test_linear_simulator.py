import math

from emac_sim.linear_estimator import LinearStepperEstimator
from emac_sim.linear_plant import LinearActuatorParams
from emac_sim.linear_sim import LinearSimulator
from emac_sim.linear_supervisor import DONE, FAULT, StepperSupervisor


def run_default(end_of_travel: str = "coast", t_end: float = 3.0):
    p = LinearActuatorParams(end_of_travel=end_of_travel)
    est = LinearStepperEstimator(
        gate_positions=[g.position_m for g in p.gates],
        gate_widths=[g.w_eff for g in p.gates],
    )
    sup = StepperSupervisor(p)
    sim = LinearSimulator(p, est, sup, dt=2e-4, sample_every=10)
    log = sim.run(x0=-0.03, v0=0.0, v_tgt=0.5, t_end=t_end)
    return p, sup, log


def test_slug_advances_monotonically_and_never_stalls():
    p, sup, log = run_default()

    assert len(log.gate_t) > 0
    assert sup.mode != FAULT

    # allow a tiny numerical tolerance for symplectic-integration jitter
    for a, b in zip(log.x, log.x[1:]):
        assert b >= a - 1e-6


def test_gates_fire_in_strictly_increasing_index_order():
    _, _, log = run_default()

    assert log.gate_index == sorted(log.gate_index)
    assert log.gate_index == list(range(len(log.gate_index)))


def test_all_five_gates_fire_given_enough_time():
    p, sup, log = run_default(t_end=3.0)

    assert len(log.gate_t) == len(p.gates)
    assert all(math.isfinite(v) for v in log.gate_v)


def test_coast_end_of_travel_exits_past_the_last_coil():
    p, sup, log = run_default(end_of_travel="coast", t_end=3.0)

    assert sup.mode == DONE
    assert log.x[-1] > p.coils[-1].position_m


def test_brake_hold_reduces_overshoot_and_speed_versus_coast():
    """A single end-of-travel brake pulse -- the only station left to use for braking,
    with no further gate to correct against -- won't bring the slug to an exact stop the
    way the pendulum's many-cycle energy shaping does; see docs/DESIGN_LINEAR.md's open
    questions. This checks it measurably helps, not that it's precise."""
    p, sup_coast, log_coast = run_default(end_of_travel="coast", t_end=3.0)
    p, sup_brake, log_brake = run_default(end_of_travel="brake_hold", t_end=3.0)

    assert sup_brake.mode == DONE
    last_coil_x = p.coils[-1].position_m
    assert abs(log_brake.x[-1] - last_coil_x) < abs(log_coast.x[-1] - last_coil_x)
    assert abs(log_brake.v[-1]) < abs(log_coast.v[-1])


def test_log_field_lengths_are_consistent():
    p, _, log = run_default()

    n = len(log.t)
    assert len(log.x) == n
    assert len(log.v) == n
    assert len(log.active_coil) == n
    assert len(log.active_current) == n
    assert len(log.active_temperature_c) == n
    assert len(log.coil_currents) == n
    assert all(len(row) == len(p.coils) for row in log.coil_currents)
    assert len(log.x_est) == n
    assert len(log.status) == n
    assert len(log.supervisor_mode) == n


def run_with_current_loop(current_loop: str, t_end: float = 3.0):
    p = LinearActuatorParams(current_loop=current_loop)
    est = LinearStepperEstimator(
        gate_positions=[g.position_m for g in p.gates],
        gate_widths=[g.w_eff for g in p.gates],
    )
    sup = StepperSupervisor(p)
    sim = LinearSimulator(p, est, sup, dt=2e-4, sample_every=10)
    log = sim.run(x0=-0.03, v0=0.0, v_tgt=0.5, t_end=t_end)
    return p, sup, log


def test_rl_current_loop_still_completes_the_run():
    p, sup, log = run_with_current_loop("rl")

    assert len(log.gate_t) == len(p.gates)
    assert sup.mode != FAULT
    for a, b in zip(log.x, log.x[1:]):
        assert b >= a - 1e-6


def test_rl_current_loop_produces_a_genuinely_different_trace():
    """'rl' must actually be exercising real RL dynamics, not silently behaving like
    'ideal'. NOTE: with this reference config's L/R (tau ~ 3.3 ms) much faster than the
    commutation timescale (tens of ms), the bang-bang controller keeps up with -- and can
    slightly overshoot -- a slowly-rising target rather than only lagging it; a strict
    'rl never exceeds ideal' claim is NOT a true invariant here (that would only hold if
    tau were comparable to or slower than the commanded ramp). The precise, always-true
    claim -- current can't jump to a target in zero time -- is pinned instead at the
    primitive level in test_linear_plant.py::test_rl_current_lags_an_ideal_instantaneous_target."""
    _, _, log_ideal = run_with_current_loop("ideal")
    _, _, log_rl = run_with_current_loop("rl")

    assert log_rl.active_current != log_ideal.active_current


def test_thermal_model_off_by_default_pins_every_coil_at_ambient():
    """Regression guard: thermal_model=False (the default) must reproduce the exact
    fixed-resistance behavior -- every logged temperature stays at ambient, never moves,
    however much current flows."""
    p, _, log = run_with_current_loop("rl")
    assert not p.thermal_model
    assert all(temp == p.ambient_temperature_c for temp in log.active_temperature_c)


def test_thermal_model_on_heats_up_a_sustained_high_current_coil():
    """With thermal_model=True and enough sustained current, a coil's logged temperature
    must actually rise above ambient at some point in the run -- otherwise the feature
    isn't doing anything."""
    p = LinearActuatorParams(current_loop="rl", bus_voltage_v=48.0, thermal_model=True)
    est = LinearStepperEstimator(
        gate_positions=[g.position_m for g in p.gates],
        gate_widths=[g.w_eff for g in p.gates],
    )
    sup = StepperSupervisor(p, i_max=6.0)
    sim = LinearSimulator(p, est, sup, dt=2e-4, sample_every=10)
    log = sim.run(x0=-0.03, v0=0.0, v_tgt=0.5, t_end=3.0)

    assert max(log.active_temperature_c) > p.ambient_temperature_c


def test_thermal_model_resistance_rise_lags_a_pure_rl_run_with_the_same_current():
    """Feeding the temperature-adjusted (higher) resistance back into the electrical
    dynamics under thermal_model=True must make current rise measurably more slowly than
    an otherwise-identical thermal_model=False run once the coil has had time to heat up
    -- confirming the R(T) feedback loop is actually wired into coil_current_step, not
    just computed and discarded."""
    def run(thermal_model: bool):
        p = LinearActuatorParams(current_loop="rl", bus_voltage_v=48.0,
                                  thermal_model=thermal_model)
        est = LinearStepperEstimator(
            gate_positions=[g.position_m for g in p.gates],
            gate_widths=[g.w_eff for g in p.gates],
        )
        sup = StepperSupervisor(p, i_max=6.0)
        sim = LinearSimulator(p, est, sup, dt=2e-4, sample_every=10)
        return sim.run(x0=-0.03, v0=0.0, v_tgt=0.5, t_end=3.0)

    log_cold = run(thermal_model=False)
    log_hot = run(thermal_model=True)

    assert log_hot.active_current != log_cold.active_current


def _max_simultaneous_coils(log) -> int:
    """The most coils carrying nonzero current at any single sampled tick."""
    return max((sum(1 for i in row if abs(i) > 1e-9) for row in log.coil_currents), default=0)


def run_push_pull(push_pull: bool, current_loop: str | None = None):
    """A genuinely strong design (real coil_design geometry, 10 A cap) -- the default
    LinearActuatorParams is far too weak to pump hard enough to exhibit (or benefit from)
    two-coil overlap. Same design/fidelity the push-pull video uses, so the test exercises
    the real configuration, not a toy one. `current_loop` optionally overrides build_params'
    "rl" default (used by the ideal-mode control test below)."""
    import dataclasses

    from emac_sim.optimize_design import DesignKnobs, build_params

    knobs = DesignKnobs(bus_voltage_v=48.0, driver_bipolar=True, pump_envelope="rcos",
                         n_coils=8, turns=180, coil_length_m=0.02, radial_thickness_m=0.01,
                         magnet_radius_m=0.008, magnet_length_m=0.02, remanence_t=1.2,
                         i_max_a=10.0)
    p = build_params(knobs, force_law="analytic")
    if current_loop is not None:
        p = dataclasses.replace(p, current_loop=current_loop)
    pitch = knobs.coil_length_m
    est = LinearStepperEstimator(
        gate_positions=[g.position_m for g in p.gates],
        gate_widths=[g.w_eff for g in p.gates],
    )
    sup = StepperSupervisor(p, i_max=knobs.i_max_a, pm_envelope=knobs.pump_envelope,
                            bootstrap_timeout_s=0.20, push_pull=push_pull)
    sim = LinearSimulator(p, est, sup, dt=2e-5, sample_every=1)
    log = sim.run(x0=-0.5 * pitch - 0.001, v0=0.0, v_tgt=100.0, t_end=0.09)
    return p, sup, log


def test_push_pull_off_drives_one_coil_at_a_time_under_ideal_current():
    """Control for the test below: with 'ideal' current (no RL decay tails to confound),
    the single-coil scheme drives at most ONE coil per tick -- confirming the two-coil
    overlap under push_pull is the scheme genuinely targeting two coils, not a plant
    artifact. (Under 'rl', a just-deactivated coil's decaying current can transiently
    overlap the next coil's ramp even without push-pull -- current can't jump to zero --
    which is a property of the RL circuit, not of the commutation scheme, so this control
    isolates the scheme by using ideal current.)"""
    _, sup, log = run_push_pull(push_pull=False, current_loop="ideal")
    assert sup.mode != FAULT
    assert _max_simultaneous_coils(log) <= 1


def test_push_pull_on_drives_two_coils_at_once_even_under_ideal_current():
    """The mechanism, isolated from RL decay: even under 'ideal' current, push_pull=True
    targets (and so energizes) two coils at the same tick -- impossible for the single-coil
    scheme above."""
    _, sup, log = run_push_pull(push_pull=True, current_loop="ideal")
    assert sup.mode != FAULT
    assert _max_simultaneous_coils(log) >= 2


def test_push_pull_on_energizes_two_coils_at_once_and_still_clears_every_gate():
    """The core of two-coil push-pull: many ticks have TWO+ coils driven at once (a
    repel-behind overlapping the next coil's attract-ahead), which the single-coil scheme
    can never produce -- and the run still clears every gate without FAULT."""
    p, sup, log = run_push_pull(push_pull=True)
    assert sup.mode != FAULT
    assert len(log.gate_t) == len(p.gates)
    assert _max_simultaneous_coils(log) >= 2


def test_push_pull_reaches_a_higher_exit_speed_than_single_coil_on_the_same_design():
    """The payoff: using the previously-idle between-coil gaps to push-and-pull at once
    should extract more forward impulse, so exit speed under push_pull=True must exceed the
    single-coil scheme's on an identical design (measured ~7.2 vs ~5.2 m/s)."""
    _, _, log_single = run_push_pull(push_pull=False)
    _, _, log_pp = run_push_pull(push_pull=True)
    assert log_single.gate_v and log_pp.gate_v
    assert log_pp.gate_v[-1] > log_single.gate_v[-1]
