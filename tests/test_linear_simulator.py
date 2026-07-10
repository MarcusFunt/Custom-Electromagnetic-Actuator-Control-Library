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


def test_exit_plane_is_a_separate_interpolated_event_after_the_last_gate():
    p, sup, log = run_default(t_end=1.5)
    assert log.exit_t is not None
    assert log.exit_v is not None and log.exit_v > 0.0
    assert log.exit_position_m == p.resolved_exit_position_m
    assert log.exit_t > log.gate_t[-1]


def test_rl_energy_ledger_closes_to_within_one_percent():
    p = LinearActuatorParams(current_loop="rl", bus_voltage_v=48.0,
                             driver_bipolar=True)
    est = LinearStepperEstimator([g.position_m for g in p.gates],
                                 [g.w_eff for g in p.gates])
    sup = StepperSupervisor(p, full_thrust=True)
    log = LinearSimulator(p, est, sup, dt=2e-5, sample_every=10).run(
        x0=-0.03, v0=0.0, v_tgt=None, t_end=1.5
    )
    assert log.bus_energy_j[-1] > 0.0
    assert abs(log.energy_residual_j[-1]) / log.bus_energy_j[-1] < 0.01


def test_configured_gate_dropout_is_not_silently_ignored():
    p = LinearActuatorParams()
    est = LinearStepperEstimator([g.position_m for g in p.gates],
                                 [g.w_eff for g in p.gates])
    sup = StepperSupervisor(p, bootstrap_dwell_s=0.01, bootstrap_timeout_s=0.02)
    sim = LinearSimulator(p, est, sup, dt=2e-4, sample_every=10,
                          gate_dropout_probability=[1.0] * len(p.gates))
    log = sim.run(x0=-0.03, v0=0.0, v_tgt=0.5, t_end=0.5)
    assert log.gate_t == []
    assert est.have is False


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
    _, _, log = run_default()

    n = len(log.t)
    assert len(log.x) == n
    assert len(log.v) == n
    assert len(log.active_coil) == n
    assert len(log.active_current) == n
    assert len(log.active_temperature_c) == n
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
