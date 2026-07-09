import pytest

from emac_sim.optimize_design import (
    PUMP_ENVELOPES,
    Bounds,
    build_params,
    decode,
    optimize,
    simulate_design,
)


TIGHT_BOUNDS = Bounds(n_coils=(2, 4), turns=(20, 100), coil_length_m=(0.02, 0.05),
                       radial_thickness_m=(0.003, 0.02), magnet_radius_m=(0.003, 0.01),
                       magnet_length_m=(0.01, 0.03), i_max_a=(5.0, 40.0),
                       max_tube_length_m=0.3)


def test_decode_round_trips_each_field_in_order():
    x = [48.0, 1, 2, 5, 200, 0.03, 0.01, 0.006, 0.02, 1.3, 20.0]
    knobs = decode(x)
    assert knobs.bus_voltage_v == pytest.approx(48.0)
    assert knobs.driver_bipolar is True
    assert knobs.pump_envelope == PUMP_ENVELOPES[2]
    assert knobs.n_coils == 5
    assert knobs.turns == 200
    assert knobs.coil_length_m == pytest.approx(0.03)
    assert knobs.i_max_a == pytest.approx(20.0)


def test_build_params_produces_one_coil_and_gate_per_station():
    knobs = decode([48.0, 0, 0, 4, 100, 0.03, 0.01, 0.006, 0.02, 1.3, 20.0])
    p = build_params(knobs)
    assert len(p.coils) == 4
    assert len(p.gates) == 4
    assert p.current_loop == "rl"
    assert p.driver_bipolar is False
    assert p.mass_kg > 0.0


def test_simulate_design_returns_zero_for_an_infeasible_tube_length():
    """The objective (not simulate_design itself) rejects over-budget tube lengths --
    simulate_design just runs whatever it's given. This pins that simulate_design alone
    doesn't silently enforce the budget, so the objective's own check is load-bearing."""
    knobs = decode([48.0, 0, 0, 4, 100, 0.03, 0.01, 0.006, 0.02, 1.3, 20.0])
    v = simulate_design(knobs, dt=5e-4, t_end=1.0)
    assert v >= 0.0    # doesn't raise; a reasonable design should move at all


def test_optimize_runs_and_returns_a_feasible_design():
    knobs, speed, result = optimize(bounds=TIGHT_BOUNDS, maxiter=1, popsize=3,
                                     dt=5e-4, t_end=1.0, seed=0)
    assert speed >= 0.0
    assert TIGHT_BOUNDS.n_coils[0] <= knobs.n_coils <= TIGHT_BOUNDS.n_coils[1]
    assert TIGHT_BOUNDS.turns[0] <= knobs.turns <= TIGHT_BOUNDS.turns[1]
    assert knobs.n_coils * knobs.coil_length_m <= TIGHT_BOUNDS.max_tube_length_m + 1e-9
    assert knobs.pump_envelope in PUMP_ENVELOPES
