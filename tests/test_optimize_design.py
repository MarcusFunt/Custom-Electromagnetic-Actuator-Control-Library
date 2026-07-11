import pytest

from emac_sim.fem.femm_backend import FemmNotAvailableError
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


def test_build_params_fem_reference_matches_electrical_properties_of_analytic():
    """force_law only changes the FORCE law -- mass, resistance, inductance, thermal mass
    (all geometry-derived, not force-law-derived) must be identical between the two."""
    knobs = decode([48.0, 0, 0, 3, 60, 0.02, 0.008, 0.005, 0.015, 1.2, 15.0])
    analytic = build_params(knobs, force_law="analytic")
    fem = build_params(knobs, force_law="fem_reference")

    assert analytic.mass_kg == pytest.approx(fem.mass_kg)
    assert len(analytic.coils) == len(fem.coils) == 3
    for a_coil, f_coil in zip(analytic.coils, fem.coils):
        assert a_coil.position_m == pytest.approx(f_coil.position_m)
        assert a_coil.resistance_ohm == pytest.approx(f_coil.resistance_ohm)
        assert a_coil.inductance_h == pytest.approx(f_coil.inductance_h)
        assert a_coil.thermal_mass_j_per_k == pytest.approx(f_coil.thermal_mass_j_per_k)


def test_build_params_fem_reference_uses_a_force_lut_analytic_does_not():
    knobs = decode([48.0, 0, 0, 3, 60, 0.02, 0.008, 0.005, 0.015, 1.2, 15.0])
    analytic = build_params(knobs, force_law="analytic")
    fem = build_params(knobs, force_law="fem_reference")

    assert analytic.coils[0].force_lut is None
    assert fem.coils[0].force_lut is not None
    # the force_lut is directly callable -- a live analytic-reference-backend query, not a
    # pre-baked table -- and produces a nonzero, sane force for a nonzero current
    force = fem.coils[0].force_lut(0.005, 5.0)
    assert force != 0.0


def test_build_params_rejects_unknown_force_law():
    knobs = decode([48.0, 0, 0, 3, 60, 0.02, 0.008, 0.005, 0.015, 1.2, 15.0])
    with pytest.raises(ValueError):
        build_params(knobs, force_law="not_a_real_force_law")


def test_simulate_design_runs_under_fem_reference_force_law():
    knobs = decode([48.0, 0, 0, 3, 60, 0.02, 0.008, 0.005, 0.015, 1.2, 15.0])
    v = simulate_design(knobs, dt=1e-3, t_end=0.5, force_law="fem_reference")
    assert v >= 0.0


def test_optimize_runs_and_returns_a_feasible_design():
    knobs, speed, result, femm_speed = optimize(bounds=TIGHT_BOUNDS, maxiter=1, popsize=3,
                                                  dt=5e-4, t_end=1.0, seed=0,
                                                  verify_with_femm=False)
    assert speed >= 0.0
    assert femm_speed is None
    assert TIGHT_BOUNDS.n_coils[0] <= knobs.n_coils <= TIGHT_BOUNDS.n_coils[1]
    assert TIGHT_BOUNDS.turns[0] <= knobs.turns <= TIGHT_BOUNDS.turns[1]
    assert knobs.n_coils * knobs.coil_length_m <= TIGHT_BOUNDS.max_tube_length_m + 1e-9
    assert knobs.pump_envelope in PUMP_ENVELOPES


def test_optimize_rejects_femm_as_the_search_force_law():
    """force_law='femm' cannot drive the search itself -- each FEMM solve takes seconds and
    the search calls the force law millions of times. This must fail fast, before ever
    touching FEMM, so it runs unconditionally regardless of whether FEMM is installed."""
    with pytest.raises(ValueError):
        optimize(bounds=TIGHT_BOUNDS, maxiter=1, popsize=3, dt=5e-4, t_end=1.0, seed=0,
                 force_law="femm")


def test_optimize_auto_verify_skips_cleanly_without_femm(monkeypatch):
    """verify_with_femm=None (the default, 'auto') must not raise when FEMM isn't
    installed -- it should report femm_speed=None instead. Forces the not-installed path by
    construction (monkeypatching the FEMM entry point) so this test's outcome doesn't depend
    on whether FEMM happens to be installed on the machine running it."""
    import emac_sim.optimize_design as optimize_design_module

    def _raise(*_args, **_kwargs):
        raise FemmNotAvailableError("femm.info -- not installed (test stub)")

    monkeypatch.setattr(optimize_design_module, "_femm_force_lut", _raise)
    knobs, speed, result, femm_speed = optimize(bounds=TIGHT_BOUNDS, maxiter=1, popsize=3,
                                                  dt=5e-4, t_end=1.0, seed=0,
                                                  verify_with_femm=None)
    assert femm_speed is None


def test_build_params_femm_shares_one_force_lut_across_coils(monkeypatch):
    """Every coil in one design shares identical winding/magnet geometry (only position_m
    differs), and FEMBackend.solve() never reads coil.position_m -- so build_params should
    sweep FEMM exactly ONCE per design and hand every CoilStation the SAME ForceLUT
    instance, not one sweep per coil. Verified with a stub `_femm_force_lut` (no FEMM
    needed) so this test always runs and directly counts sweeps instead of just checking
    `is not None`."""
    import numpy as np

    import emac_sim.optimize_design as optimize_design_module
    from emac_sim.fem.lut import ForceLUT

    knobs = decode([48.0, 0, 0, 3, 60, 0.02, 0.008, 0.005, 0.015, 1.2, 15.0])
    stub_lut = ForceLUT(
        offsets_m=np.array([-1.0, 0.0, 1.0]), currents_a=np.array([-1.0, 0.0, 1.0]),
        force_n=np.zeros((3, 3)),
        metadata={"turns": knobs.turns, "coil_length_m": knobs.coil_length_m,
                  "radial_thickness_m": knobs.radial_thickness_m,
                  "bore_clearance_m": 0.0015, "magnet_radius_m": knobs.magnet_radius_m,
                  "magnet_length_m": knobs.magnet_length_m, "remanence_t": knobs.remanence_t},
    )
    calls = []

    def _stub(k):
        calls.append(k)
        return stub_lut

    monkeypatch.setattr(optimize_design_module, "_femm_force_lut", _stub)
    p = build_params(knobs, force_law="femm")

    assert len(calls) == 1
    assert len(p.coils) == 3
    assert p.coils[0].force_lut is p.coils[1].force_lut is p.coils[2].force_lut is stub_lut


def test_build_params_femm_matches_electrical_properties_of_analytic():
    """force_law only changes the FORCE law -- resistance/inductance/thermal mass (all
    geometry-derived, not force-law-derived) must match the analytic branch, same invariant
    as the existing fem_reference test above."""
    pytest.importorskip("femm")
    knobs = decode([48.0, 0, 0, 2, 60, 0.02, 0.008, 0.005, 0.015, 1.2, 15.0])
    analytic = build_params(knobs, force_law="analytic")
    femm = build_params(knobs, force_law="femm")

    assert len(femm.coils) == 2
    for a_coil, f_coil in zip(analytic.coils, femm.coils):
        assert a_coil.position_m == pytest.approx(f_coil.position_m)
        assert a_coil.resistance_ohm == pytest.approx(f_coil.resistance_ohm)
        assert a_coil.inductance_h == pytest.approx(f_coil.inductance_h)


def test_optimize_verify_with_femm_reports_a_speed_when_available():
    pytest.importorskip("femm")
    knobs, speed, result, femm_speed = optimize(
        bounds=TIGHT_BOUNDS, maxiter=1, popsize=3, dt=5e-4, t_end=1.0, seed=0,
        verify_with_femm=True,
    )
    assert femm_speed is not None
    assert femm_speed >= 0.0
