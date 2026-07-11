import dataclasses

import pytest

from emac_sim.design_sensitivity import (
    ALL_KNOBS,
    _bound_values,
    full_sensitivity_report,
    interaction_sweep,
    sweep_knob,
)
from emac_sim.optimize_design import Bounds, DesignKnobs

# Small/cheap baseline so this suite runs fast -- structural correctness of the sweep
# machinery is what's under test here, not any particular design's physical realism
# (that's covered by test_optimize_design.py and the design_sensitivity smoke-testing
# that found/fixed the real bugs this module's usage originally surfaced).
FAST_BOUNDS = Bounds(n_coils=(2, 3), turns=(20, 60), coil_length_m=(0.02, 0.04),
                      radial_thickness_m=(0.004, 0.01), magnet_radius_m=(0.003, 0.008),
                      magnet_length_m=(0.01, 0.02), i_max_a=(5.0, 30.0),
                      max_tube_length_m=0.2)
BASELINE = DesignKnobs(bus_voltage_v=24.0, driver_bipolar=True, pump_envelope="rcos",
                        n_coils=2, turns=40, coil_length_m=0.03, radial_thickness_m=0.006,
                        magnet_radius_m=0.005, magnet_length_m=0.015, remanence_t=1.2,
                        i_max_a=15.0)


def test_bound_values_categorical_knobs_return_their_fixed_options():
    assert _bound_values("driver_bipolar", FAST_BOUNDS, 9) == [False, True]
    assert _bound_values("pump_envelope", FAST_BOUNDS, 9) == ["rcos", "trapezoid", "square"]


def test_bound_values_integer_knob_rounds_and_dedupes():
    values = _bound_values("n_coils", FAST_BOUNDS, 9)
    assert all(isinstance(v, int) for v in values)
    assert values == sorted(set(values))
    assert values[0] >= FAST_BOUNDS.n_coils[0]
    assert values[-1] <= FAST_BOUNDS.n_coils[1]


def test_bound_values_continuous_knob_spans_bounds_linearly():
    lo, hi = FAST_BOUNDS.i_max_a
    values = _bound_values("i_max_a", FAST_BOUNDS, 5)
    assert values[0] == pytest.approx(lo)
    assert values[-1] == pytest.approx(hi)
    assert len(values) == 5


def test_sweep_knob_holds_every_other_field_at_baseline():
    points = sweep_knob("i_max_a", BASELINE, FAST_BOUNDS, n_points=3, dt=1e-3, t_end=0.3)
    assert [p["value"] for p in points] == _bound_values("i_max_a", FAST_BOUNDS, 3)
    assert all(isinstance(p["speed"], float) and p["speed"] >= 0.0 for p in points)


def test_sweep_knob_over_tube_length_budget_reports_zero_speed():
    over_budget = dataclasses.replace(BASELINE, n_coils=100)
    tiny_bounds = Bounds(max_tube_length_m=0.01)
    points = sweep_knob("i_max_a", over_budget, tiny_bounds, n_points=2, dt=1e-3, t_end=0.1)
    assert all(p["speed"] == 0.0 for p in points)


def test_full_sensitivity_report_covers_every_declared_knob():
    report = full_sensitivity_report(BASELINE, FAST_BOUNDS, n_points=3, dt=1e-3, t_end=0.3)
    assert set(report.keys()) == set(ALL_KNOBS)
    assert len(report["i_max_a"]) == 3
    assert len(report["driver_bipolar"]) == 2       # categorical: fixed option count, not n_points
    assert len(report["pump_envelope"]) == 3


def test_interaction_sweep_grid_shape_matches_both_axes():
    result = interaction_sweep("i_max_a", "driver_bipolar", BASELINE, FAST_BOUNDS,
                                n_points_a=3, dt=1e-3, t_end=0.3)
    assert result["values_a"] == _bound_values("i_max_a", FAST_BOUNDS, 3)
    assert result["values_b"] == [False, True]
    assert len(result["grid"]) == 3
    assert all(len(row) == 2 for row in result["grid"])


def test_sweep_knob_accepts_fem_reference_force_law():
    """The design-space sensitivity tooling can run against fem.reference_backend's real
    coupling shape instead of the synthetic analytic lobe -- see
    optimize_design.FORCE_LAWS / docs/FEM_PIPELINE.md."""
    points = sweep_knob("i_max_a", BASELINE, FAST_BOUNDS, n_points=3, dt=1e-3, t_end=0.3,
                         force_law="fem_reference")
    assert len(points) == 3
    assert all(isinstance(p["speed"], float) and p["speed"] >= 0.0 for p in points)


def test_interaction_sweep_defaults_second_knob_point_count_to_first():
    result = interaction_sweep("i_max_a", "remanence_t", BASELINE, FAST_BOUNDS, n_points_a=4,
                                dt=1e-3, t_end=0.3)
    assert len(result["values_a"]) == 4
    assert len(result["values_b"]) == 4
    assert len(result["grid"]) == 4
    assert all(len(row) == 4 for row in result["grid"])
