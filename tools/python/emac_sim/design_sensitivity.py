"""Sensitivity and interaction analysis for the design optimizer (docs/DESIGN_OPTIMIZER.md).

Where optimize_design.py finds ONE best point, this module maps out the RELATIONSHIPS
between knobs and exit speed around a baseline design -- e.g. "does more current help
more with an H-bridge or a single half-bridge", "which knob does speed respond to most
steeply". Two kinds of sweep:

  - one-at-a-time (OAT) sensitivity: vary ONE knob across its bounds (or its 2-3 discrete
    options, for the two categorical knobs), holding every other knob at the baseline's
    value, record speed. Gives a MAIN EFFECT curve per knob.
  - pairwise interaction: vary TWO knobs across a grid simultaneously, holding the rest
    fixed. Reveals whether one knob's effect DEPENDS on the other's level -- e.g. whether
    i_max_a's effect is much steeper under driver_bipolar=True than False -- which neither
    a single optimum nor either knob's own OAT sweep can show on its own.

Everything here is evaluated LOCALLY around one baseline design (by default, whatever
optimize() found) -- these are relationships in that neighborhood, not global truths about
the whole design space; a knob that looks flat here might matter a lot somewhere else.
Results are plain, JSON-serializable dicts, so they can be cached/reused/plotted without
re-running any simulations.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .optimize_design import PUMP_ENVELOPES, Bounds, DesignKnobs, simulate_design

CONTINUOUS_KNOBS = [
    "bus_voltage_v", "coil_length_m", "radial_thickness_m", "magnet_radius_m",
    "magnet_length_m", "remanence_t", "i_max_a",
]
INTEGER_KNOBS = ["n_coils", "turns"]
CATEGORICAL_KNOBS = ["driver_bipolar", "pump_envelope"]
ALL_KNOBS = CONTINUOUS_KNOBS + INTEGER_KNOBS + CATEGORICAL_KNOBS


def _bound_values(knob: str, bounds: Bounds, n_points: int) -> list[Any]:
    if knob == "driver_bipolar":
        return [False, True]
    if knob == "pump_envelope":
        return list(PUMP_ENVELOPES)
    lo, hi = getattr(bounds, knob)
    raw = [lo + k * (hi - lo) / (n_points - 1) for k in range(n_points)]
    if knob in INTEGER_KNOBS:
        return sorted(set(int(round(v)) for v in raw))
    return raw


def _safe_speed(knobs: DesignKnobs, bounds: Bounds, dt: float, t_end: float,
                 force_law: str = "analytic") -> float:
    if knobs.n_coils * knobs.coil_length_m > bounds.max_tube_length_m:
        return 0.0    # over the tube-length budget -- same treatment as optimize_design.py
    try:
        return simulate_design(knobs, dt=dt, t_end=t_end, force_law=force_law)
    except (ValueError, ZeroDivisionError):
        return 0.0


def sweep_knob(knob: str, baseline: DesignKnobs, bounds: Bounds = Bounds(),
               n_points: int = 9, dt: float = 2e-4, t_end: float = 3.0,
               force_law: str = "analytic") -> list[dict]:
    """Vary ONE knob across its bounds (or discrete options), holding every other knob at
    baseline's value. Returns [{"value": ..., "speed": ...}, ...] in the order tried.
    `force_law`: "analytic" (default, matches optimize_design.py's own default) or
    "fem_reference" -- see optimize_design.FORCE_LAWS / docs/FEM_PIPELINE.md. Sweeping the
    SAME baseline under both is how you see where the two coupling models actually
    disagree, rather than just trusting one of them."""
    values = _bound_values(knob, bounds, n_points)
    points = []
    for value in values:
        knobs = dataclasses.replace(baseline, **{knob: value})
        points.append({"value": value, "speed": _safe_speed(knobs, bounds, dt, t_end, force_law)})
    return points


def full_sensitivity_report(baseline: DesignKnobs, bounds: Bounds = Bounds(),
                             n_points: int = 9, dt: float = 2e-4,
                             t_end: float = 3.0, force_law: str = "analytic") -> dict:
    """OAT sweep for every knob. {knob_name: [{"value":..., "speed":...}, ...]}."""
    return {
        knob: sweep_knob(knob, baseline, bounds, n_points=n_points, dt=dt, t_end=t_end,
                          force_law=force_law)
        for knob in ALL_KNOBS
    }


def interaction_sweep(knob_a: str, knob_b: str, baseline: DesignKnobs,
                       bounds: Bounds = Bounds(), n_points_a: int = 8, n_points_b: int = None,
                       dt: float = 2e-4, t_end: float = 3.0,
                       force_law: str = "analytic") -> dict:
    """Vary TWO knobs across a grid simultaneously, holding everything else fixed at
    baseline. grid[i][j] is the speed at (values_a[i], values_b[j])."""
    values_a = _bound_values(knob_a, bounds, n_points_a)
    values_b = _bound_values(knob_b, bounds, n_points_b or n_points_a)
    grid = []
    for va in values_a:
        row = []
        for vb in values_b:
            knobs = dataclasses.replace(baseline, **{knob_a: va, knob_b: vb})
            row.append(_safe_speed(knobs, bounds, dt, t_end, force_law))
        grid.append(row)
    return {"knob_a": knob_a, "knob_b": knob_b, "values_a": values_a, "values_b": values_b,
            "grid": grid}
