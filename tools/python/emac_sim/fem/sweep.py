"""Sweep a FEMBackend over a (offset, current) grid into a ForceLUT."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .backend import FEMBackend
from .geometry import CoilWindingGeometry, SlugGeometry, default_sweep_ranges
from .lut import ForceLUT


def sweep_coil(coil: CoilWindingGeometry, slug: SlugGeometry, backend: FEMBackend,
                offsets_m: Sequence[float] | None = None,
                currents_a: Sequence[float] | None = None,
                on_point: Callable[[int, int, int, int], None] | None = None) -> ForceLUT:
    """Solve `backend` at every (offset, current) grid point and pack the results into a
    ForceLUT. Defaults to `geometry.default_sweep_ranges(coil, slug)` when a range isn't
    given explicitly. `on_point`, if given, is called after each solve as
    (offset_index, current_index, n_done, n_total) -- a hook for CLI progress reporting,
    since a real FEMM sweep is slow enough (one full solve per point) that silent progress
    would be a poor experience for anything past a handful of points."""
    if offsets_m is None or currents_a is None:
        default_offsets, default_currents = default_sweep_ranges(coil, slug)
        offsets_m = default_offsets if offsets_m is None else offsets_m
        currents_a = default_currents if currents_a is None else currents_a

    offsets = np.asarray(offsets_m, dtype=float)
    currents = np.asarray(currents_a, dtype=float)
    force = np.empty((offsets.size, currents.size), dtype=float)
    n_total = offsets.size * currents.size

    n_done = 0
    for i, offset in enumerate(offsets):
        for j, current in enumerate(currents):
            point = backend.solve(coil, slug, float(offset), float(current))
            force[i, j] = point.force_n
            n_done += 1
            if on_point is not None:
                on_point(i, j, n_done, n_total)

    metadata = {
        "coil_position_m": coil.position_m,
        "turns": coil.turns,
        "coil_length_m": coil.coil_length_m,
        "radial_thickness_m": coil.radial_thickness_m,
        "bore_clearance_m": coil.bore_clearance_m,
        "magnet_radius_m": slug.magnet_radius_m,
        "magnet_length_m": slug.magnet_length_m,
        "remanence_t": slug.remanence_t,
        "slug_type": slug.slug_type,           # "pm" | "reluctance" -- lets every downstream
        "steel_material": slug.steel_material,  # analyzer/QC/GUI auto-detect a reluctance table
        "backend": type(backend).__name__,
    }
    return ForceLUT(offsets_m=offsets, currents_a=currents, force_n=force, metadata=metadata)
