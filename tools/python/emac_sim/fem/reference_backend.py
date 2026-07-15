"""Analytic reference FEMBackend -- NOT a real FEM solve.

This exists purely to validate the geometry/sweep/LUT/plant plumbing end to end on a
machine without FEMM installed (see femm_backend.py for the real solver). It calls
`coil_design.winding_averaged_force_per_amp` -- the SAME shared kernel the analytic plant's
k_a is built from -- at the requested offset instead of only at its peak, so a swept LUT
built from this backend traces out the coil's REAL (non-Gaussian) coupling shape rather than
assuming `plant.q_shape`'s Gaussian-lobe approximation, and is guaranteed consistent with the
plant's own peak thrust. That is a genuine accuracy improvement over the synthetic lobe even
without FEMM, and it is validated against real FEMM to ~1-2% (see docs/VALIDATION.md), though
it still shares FEMM's idealizations: no iron, no saturation, vacuum permeability everywhere,
a uniformly magnetized cylinder for the slug.

Force sign convention (see backend.FEMBackend.solve): the Lorentz force on an azimuthal coil
current in the magnet's radial field, F_on_slug = i * <B_rho> * wire_length, winding-averaged
over the (r, z) envelope. Positive offset_m with positive current_a pulls the slug back
toward the coil (attraction), matching plant.f_current_pm and plant.q_shape's odd, restoring
sign.
"""

from __future__ import annotations

from ..coil_design import winding_averaged_force_per_amp
from .backend import ForcePoint
from .geometry import CoilWindingGeometry, SlugGeometry


class AnalyticReferenceBackend:
    """FEMBackend-shaped wrapper around coil_design's closed-form PM field. See module
    docstring: use for pipeline testing/plumbing, or as a fast fallback when FEMM isn't
    installed -- not a substitute for a real FEM solve when accuracy matters."""

    def solve(self, coil: CoilWindingGeometry, slug: SlugGeometry,
              offset_m: float, current_a: float) -> ForcePoint:
        # Force = current * the winding-averaged force per amp. This is coil_design's shared
        # kernel (winding_averaged_force_per_amp) -- the SAME function build_coil_station /
        # estimate_k_a use for the analytic plant's k_a -- so a swept LUT from this backend
        # and the analytic-lobe plant agree by construction rather than by two hand-kept-in-
        # sync copies of the quadrature (they previously drifted 25-75% apart). The 7x3
        # Gauss-Legendre average keeps quadrature error below ~0.6% of the true
        # winding-averaged force across the geometries the optimizer explores.
        force_per_amp = winding_averaged_force_per_amp(
            offset_m, coil.bore_radius_m(slug), coil.outer_radius_m(slug), coil.coil_length_m,
            coil.turns, slug.magnet_radius_m, slug.magnet_length_m, slug.remanence_t,
        )
        return ForcePoint(force_n=current_a * force_per_amp)
