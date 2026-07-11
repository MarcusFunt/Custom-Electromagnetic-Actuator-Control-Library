"""Analytic reference FEMBackend -- NOT a real FEM solve.

This exists purely to validate the geometry/sweep/LUT/plant plumbing end to end on a
machine without FEMM installed (see femm_backend.py for the real solver). It reuses
`coil_design.off_axis_radial_field_cylinder_magnet`, the same closed-form Biot-Savart
result `coil_design.estimate_k_a` already uses to get a single k_a constant -- but
evaluated AT the requested offset instead of only at its peak, so a swept LUT built from
this backend traces out the coil's REAL (non-Gaussian) coupling shape rather than assuming
`plant.q_shape`'s Gaussian-lobe approximation. That is a genuine accuracy improvement over
the existing analytic path even without FEMM, though it still shares FEMM's real
limitations: no iron, no saturation, vacuum permeability everywhere, an idealized uniformly
magnetized cylinder for the slug.

Force sign convention (see backend.FEMBackend.solve): derived from the Lorentz force on an
azimuthal coil current in the magnet's radial field, F_on_slug = i * B_rho(z_coil) *
wire_length_m where z_coil = -offset_m is the coil's position measured from the magnet's
own axial center (off_axis_radial_field_cylinder_magnet's convention). Verified against the
existing q_shape/f_current_pm sign convention: positive offset_m with positive current_a
pulls the slug back toward the coil (attraction), matching plant.f_current_pm.
"""

from __future__ import annotations

from ..coil_design import off_axis_radial_field_cylinder_magnet
from .backend import ForcePoint
from .geometry import CoilWindingGeometry, SlugGeometry, wire_length_m


class AnalyticReferenceBackend:
    """FEMBackend-shaped wrapper around coil_design's closed-form PM field. See module
    docstring: use for pipeline testing/plumbing, or as a fast fallback when FEMM isn't
    installed -- not a substitute for a real FEM solve when accuracy matters."""

    def solve(self, coil: CoilWindingGeometry, slug: SlugGeometry,
              offset_m: float, current_a: float) -> ForcePoint:
        mean_radius_m = coil.mean_radius_m(slug)
        b_rho = off_axis_radial_field_cylinder_magnet(
            mean_radius_m, -offset_m, slug.magnet_radius_m, slug.magnet_length_m,
            slug.remanence_t,
        )
        force_n = current_a * b_rho * wire_length_m(coil, slug)
        return ForcePoint(force_n=force_n)
