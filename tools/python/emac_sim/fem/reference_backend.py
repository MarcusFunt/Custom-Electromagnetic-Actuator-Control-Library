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

import numpy as np

from ..coil_design import off_axis_radial_field_cylinder_magnet
from .backend import ForcePoint
from .geometry import CoilWindingGeometry, SlugGeometry, wire_length_m

# Gauss-Legendre nodes/weights for averaging B_rho over the winding cross-section (axial x
# radial). 7x3 keeps the quadrature error below ~0.6% of the true winding-averaged force
# across the geometries the optimizer explores (checked in tests) while staying only ~21
# field evaluations per solve. The nodes/weights are on [-1, 1]; solve() maps them onto the
# winding's actual (z, r) extent.
_GL_Z_NODES, _GL_Z_WEIGHTS = np.polynomial.legendre.leggauss(7)
_GL_R_NODES, _GL_R_WEIGHTS = np.polynomial.legendre.leggauss(3)


class AnalyticReferenceBackend:
    """FEMBackend-shaped wrapper around coil_design's closed-form PM field. See module
    docstring: use for pipeline testing/plumbing, or as a fast fallback when FEMM isn't
    installed -- not a substitute for a real FEM solve when accuracy matters."""

    def solve(self, coil: CoilWindingGeometry, slug: SlugGeometry,
              offset_m: float, current_a: float) -> ForcePoint:
        # The axial force on the winding is F = i * integral over the winding cross-section
        # of B_rho(r, z) * (turn-length density) dA. Every turn at radius r contributes a
        # 2*pi*r loop, so the correct field to use is B_rho AVERAGED over the (z, r) envelope
        # WEIGHTED by r -- NOT B_rho at the single mean-radius/coil-center point, which
        # over-estimates the peak force per amp by ~60% for a coil whose length is comparable
        # to the coupling scale (the single point sits at the field maximum; the rest of the
        # winding sees less). This averaged form reduces EXACTLY to the old single-point
        # expression in the uniform-field limit, so the sign/odd-symmetry/linearity
        # properties are unchanged.
        bore_m = coil.bore_radius_m(slug)
        outer_m = coil.outer_radius_m(slug)
        half_len_m = 0.5 * coil.coil_length_m
        r_mid, r_half = 0.5 * (bore_m + outer_m), 0.5 * (outer_m - bore_m)

        num = 0.0
        den = 0.0
        for zn, zw in zip(_GL_Z_NODES, _GL_Z_WEIGHTS):
            z_m = half_len_m * zn                       # coil center is the origin (z=0)
            for rn, rw in zip(_GL_R_NODES, _GL_R_WEIGHTS):
                r_m = r_mid + r_half * rn
                b_rho = off_axis_radial_field_cylinder_magnet(
                    r_m, -offset_m + z_m, slug.magnet_radius_m, slug.magnet_length_m,
                    slug.remanence_t,
                )
                weight = zw * rw * r_m                  # turn-length weight ~ r
                num += weight * b_rho
                den += weight
        b_rho_avg = num / den

        force_n = current_a * b_rho_avg * wire_length_m(coil, slug)
        return ForcePoint(force_n=force_n)
