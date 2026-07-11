"""Axisymmetric geometry for one coil/slug pair -- the physical description an FEM
backend needs to build a 2D (r, z) magnetostatic model, and the reference backend needs to
evaluate closed-form fields against. Deliberately reuses the same knobs `coil_design.py`
already uses (turns, coil_length_m, radial_thickness_m, bore_clearance_m, magnet_radius_m,
magnet_length_m, remanence_t) rather than inventing a parallel geometry vocabulary -- a
coil built from these numbers is exactly the coil `coil_design.build_coil_station` would
also estimate k_a/x_c for, so the FEM path and the analytic path are directly comparable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SlugGeometry:
    """The moving permanent-magnet slug. No iron core -- matches this build's default
    (Cmag=0.0) PM-only slug; see linear_plant.py's module docstring for why the reluctance
    branch is a separate, currently-off term this pipeline does not (yet) model."""

    magnet_radius_m: float
    magnet_length_m: float
    remanence_t: float                  # NdFeB N42-ish ~1.2-1.3 T is a typical default

    def __post_init__(self) -> None:
        if self.magnet_radius_m <= 0.0:
            raise ValueError("magnet_radius_m must be > 0")
        if self.magnet_length_m <= 0.0:
            raise ValueError("magnet_length_m must be > 0")


@dataclass(frozen=True)
class CoilWindingGeometry:
    """One coil's winding envelope, in the same terms `coil_design.wind_coil` takes.
    bore_radius is derived from the slug's magnet_radius_m + bore_clearance_m (the bore
    has to clear the slug, so it isn't an independent knob) -- see `bore_radius_m`."""

    position_m: float                   # coil center, along the tube axis
    turns: int
    coil_length_m: float                # axial extent of the winding envelope
    radial_thickness_m: float           # radial extent of the winding envelope
    bore_clearance_m: float = 0.0015
    packing_factor: float = 0.8
    temperature_c: float = 20.0

    def __post_init__(self) -> None:
        if self.turns < 1:
            raise ValueError("turns must be >= 1")
        if self.coil_length_m <= 0.0:
            raise ValueError("coil_length_m must be > 0")
        if self.radial_thickness_m <= 0.0:
            raise ValueError("radial_thickness_m must be > 0")

    def bore_radius_m(self, slug: SlugGeometry) -> float:
        return slug.magnet_radius_m + self.bore_clearance_m

    def mean_radius_m(self, slug: SlugGeometry) -> float:
        return self.bore_radius_m(slug) + 0.5 * self.radial_thickness_m

    def outer_radius_m(self, slug: SlugGeometry) -> float:
        return self.bore_radius_m(slug) + self.radial_thickness_m


# A coil-magnet coupling is a near-dipole interaction: it falls off roughly like 1/r^3 away
# from the coil's own near-field scale (COUPLING_SCALE_FACTOR below). FAR_SPAN_FACTOR sets
# how many multiples of that scale the sweep grid extends to before ForceLUT starts clamping
# queries to the edge value (fem/lut.py) -- verified numerically (see
# tests/test_fem_analysis_regression.py) to already be within ~0.01% of peak for a
# representative coil/slug, i.e. small enough that clamping there is indistinguishable from
# the physically-correct answer (force -> 0 far from the coil). This matters: a NARROW sweep
# (the original span, 1x this scale) left the clamped edge value at several percent of
# peak -- meaning every coil the slug had already passed kept exerting a spurious,
# non-vanishing "phantom" force forever, instead of decaying the way the old analytic
# q_shape() (a fast Gaussian) correctly did. See docs/FEM_PIPELINE.md.
COUPLING_SCALE_FACTOR = 1.5   # x coil_length_m, plus 0.5x magnet_length_m -- see coupling_scale_m
FAR_SPAN_FACTOR = 5.0
# The coupling's PEAK sits partway out into the falloff (not at offset=0, a ZERO of this
# coupling -- see reference_backend's odd-symmetry test), and its curvature stays
# significant well past the "near-field scale" itself: FINE_SPAN_FACTOR widens the
# uniformly-resolved region to 1.5x that scale, which is what actually brought a typical
# adjacent-coil pitch (comparable to 1x the near-field scale) inside the well-resolved
# region instead of landing right in the coarse part of the tail grid.
FINE_SPAN_FACTOR = 1.5


def coupling_scale_m(coil: CoilWindingGeometry, slug: SlugGeometry) -> float:
    """Where a coil-magnet coupling's peak and initial falloff live, in meters -- 1.5 coil
    lengths + 0.5 magnet length. Shared by both the fine- and far-span calculations below so
    they stay proportional to each other as geometry changes."""
    return COUPLING_SCALE_FACTOR * coil.coil_length_m + 0.5 * slug.magnet_length_m


def default_sweep_ranges(coil: CoilWindingGeometry, slug: SlugGeometry,
                          n_offsets: int = 31, n_currents: int = 11,
                          max_current_a: float = 6.0) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """A reasonable default (offset, current) grid for sweeping this coil.

    Offsets span +/- FAR_SPAN_FACTOR times the coil's own coupling scale (see
    `coupling_scale_m`) -- far enough that the swept range's edge is already a physically
    negligible fraction of the peak force. Spacing is TWO-REGION (see `_two_region_grid`):
    uniformly dense across [0, FINE_SPAN_FACTOR * coupling_scale], covering the peak and its
    steep initial falloff, then sparser out to the far edge, where the field is smooth and
    slowly decaying and doesn't need much resolution. Currents span +/- max_current_a
    (signed, since the PM branch is signed -- see plant.f_current_pm) including 0,
    uniformly (the current axis has no comparable near/far structure)."""
    scale_m = coupling_scale_m(coil, slug)
    fine_span_m = FINE_SPAN_FACTOR * scale_m
    far_span_m = FAR_SPAN_FACTOR * scale_m
    offsets = _two_region_grid(fine_span_m, far_span_m, n_offsets)
    currents = tuple(
        -max_current_a + 2.0 * max_current_a * k / (n_currents - 1) for k in range(n_currents)
    ) if n_currents > 1 else (0.0,)
    return offsets, currents


# Fraction of each side's point budget spent on the uniform fine-region grid (the rest goes
# to the sparse far tail). Tuned empirically against AnalyticReferenceBackend (see
# tests/test_fem_analysis_regression.py): 0.6 left worst-case linear-interpolation error
# against the true curve at several percent (the peak's curvature was under-resolved); 0.8
# does noticeably better at the SAME total point count -- purely a better allocation of the
# existing budget, not a more expensive sweep.
NEAR_REGION_FRACTION = 0.8


def _two_region_grid(fine_span: float, far_span: float, n: int) -> tuple[float, ...]:
    """n points symmetric about 0 spanning [-far_span, far_span]: NEAR_REGION_FRACTION of
    the per-side budget spent as a UNIFORM grid across [0, fine_span] (the interesting
    region), the rest spent as a widening-step grid across [fine_span, far_span] (the smooth
    tail). Exact zero at the center point when n is odd (matching a plain uniform grid's
    behavior, and exercised by tests that expect it)."""
    if n <= 1:
        return (0.0,)
    if n == 2:
        return (-far_span, far_span)

    include_center = n % 2 == 1
    half = (n - 1) // 2 if include_center else n // 2

    n_near = min(half, max(1, round(NEAR_REGION_FRACTION * half)))
    n_far = half - n_near

    positive = [fine_span * (k + 1) / n_near for k in range(n_near)]
    for k in range(n_far):
        t = (k + 1) / n_far
        positive.append(fine_span + (far_span - fine_span) * t * t)

    negative = tuple(-p for p in reversed(positive))
    return negative + ((0.0,) + tuple(positive) if include_center else tuple(positive))


def wire_length_m(coil: CoilWindingGeometry, slug: SlugGeometry) -> float:
    """Total conductor length -- same formula coil_design.estimate_k_a uses."""
    return coil.turns * 2.0 * math.pi * coil.mean_radius_m(slug)
