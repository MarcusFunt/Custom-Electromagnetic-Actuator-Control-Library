"""Shared FEM/analytic backend interface: one (coil, slug, offset, current) -> force point.

Keeping this as a narrow Protocol (rather than a base class) means reference_backend.py and
femm_backend.py share nothing but this call shape -- sweep.py doesn't care which one it's
driving, and a test can hand it a trivial stub with no other machinery involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .geometry import CoilWindingGeometry, SlugGeometry


@dataclass(frozen=True)
class ForcePoint:
    """One solved operating point. flux_linkage_wb is optional (NaN when a backend has no
    way to estimate it, e.g. the analytic reference backend) -- force_n is the only field
    the LUT/plant path actually consumes today; flux linkage is kept for a future
    FEM-derived inductance table without a schema change."""

    force_n: float
    flux_linkage_wb: float = float("nan")


class FEMBackend(Protocol):
    def solve(self, coil: CoilWindingGeometry, slug: SlugGeometry,
              offset_m: float, current_a: float) -> ForcePoint:
        """Axial force (N) on the slug, and (if available) flux linkage (Wb), with the
        slug's center at `offset_m` from the coil's own center (signed, same convention as
        plant.q_shape(x - coil.position_m, ...)) and `current_a` (signed) driven through
        the coil. Force sign convention matches f_current_pm: positive offset_m together
        with positive current_a should attract the slug back toward the coil (negative
        force), mirroring q_shape's odd, restoring shape."""
        ...
