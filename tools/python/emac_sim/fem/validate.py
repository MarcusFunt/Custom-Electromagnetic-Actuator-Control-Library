"""Quantify how well the fast analytic coupling model matches a real FEMM field solve --
the check that turns "trust the analytic force law" from an assumption into a number you
can report for YOUR geometry.

Why this exists: the analytic path (coil_design.winding_averaged_force_per_amp, used by the
plant's k_a and by AnalyticReferenceBackend) is a closed-form Biot-Savart integral over an
idealized uniformly-magnetized cylinder in vacuum -- no iron, no saturation, mu_r=1
everywhere. That is fast and, for a bare-PM-slug/air-coil actuator, accurate -- but "how
accurate?" is a property of the specific geometry. This module sweeps both the analytic
reference backend and the real FEMM backend over the same (offset, current) grid and
reports the relative disagreement, so a researcher can decide whether the analytic model is
good enough for their design or whether they need a swept FEMM LUT.

Requires FEMM only for the FEMM side (compare_analytic_to_femm); the data structure and the
error math are FEMM-free and unit-tested on synthetic grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .backend import FEMBackend
from .geometry import CoilWindingGeometry, SlugGeometry, default_sweep_ranges
from .reference_backend import AnalyticReferenceBackend
from .sweep import sweep_coil


@dataclass(frozen=True)
class BackendComparison:
    """Two force grids over the same (offset, current) sweep, plus the error math. `label_a`
    is the fast/reference model (analytic), `label_b` the trusted one (FEMM)."""

    offsets_m: np.ndarray
    currents_a: np.ndarray
    force_a: np.ndarray            # shape (n_offsets, n_currents), analytic
    force_b: np.ndarray            # shape (n_offsets, n_currents), FEMM
    label_a: str = "analytic"
    label_b: str = "femm"

    def peak_force(self) -> float:
        """Largest |force| anywhere on the grid (the natural scale for relative error)."""
        return float(max(np.max(np.abs(self.force_a)), np.max(np.abs(self.force_b))))

    def relative_errors(self, floor_frac: float = 0.05) -> np.ndarray:
        """|a - b| / peak at every grid point where EITHER model exceeds `floor_frac` of the
        peak force; NaN elsewhere. Normalizing by the peak (not the local value) is what
        keeps the far-field tail -- where both forces are a physically negligible few
        milli-newtons and a raw ratio explodes -- from dominating the summary. `floor_frac`
        is the cut for 'a point worth scoring at all'."""
        peak = self.peak_force()
        if peak == 0.0:
            return np.zeros_like(self.force_a)
        diff = np.abs(self.force_a - self.force_b) / peak
        significant = (np.abs(self.force_a) >= floor_frac * peak) | \
                      (np.abs(self.force_b) >= floor_frac * peak)
        return np.where(significant, diff, np.nan)

    def max_relative_error(self, floor_frac: float = 0.05) -> float:
        errs = self.relative_errors(floor_frac)
        return float(np.nanmax(errs)) if np.any(~np.isnan(errs)) else 0.0

    def mean_relative_error(self, floor_frac: float = 0.05) -> float:
        errs = self.relative_errors(floor_frac)
        return float(np.nanmean(errs)) if np.any(~np.isnan(errs)) else 0.0

    def summary(self, floor_frac: float = 0.05) -> dict:
        return {
            "peak_force_n": self.peak_force(),
            "max_rel_error": self.max_relative_error(floor_frac),
            "mean_rel_error": self.mean_relative_error(floor_frac),
            "n_offsets": int(self.offsets_m.size),
            "n_currents": int(self.currents_a.size),
        }

    def report(self, floor_frac: float = 0.05) -> str:
        """A one-current-slice (peak current) table plus the summary, as plain text."""
        s = self.summary(floor_frac)
        lines = [
            f"{self.label_a} vs {self.label_b}: peak |F| = {s['peak_force_n']:.4f} N, "
            f"max rel err = {s['max_rel_error']*100:.1f}%, "
            f"mean rel err = {s['mean_rel_error']*100:.1f}% "
            f"(over the {s['n_offsets']}x{s['n_currents']} grid, points >= "
            f"{floor_frac*100:.0f}% of peak)",
            f"{'offset_mm':>10} {self.label_a+'_N':>12} {self.label_b+'_N':>12} {'rel_err':>8}",
        ]
        j = int(np.argmax(np.abs(self.currents_a)))   # the highest-|current| slice
        for i, off in enumerate(self.offsets_m):
            fa, fb = self.force_a[i, j], self.force_b[i, j]
            rel = abs(fa - fb) / s["peak_force_n"] if s["peak_force_n"] else 0.0
            lines.append(f"{off*1e3:10.2f} {fa:12.4f} {fb:12.4f} {rel*100:7.1f}%")
        return "\n".join(lines)


def compare_backends(coil: CoilWindingGeometry, slug: SlugGeometry,
                     backend_a: FEMBackend, backend_b: FEMBackend,
                     offsets_m: Sequence[float] | None = None,
                     currents_a: Sequence[float] | None = None,
                     label_a: str = "a", label_b: str = "b") -> BackendComparison:
    """Sweep two backends over the SAME grid and package the two force grids for comparison.
    Grid defaults to geometry.default_sweep_ranges(coil, slug)."""
    if offsets_m is None or currents_a is None:
        default_offsets, default_currents = default_sweep_ranges(coil, slug)
        offsets_m = default_offsets if offsets_m is None else offsets_m
        currents_a = default_currents if currents_a is None else currents_a
    lut_a = sweep_coil(coil, slug, backend_a, offsets_m, currents_a)
    lut_b = sweep_coil(coil, slug, backend_b, offsets_m, currents_a)
    return BackendComparison(offsets_m=lut_a.offsets_m, currents_a=lut_a.currents_a,
                             force_a=lut_a.force_n, force_b=lut_b.force_n,
                             label_a=label_a, label_b=label_b)


def compare_analytic_to_femm(coil: CoilWindingGeometry, slug: SlugGeometry,
                             offsets_m: Sequence[float] | None = None,
                             currents_a: Sequence[float] | None = None,
                             mesh_size_m: float | None = None) -> BackendComparison:
    """Validate the fast analytic coupling against a real FEMM solve for one geometry.
    Requires FEMM (imported lazily so this module is import-safe without it). The analytic
    side is AnalyticReferenceBackend (the same winding-averaged kernel the plant's k_a
    uses); the FEMM side is the corrected coil-Lorentz-force extraction."""
    from .femm_backend import FemmBackend
    analytic = AnalyticReferenceBackend()
    with FemmBackend(mesh_size_m=mesh_size_m) as femm_backend:
        return compare_backends(coil, slug, analytic, femm_backend, offsets_m, currents_a,
                                label_a="analytic", label_b="femm")
