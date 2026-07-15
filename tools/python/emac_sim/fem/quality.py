"""Physical-sanity quality control for force LUTs -- the automated version of the checks
that caught (and would have caught) the FEMM force-extraction bug.

A large FEM run produces hundreds of force tables, one per geometry cell, each an expensive
black-box solve. You cannot eyeball them all, and a wrong-but-plausible table (the old
weighted-stress-tensor extraction returned forces up to 2x too large and sign-flipped in the
far field) silently corrupts every design built on it. This module scores each `ForceLUT`
against the physical invariants a real coil-magnet coupling MUST satisfy and flags the cells
that violate them, so a run's output can be triaged in seconds instead of trusted on faith.

Every check is backend-agnostic (it inspects the saved grid, not how it was produced) and
FEMM-free, so it runs anywhere. `check_backend` sweeps a live backend first if you want to
validate a geometry before committing it to a long run; `scan_lut_files` triages a finished
run's `.npz` tables.

The invariants (for the modeled bare-PM-slug / air-coil actuator):
  - finite:            no NaN/Inf anywhere.
  - zero-current null: no current => no force.
  - offset null:       the coupling is odd, so force ~ 0 with the slug centered on the coil.
  - far-field decay:   |force| at the swept offset edges is a small fraction of the peak
                       (the LUT clamps beyond its edges, so the edge MUST have decayed or the
                       clamp injects a phantom force forever).
  - restoring sign:    positive offset with positive current attracts the slug back toward
                       the coil (negative force); the sign is odd in both offset and current.
  - odd symmetry:      force(-offset, i) = -force(offset, i) on a symmetric offset grid.
  - current linearity: a PM slug's force is linear in current, so force/current is ~constant
                       across the current axis (this is the check the stress-tensor bug's
                       spurious ~50% F/i drift failed).
  - monotone tail:     beyond the peak, |force| decays monotonically -- no far-field bumps
                       (the stress-tensor garbage oscillated and changed sign out there).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .backend import FEMBackend
from .geometry import CoilWindingGeometry, SlugGeometry, default_sweep_ranges
from .lut import ForceLUT
from .sweep import sweep_coil


@dataclass(frozen=True)
class QualityTolerances:
    """Fractions of the peak force (or, for linearity, of the mean ratio) a check may deviate
    by before it fails. Defaults are loose enough to pass a correct solve with realistic
    discretization/quadrature noise, tight enough to fail the stress-tensor bug's ~2x errors,
    sign flips, and far-field oscillations."""
    zero_current_null: float = 0.02      # |F(i=0)| / peak
    offset_null: float = 0.08            # |F(offset=0)| / peak  (odd-coupling zero)
    far_field_decay: float = 0.15        # |F(edge offset)| / peak
    restoring_sign: float = 0.02         # wrong-sign |F| / peak tolerated per point
    odd_symmetry: float = 0.05           # |F(-x,i)+F(x,i)| / peak
    current_linearity: float = 0.05      # spread of F/i across current, / mean |F/i|
    monotone_tail: float = 0.03          # tolerated |F| increase per step beyond the peak, / peak
    significant_frac: float = 0.1        # a point must exceed this frac of peak to be scored
                                          # by the sign / linearity checks (skips noisy tail)


@dataclass(frozen=True)
class QualityCheck:
    name: str
    passed: bool
    value: float                          # the measured quantity the tolerance is compared to
    tolerance: float
    detail: str
    applicable: bool = True               # False => skipped (e.g. asymmetric grid), not failed

    def __str__(self) -> str:
        if not self.applicable:
            return f"  [skip] {self.name}: {self.detail}"
        mark = "ok  " if self.passed else "FAIL"
        return f"  [{mark}] {self.name}: {self.detail} (value {self.value:.4f} vs tol {self.tolerance:.4f})"


@dataclass(frozen=True)
class LutQualityReport:
    checks: list[QualityCheck]
    peak_force_n: float
    label: str = ""

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks if c.applicable)

    def failures(self) -> list[QualityCheck]:
        return [c for c in self.checks if c.applicable and not c.passed]

    def __str__(self) -> str:
        head = f"{self.label or 'LUT'}: {'OK' if self.ok else 'SUSPECT'} " \
               f"(peak |F| = {self.peak_force_n:.4f} N, {len(self.failures())} failed)"
        return "\n".join([head, *(str(c) for c in self.checks)])


def check_lut(lut: ForceLUT, tol: QualityTolerances | None = None,
              expect_linear_current: bool = True, label: str = "") -> LutQualityReport:
    """Score one force table against the physical invariants (see module docstring). Set
    `expect_linear_current=False` for a slug with a ferromagnetic (reluctance) body, whose
    force is genuinely nonlinear in current -- the current-linearity check is then skipped
    rather than failed."""
    tol = tol or QualityTolerances()
    offsets = np.asarray(lut.offsets_m, dtype=float)
    currents = np.asarray(lut.currents_a, dtype=float)
    force = np.asarray(lut.force_n, dtype=float)
    checks: list[QualityCheck] = []

    finite = bool(np.isfinite(force).all())
    checks.append(QualityCheck(
        "finite", finite, 0.0 if finite else 1.0, 0.0,
        "all force values finite" if finite else f"{int((~np.isfinite(force)).sum())} non-finite values"))
    if not finite:
        # Nothing else is meaningful on a table with NaN/Inf.
        return LutQualityReport(checks, float("nan"), label)

    peak = float(np.max(np.abs(force)))
    if peak == 0.0:
        checks.append(QualityCheck("nonzero", False, 0.0, 0.0, "table is entirely zero"))
        return LutQualityReport(checks, 0.0, label)

    j_zero = int(np.argmin(np.abs(currents)))
    j_peak = int(np.argmax(np.abs(currents)))
    i_zero = int(np.argmin(np.abs(offsets)))

    # zero-current null
    if np.isclose(currents[j_zero], 0.0):
        v = float(np.max(np.abs(force[:, j_zero]))) / peak
        checks.append(QualityCheck("zero_current_null", v <= tol.zero_current_null, v,
                                   tol.zero_current_null, "force at i=0 vs peak"))
    else:
        checks.append(QualityCheck("zero_current_null", True, 0.0, tol.zero_current_null,
                                   "no i=0 column in grid", applicable=False))

    # offset null (odd coupling => ~0 with slug centered)
    if np.isclose(offsets[i_zero], 0.0):
        v = float(np.max(np.abs(force[i_zero, :]))) / peak
        checks.append(QualityCheck("offset_null", v <= tol.offset_null, v, tol.offset_null,
                                   "force at offset=0 vs peak"))
    else:
        checks.append(QualityCheck("offset_null", True, 0.0, tol.offset_null,
                                   "no offset=0 row in grid", applicable=False))

    # far-field decay at the swept edges (peak-current column)
    edge = max(abs(force[0, j_peak]), abs(force[-1, j_peak])) / peak
    checks.append(QualityCheck("far_field_decay", edge <= tol.far_field_decay, edge,
                               tol.far_field_decay, "force at offset edges vs peak"))

    # restoring sign: F * offset * current <= 0 (attraction back toward the coil), scored only
    # where all three are significant so the noisy null regions don't create false positives.
    sig = tol.significant_frac * peak
    prod = force * offsets[:, None] * currents[None, :]
    scored = (np.abs(force) >= sig) & (np.abs(offsets)[:, None] >= offsets.max() * 0.02) & \
             (np.abs(currents)[None, :] >= np.abs(currents).max() * 0.02)
    wrong = (prod > 0.0) & scored
    worst = float(np.max(np.abs(force[wrong]))) / peak if np.any(wrong) else 0.0
    checks.append(QualityCheck("restoring_sign", worst <= tol.restoring_sign, worst,
                               tol.restoring_sign,
                               f"{int(wrong.sum())} wrong-sign points, worst |F| vs peak"))

    # odd symmetry (only if the offset grid is symmetric about 0)
    symmetric = offsets.size >= 3 and np.allclose(offsets, -offsets[::-1], atol=1e-12)
    if symmetric:
        asym = float(np.max(np.abs(force + force[::-1, :]))) / peak
        checks.append(QualityCheck("odd_symmetry", asym <= tol.odd_symmetry, asym,
                                   tol.odd_symmetry, "max |F(-x,i)+F(x,i)| vs peak"))
    else:
        checks.append(QualityCheck("odd_symmetry", True, 0.0, tol.odd_symmetry,
                                   "offset grid not symmetric about 0", applicable=False))

    # current linearity: F/i constant across current at each significant offset (PM slug)
    if expect_linear_current:
        worst_lin = 0.0
        nz = ~np.isclose(currents, 0.0)
        for i in range(offsets.size):
            if np.max(np.abs(force[i, nz])) < sig:
                continue
            ratios = force[i, nz] / currents[nz]
            mean = np.mean(ratios)
            if abs(mean) < 1e-12:
                continue
            spread = (np.max(ratios) - np.min(ratios)) / abs(mean)
            worst_lin = max(worst_lin, float(spread))
        checks.append(QualityCheck("current_linearity", worst_lin <= tol.current_linearity,
                                   worst_lin, tol.current_linearity,
                                   "worst F/i spread across current vs mean"))
    else:
        checks.append(QualityCheck("current_linearity", True, 0.0, tol.current_linearity,
                                   "reluctance slug: force nonlinear in current by design",
                                   applicable=False))

    # monotone tail: the coupling has TWO lobes (a peak at +x and its odd image at -x, with a
    # null between them at offset 0). Beyond EACH lobe's peak, |F| must decay monotonically
    # OUTWARD to the swept edge -- a real dipole tail never climbs again. (The stress-tensor
    # bug's far field oscillated and changed sign, which this catches.) The rise from the
    # center null up to each peak is expected and is NOT scored, so the scan runs outward from
    # each side's peak, never inward across the null.
    col = np.abs(force[:, j_peak])
    worst_bump = 0.0
    pos = np.where(offsets >= 0.0)[0]
    if pos.size:
        ipk = int(pos[np.argmax(col[pos])])
        for i in range(ipk + 1, offsets.size):        # +lobe: peak -> +edge
            worst_bump = max(worst_bump, (col[i] - col[i - 1]) / peak)
    neg = np.where(offsets <= 0.0)[0]
    if neg.size:
        ipk = int(neg[np.argmax(col[neg])])
        for i in range(ipk - 1, -1, -1):              # -lobe: peak -> -edge
            worst_bump = max(worst_bump, (col[i] - col[i + 1]) / peak)
    checks.append(QualityCheck("monotone_tail", worst_bump <= tol.monotone_tail, worst_bump,
                               tol.monotone_tail, "largest |F| increase per step beyond a lobe peak"))

    return LutQualityReport(checks, peak, label)


def check_backend(coil: CoilWindingGeometry, slug: SlugGeometry, backend: FEMBackend,
                  offsets_m: Sequence[float] | None = None,
                  currents_a: Sequence[float] | None = None,
                  tol: QualityTolerances | None = None,
                  expect_linear_current: bool = True) -> LutQualityReport:
    """Sweep a live backend over a geometry and QC the result -- validate a backend/geometry
    before trusting it in a long run. Grid defaults to `default_sweep_ranges`."""
    if offsets_m is None or currents_a is None:
        d_off, d_cur = default_sweep_ranges(coil, slug)
        offsets_m = d_off if offsets_m is None else offsets_m
        currents_a = d_cur if currents_a is None else currents_a
    lut = sweep_coil(coil, slug, backend, offsets_m, currents_a)
    return check_lut(lut, tol=tol, expect_linear_current=expect_linear_current,
                     label=type(backend).__name__)


def scan_lut_files(paths: Sequence[str | Path], tol: QualityTolerances | None = None,
                   expect_linear_current: bool = True) -> dict[str, LutQualityReport]:
    """QC every saved `.npz` ForceLUT in `paths`, keyed by path -- triage a finished run."""
    reports: dict[str, LutQualityReport] = {}
    for p in paths:
        p = Path(p)
        lut = ForceLUT.load(p)
        reports[str(p)] = check_lut(lut, tol=tol, expect_linear_current=expect_linear_current,
                                    label=p.name)
    return reports


def _iter_lut_paths(inputs: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            out.extend(sorted(path.rglob("*.npz")))
        else:
            out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="emac-femqc",
        description="Quality-control force LUTs (.npz) against physical invariants and flag "
                    "the tables a FEM run got wrong. Exits non-zero if any LUT is suspect.")
    parser.add_argument("inputs", nargs="+", help="LUT .npz files and/or directories to scan")
    parser.add_argument("--reluctance-slug", action="store_true",
                        help="skip the current-linearity check (force is nonlinear for an "
                             "iron/reluctance slug)")
    parser.add_argument("--quiet", action="store_true", help="only print suspect LUTs")
    args = parser.parse_args(argv)

    paths = _iter_lut_paths(args.inputs)
    if not paths:
        print("no .npz LUT files found", flush=True)
        return 1
    reports = scan_lut_files(paths, expect_linear_current=not args.reluctance_slug)
    n_suspect = 0
    for path, report in reports.items():
        if not report.ok:
            n_suspect += 1
            print(report)
        elif not args.quiet:
            print(f"{report.label}: OK (peak |F| = {report.peak_force_n:.4f} N)")
    print(f"\n{len(reports)} LUTs scanned, {n_suspect} suspect.", flush=True)
    return 1 if n_suspect else 0


if __name__ == "__main__":
    raise SystemExit(main())
