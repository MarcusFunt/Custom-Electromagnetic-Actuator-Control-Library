import numpy as np
import pytest

from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry
from emac_sim.fem.lut import ForceLUT
from emac_sim.fem.quality import (QualityTolerances, check_backend, check_lut,
                                  scan_lut_files)
from emac_sim.fem.reference_backend import AnalyticReferenceBackend

_COIL = CoilWindingGeometry(0.0, turns=400, coil_length_m=0.03, radial_thickness_m=0.02)
_SLUG = SlugGeometry(magnet_radius_m=0.006, magnet_length_m=0.02, remanence_t=1.2)


def _good_lut() -> ForceLUT:
    from emac_sim.fem.sweep import sweep_coil
    return sweep_coil(_COIL, _SLUG, AnalyticReferenceBackend())


def test_clean_reference_lut_passes_every_check():
    report = check_backend(_COIL, _SLUG, AnalyticReferenceBackend())
    assert report.ok, str(report)
    # every applicable check actually ran (nothing silently skipped except maybe none here)
    assert all(c.passed for c in report.checks if c.applicable)


def test_nan_fails_finite_check():
    lut = _good_lut()
    f = lut.force_n.copy()
    f[3, 1] = np.nan
    report = check_lut(ForceLUT(lut.offsets_m, lut.currents_a, f))
    assert not report.ok
    assert "finite" in {c.name for c in report.failures()}


def test_far_field_sign_flip_is_flagged():
    """A stress-tensor-style far-field sign flip + amplification just past the peak must trip
    the sign / symmetry / monotone-tail checks -- the exact failure the FEMM bug produced."""
    lut = _good_lut()
    f = lut.force_n.copy()
    ipk = int(np.argmax(np.abs(f[:, -1])))
    f[ipk + 3, :] = -2.0 * f[ipk + 3, :]
    report = check_lut(ForceLUT(lut.offsets_m, lut.currents_a, f))
    assert not report.ok
    failed = {c.name for c in report.failures()}
    assert "monotone_tail" in failed
    assert failed & {"restoring_sign", "odd_symmetry"}


def test_current_nonlinearity_is_flagged():
    """A force whose F/i drifts with |current| (the stress-tensor artifact) fails linearity."""
    lut = _good_lut()
    f = lut.force_n.copy()
    cmax = np.abs(lut.currents_a).max()
    for j, cur in enumerate(lut.currents_a):
        if cur != 0.0:
            f[:, j] *= (1.0 + 0.3 * abs(cur) / cmax)
    report = check_lut(ForceLUT(lut.offsets_m, lut.currents_a, f))
    assert not report.ok
    assert "current_linearity" in {c.name for c in report.failures()}


def test_reluctance_flag_skips_current_linearity():
    """A reluctance slug is legitimately nonlinear in current, so the check must be SKIPPED
    (not failed) when the caller says so."""
    lut = _good_lut()
    f = lut.force_n.copy()
    cmax = np.abs(lut.currents_a).max()
    for j, cur in enumerate(lut.currents_a):
        if cur != 0.0:
            f[:, j] *= (1.0 + 0.3 * abs(cur) / cmax)
    report = check_lut(ForceLUT(lut.offsets_m, lut.currents_a, f), expect_linear_current=False)
    lin = next(c for c in report.checks if c.name == "current_linearity")
    assert not lin.applicable
    # the nonlinearity itself is no longer counted as a failure
    assert "current_linearity" not in {c.name for c in report.failures()}


def test_truncated_tail_fails_far_field_decay():
    """If the swept offset range is too narrow, the edge force is a large fraction of peak;
    clamping the LUT there injects a phantom force, so the check must fail."""
    lut = _good_lut()
    # keep only the dense near region around the peak -> edges are still a big fraction of peak
    n = lut.offsets_m.size
    keep = slice(n // 2 - 3, n // 2 + 4)
    truncated = ForceLUT(lut.offsets_m[keep], lut.currents_a, lut.force_n[keep, :])
    report = check_lut(truncated)
    assert "far_field_decay" in {c.name for c in report.failures()}


def test_tolerances_are_configurable():
    """A LUT with a known ~10% F/i nonlinearity passes a loose linearity tol and fails a
    tight one -- the tolerance, not just the data, decides the verdict."""
    lut = _good_lut()
    f = lut.force_n.copy()
    cmax = np.abs(lut.currents_a).max()
    for j, cur in enumerate(lut.currents_a):
        if cur != 0.0:
            f[:, j] *= (1.0 + 0.10 * abs(cur) / cmax)
    corrupted = ForceLUT(lut.offsets_m, lut.currents_a, f)
    assert check_lut(corrupted, tol=QualityTolerances(current_linearity=0.30)).ok
    strict = check_lut(corrupted, tol=QualityTolerances(current_linearity=0.02))
    assert "current_linearity" in {c.name for c in strict.failures()}


def test_scan_lut_files_round_trips_saved_luts(tmp_path):
    good = _good_lut()
    good.save(tmp_path / "good.npz")
    bad_f = good.force_n.copy()
    bad_f[2, :] = np.nan
    ForceLUT(good.offsets_m, good.currents_a, bad_f).save(tmp_path / "bad.npz")
    reports = scan_lut_files([tmp_path / "good.npz", tmp_path / "bad.npz"])
    assert reports[str(tmp_path / "good.npz")].ok
    assert not reports[str(tmp_path / "bad.npz")].ok
