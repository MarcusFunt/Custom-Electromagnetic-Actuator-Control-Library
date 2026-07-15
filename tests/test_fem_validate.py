import numpy as np
import pytest

from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry
from emac_sim.fem.reference_backend import AnalyticReferenceBackend
from emac_sim.fem.validate import BackendComparison, compare_backends


def _grid():
    offsets = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    currents = np.array([-3.0, 0.0, 3.0])
    return offsets, currents


def test_backend_comparison_zero_error_for_identical_grids():
    offsets, currents = _grid()
    force = np.outer(np.linspace(-1, 1, 5), np.linspace(-1, 1, 3))
    cmp = BackendComparison(offsets, currents, force, force.copy())
    assert cmp.max_relative_error() == 0.0
    assert cmp.mean_relative_error() == 0.0


def test_backend_comparison_relative_error_normalizes_by_peak_and_floors_far_field():
    offsets, currents = _grid()
    a = np.zeros((5, 3))
    b = np.zeros((5, 3))
    a[0, 0] = 1.0            # peak
    b[0, 0] = 1.1           # 10% of peak error at the peak point
    # a tiny far-field point: both ~1% of peak, large RAW ratio, must be floored out
    a[4, 2] = 0.010
    b[4, 2] = 0.001
    cmp = BackendComparison(offsets, currents, a, b)
    # peak is 1.1; error at peak point is |1.0-1.1|/1.1 = 9.1%
    assert cmp.max_relative_error(floor_frac=0.05) == pytest.approx(0.1 / 1.1, rel=1e-9)


def test_compare_backends_against_itself_is_zero_error():
    coil = CoilWindingGeometry(0.0, turns=300, coil_length_m=0.02, radial_thickness_m=0.01)
    slug = SlugGeometry(magnet_radius_m=0.006, magnet_length_m=0.02, remanence_t=1.2)
    ref = AnalyticReferenceBackend()
    cmp = compare_backends(coil, slug, ref, ref, label_a="ref", label_b="ref2")
    assert cmp.max_relative_error() == 0.0
    assert "ref vs ref2" in cmp.report()


def test_analytic_agrees_with_real_femm_over_a_full_sweep():
    """The headline research-tool guarantee: over a whole (offset, current) sweep, the fast
    analytic coupling model agrees with a real FEMM solve to within a bound. This is the
    automated, geometry-general version of the point checks in test_fem_femm_backend.py."""
    pytest.importorskip("femm")
    from emac_sim.fem.validate import compare_analytic_to_femm
    coil = CoilWindingGeometry(0.0, turns=400, coil_length_m=0.030, radial_thickness_m=0.020)
    slug = SlugGeometry(magnet_radius_m=0.006, magnet_length_m=0.020, remanence_t=1.2)
    # A modest grid keeps this FEMM-gated test to a few solves.
    offsets = np.linspace(-0.06, 0.06, 9)
    currents = np.array([-4.0, 0.0, 4.0])
    cmp = compare_analytic_to_femm(coil, slug, offsets_m=offsets, currents_a=currents)
    assert cmp.max_relative_error(floor_frac=0.05) < 0.10, "\n" + cmp.report()
