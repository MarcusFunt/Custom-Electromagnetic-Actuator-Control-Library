import pytest

from emac_sim.fem.femm_backend import FemmBackend, FemmNotAvailableError
from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry
from emac_sim.fem.reference_backend import AnalyticReferenceBackend


def test_femm_backend_raises_clear_error_without_femm_installed():
    try:
        import femm  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("femm is installed -- this checks the not-installed error path specifically")

    with pytest.raises(FemmNotAvailableError, match="femm.info"):
        FemmBackend()


def test_femm_backend_solves_a_point_when_femm_is_installed():
    pytest.importorskip("femm")

    slug = SlugGeometry(magnet_radius_m=0.008, magnet_length_m=0.020, remanence_t=1.2)
    coil = CoilWindingGeometry(position_m=0.0, turns=200, coil_length_m=0.020, radial_thickness_m=0.010)

    with FemmBackend() as backend:
        point = backend.solve(coil, slug, offset_m=0.01, current_a=3.0)

    assert point.force_n == point.force_n  # not NaN


def test_femm_backend_force_sign_matches_reference_backend():
    """Regression test for a real bug found by actually running FEMM (not just plumbing):
    the slug's magnetization-direction angle was inverted, which flipped the ENTIRE force
    curve's sign relative to backend.py's documented contract (positive offset_m + positive
    current_a should attract, i.e. force_n < 0) -- confirmed at the time by comparing
    point-for-point against AnalyticReferenceBackend, which was exactly this backend's
    negative at every sampled offset. A design search's `force_law="femm"` verification
    pass silently failed every design (0 m/s, indistinguishable from a genuinely bad
    design) because the plant was being pushed away from every coil instead of pulled
    toward it. Magnitudes are expected to differ (FEMM vs. a closed-form dipole
    approximation) -- only the SIGN must match, at an offset large enough that both
    backends are unambiguously past their own zero-crossing."""
    pytest.importorskip("femm")

    slug = SlugGeometry(magnet_radius_m=0.008, magnet_length_m=0.020, remanence_t=1.2)
    coil = CoilWindingGeometry(position_m=0.0, turns=200, coil_length_m=0.020, radial_thickness_m=0.010)
    reference = AnalyticReferenceBackend()

    with FemmBackend() as backend:
        for offset_m in (-0.01, 0.01):
            femm_point = backend.solve(coil, slug, offset_m, current_a=3.0)
            reference_point = reference.solve(coil, slug, offset_m, current_a=3.0)
            assert femm_point.force_n * reference_point.force_n > 0, (
                f"offset_m={offset_m}: femm={femm_point.force_n} and "
                f"reference={reference_point.force_n} have opposite signs"
            )
