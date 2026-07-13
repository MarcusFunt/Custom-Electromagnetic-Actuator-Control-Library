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


def test_femm_backend_agrees_in_sign_with_reference_backend():
    """The real FEMM backend must return force with the SAME sign convention as the
    analytic reference backend (which is itself verified against plant.f_current_pm) at
    every (offset, current). They differ in MAGNITUDE -- that's the whole point of FEMM --
    but a sign disagreement means a FEMM-built LUT drives the slug the wrong way, giving 0
    exit speed for every design. This directly guards the sign bug the not-NaN test missed."""
    pytest.importorskip("femm")
    slug = SlugGeometry(magnet_radius_m=0.008, magnet_length_m=0.020, remanence_t=1.2)
    coil = CoilWindingGeometry(position_m=0.0, turns=200, coil_length_m=0.020, radial_thickness_m=0.010)
    ref = AnalyticReferenceBackend()

    with FemmBackend() as backend:
        peak = abs(backend.solve(coil, slug, 0.012, 6.0).force_n)
        for offset_m, current_a in [(0.012, 6.0), (-0.012, 6.0), (0.012, -6.0), (0.020, 3.0)]:
            f_femm = backend.solve(coil, slug, offset_m, current_a).force_n
            f_ref = ref.solve(coil, slug, offset_m, current_a).force_n
            assert f_femm * f_ref > 0.0, (
                f"FEMM force {f_femm:+.3f} N and reference {f_ref:+.3f} N disagree in sign "
                f"at offset={offset_m}, I={current_a} -- a FEMM LUT would drive the slug backward"
            )
        # Offset 0 is a coupling zero: force must be a small fraction of peak (symmetry).
        f_zero = backend.solve(coil, slug, 0.0, 6.0).force_n
        assert abs(f_zero) < 0.05 * peak
