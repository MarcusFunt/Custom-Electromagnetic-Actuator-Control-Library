import pytest

from emac_sim.fem.femm_backend import FemmBackend, FemmNotAvailableError
from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry


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
