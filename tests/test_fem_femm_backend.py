import pytest

from emac_sim.fem.femm_backend import (
    FemmBackend,
    FemmNotAvailableError,
    _perimeter_seeds,
    _trace_field_line,
)
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


def test_trace_field_line_follows_a_synthetic_uniform_field():
    """_trace_field_line takes a plain (r,z) -> (br,bz) callable, not the femm module --
    verify its RK4 integration is correct against a trivial synthetic field (no FEMM
    needed, always runs): a uniform field pointing in +z should trace a straight line of
    constant r, monotonically increasing z, stepping roughly step_m per point."""
    def uniform_bz(r, z):
        return (0.0, 1.0)

    def in_bounds(r, z):
        return -1.0 <= r <= 1.0 and -1.0 <= z <= 1.0

    seed = (0.1, 0.0)
    step_m = 0.05
    line = _trace_field_line(uniform_bz, seed, step_m, +1.0, in_bounds, max_points=20)

    assert len(line) == 20
    for r, _z in line:
        assert r == pytest.approx(0.1, abs=1e-9)
    zs = [z for _r, z in line]
    assert zs == sorted(zs)
    assert zs[-1] == pytest.approx((len(line) - 1) * step_m, abs=1e-9)


def test_trace_field_line_stops_when_leaving_bounds():
    def uniform_bz(r, z):
        return (0.0, 1.0)

    def in_bounds(r, z):
        return z <= 0.12

    line = _trace_field_line(uniform_bz, (0.0, 0.0), 0.05, +1.0, in_bounds, max_points=100)
    assert 2 <= len(line) < 100
    assert all(z <= 0.12 for _r, z in line)


def test_perimeter_seeds_stay_on_the_magnets_real_boundary_faces():
    """Seeds must land on the magnet's physical surface (bottom cap, outer side, top cap)
    and never on the r=0 symmetry axis (not a real boundary) or outside [z0,z1]/[0,R]."""
    seeds = _perimeter_seeds(magnet_radius_m=0.01, z0=-0.02, z1=0.02, n=12)
    assert len(seeds) == 12
    for r, z in seeds:
        assert 0.0 <= r <= 0.01 + 1e-12
        assert -0.02 - 1e-12 <= z <= 0.02 + 1e-12
        on_bottom_cap = z == pytest.approx(-0.02)
        on_top_cap = z == pytest.approx(0.02)
        on_outer_side = r == pytest.approx(0.01)
        assert on_bottom_cap or on_top_cap or on_outer_side


def test_femm_backend_traces_field_lines_when_femm_is_installed():
    pytest.importorskip("femm")

    slug = SlugGeometry(magnet_radius_m=0.008, magnet_length_m=0.020, remanence_t=1.2)
    coil = CoilWindingGeometry(position_m=0.0, turns=200, coil_length_m=0.020, radial_thickness_m=0.010)

    with FemmBackend() as backend:
        lines = backend.field_lines(coil, slug, offset_m=0.0, current_a=5.0, n_lines=8)

    assert len(lines) > 0
    for line in lines:
        assert len(line) >= 2
        for r, z in line:
            assert r == r and z == z  # not NaN
            assert r >= -1e-9
