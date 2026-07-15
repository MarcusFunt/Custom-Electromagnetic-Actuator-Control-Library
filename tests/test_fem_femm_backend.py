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


# A representative coil/slug used by the quantitative checks below. Forces here are ~0.1-1 N
# (well above FEMM's numerical floor), so relative comparisons are meaningful.
_Q_SLUG = SlugGeometry(magnet_radius_m=0.006, magnet_length_m=0.020, remanence_t=1.2)
_Q_COIL = CoilWindingGeometry(position_m=0.0, turns=400, coil_length_m=0.030,
                              radial_thickness_m=0.020)


def test_femm_backend_magnitude_agrees_with_reference_backend():
    """The real FEMM force must agree in MAGNITUDE (not just sign) with the closed-form
    AnalyticReferenceBackend -- both compute the same physical PM+air-coil interaction, so
    they should match to a few percent. This is the check that would have caught the
    force-extraction bug where the weighted stress tensor over the magnet block returned
    forces up to ~2x too large: the old test only compared SIGN, which happened to be right
    near-field. Tolerance is deliberately loose (15%) to allow the magnet's mu_r=1.05
    reluctance term and mesh discretization, but tight enough to fail on a 2x error."""
    pytest.importorskip("femm")
    ref = AnalyticReferenceBackend()
    scale = 1.5 * _Q_COIL.coil_length_m + 0.5 * _Q_SLUG.magnet_length_m
    with FemmBackend() as backend:
        for frac in (0.35, 0.6, 1.0):
            offset = frac * scale
            f_femm = backend.solve(_Q_COIL, _Q_SLUG, offset, 3.0).force_n
            f_ref = ref.solve(_Q_COIL, _Q_SLUG, offset, 3.0).force_n
            rel = abs(f_femm - f_ref) / abs(f_ref)
            assert rel < 0.15, (
                f"FEMM force {f_femm:+.4f} N disagrees with reference {f_ref:+.4f} N by "
                f"{rel*100:.0f}% at offset={offset*1e3:.1f}mm (>15%)"
            )


def test_femm_backend_force_converges_under_mesh_refinement():
    """Force must be stable under mesh refinement -- a well-posed extraction converges. The
    old weighted-stress-tensor-over-the-magnet extraction did NOT: it swung by 2x+ and even
    flipped sign between mesh sizes. The Lorentz-force-on-the-coil extraction is mesh-robust
    because the coil is linear, non-magnetic, and carries a known current density."""
    pytest.importorskip("femm")
    scale = 1.5 * _Q_COIL.coil_length_m + 0.5 * _Q_SLUG.magnet_length_m
    offset = 0.6 * scale
    fine = 0.05 * min(_Q_COIL.radial_thickness_m, _Q_SLUG.magnet_radius_m)
    coarse = 0.15 * min(_Q_COIL.radial_thickness_m, _Q_SLUG.magnet_radius_m)
    with FemmBackend(mesh_size_m=coarse) as b_coarse:
        f_coarse = b_coarse.solve(_Q_COIL, _Q_SLUG, offset, 3.0).force_n
    with FemmBackend(mesh_size_m=fine) as b_fine:
        f_fine = b_fine.solve(_Q_COIL, _Q_SLUG, offset, 3.0).force_n
    rel = abs(f_coarse - f_fine) / abs(f_fine)
    assert rel < 0.05, (
        f"FEMM force not mesh-convergent: coarse {f_coarse:+.4f} N vs fine {f_fine:+.4f} N "
        f"differ by {rel*100:.0f}% (>5%) -- extraction is ill-posed"
    )


def test_femm_backend_force_is_linear_in_current():
    """A PM slug in an air (mu_r=1) coil gives a force linear in coil current to well under
    1% (the mu_r=1.05 magnet contributes a negligible i^2 term). F/i must therefore be
    nearly constant. The old stress-tensor extraction showed a spurious ~50% drift in F/i
    across currents; the coil-Lorentz extraction is linear by construction."""
    pytest.importorskip("femm")
    scale = 1.5 * _Q_COIL.coil_length_m + 0.5 * _Q_SLUG.magnet_length_m
    offset = 0.4 * scale
    with FemmBackend() as backend:
        ratios = [backend.solve(_Q_COIL, _Q_SLUG, offset, i).force_n / i
                  for i in (1.0, 3.0, 6.0)]
    spread = (max(ratios) - min(ratios)) / abs(sum(ratios) / len(ratios))
    assert spread < 0.03, f"F/i not constant across current (spread {spread*100:.1f}% > 3%): {ratios}"
