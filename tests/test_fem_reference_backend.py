import pytest

from emac_sim.coil_design import off_axis_radial_field_cylinder_magnet
from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry, wire_length_m
from emac_sim.fem.reference_backend import AnalyticReferenceBackend


def default_slug() -> SlugGeometry:
    return SlugGeometry(magnet_radius_m=0.008, magnet_length_m=0.020, remanence_t=1.2)


def default_coil() -> CoilWindingGeometry:
    return CoilWindingGeometry(position_m=0.0, turns=200, coil_length_m=0.020, radial_thickness_m=0.010)


def test_force_is_zero_at_zero_current():
    backend = AnalyticReferenceBackend()
    slug, coil = default_slug(), default_coil()
    point = backend.solve(coil, slug, offset_m=0.01, current_a=0.0)
    assert point.force_n == pytest.approx(0.0)


def test_force_is_zero_at_zero_offset():
    """The coil sits exactly at the slug's field null (odd field about the magnet's own
    center) -- same zero-at-own-center property plant.q_shape has by construction."""
    backend = AnalyticReferenceBackend()
    slug, coil = default_slug(), default_coil()
    point = backend.solve(coil, slug, offset_m=0.0, current_a=4.0)
    assert point.force_n == pytest.approx(0.0, abs=1e-9)


def test_force_is_odd_in_offset():
    backend = AnalyticReferenceBackend()
    slug, coil = default_slug(), default_coil()
    plus = backend.solve(coil, slug, offset_m=0.012, current_a=3.0)
    minus = backend.solve(coil, slug, offset_m=-0.012, current_a=3.0)
    assert minus.force_n == pytest.approx(-plus.force_n)


def test_force_is_linear_in_current():
    backend = AnalyticReferenceBackend()
    slug, coil = default_slug(), default_coil()
    one_amp = backend.solve(coil, slug, offset_m=0.01, current_a=1.0)
    three_amp = backend.solve(coil, slug, offset_m=0.01, current_a=3.0)
    assert three_amp.force_n == pytest.approx(3.0 * one_amp.force_n)


def test_positive_current_attracts_slug_back_toward_coil():
    """i>0 with the slug ahead of the coil (offset>0) should pull it back -- negative
    force -- matching plant.f_current_pm's sign convention (i>0 attracts)."""
    backend = AnalyticReferenceBackend()
    slug, coil = default_slug(), default_coil()
    ahead = backend.solve(coil, slug, offset_m=0.01, current_a=2.0)
    behind = backend.solve(coil, slug, offset_m=-0.01, current_a=2.0)
    assert ahead.force_n < 0.0
    assert behind.force_n > 0.0


def test_negative_current_repels():
    backend = AnalyticReferenceBackend()
    slug, coil = default_slug(), default_coil()
    point = backend.solve(coil, slug, offset_m=0.01, current_a=-2.0)
    assert point.force_n > 0.0


def _single_point_force(coil, slug, offset_m, current_a):
    """The OLD reference-backend force: B_rho sampled at the single (mean-radius, coil-center)
    point times total wire length -- i.e. every turn assumed to sit at the field peak."""
    b_rho = off_axis_radial_field_cylinder_magnet(
        coil.mean_radius_m(slug), -offset_m, slug.magnet_radius_m, slug.magnet_length_m,
        slug.remanence_t,
    )
    return current_a * b_rho * wire_length_m(coil, slug)


def test_force_is_winding_averaged_not_single_point_at_the_peak():
    """The accuracy fix: the winding spans a finite axial length and radial build, and most
    of it sees LESS radial field than the single mean-radius/coil-center point does (that
    point sits at the field maximum). Averaging over the cross-section must therefore give a
    smaller peak force per amp than the old single-point estimate -- here by a clearly
    non-negligible margin -- while the uniform-field-limit reduction keeps every sign and
    symmetry property (covered by the tests above)."""
    backend = AnalyticReferenceBackend()
    slug, coil = default_slug(), default_coil()

    # Scan for each method's peak |force per amp|.
    averaged_peak = 0.0
    single_peak = 0.0
    for k in range(1, 60):
        offset = 0.001 * k
        averaged_peak = max(averaged_peak, abs(backend.solve(coil, slug, offset, 1.0).force_n))
        single_peak = max(single_peak, abs(_single_point_force(coil, slug, offset, 1.0)))

    # The single-point estimate over-states the peak coupling by a large margin for a coil
    # whose length is comparable to the coupling scale.
    assert averaged_peak < 0.85 * single_peak


def test_force_reduces_to_single_point_for_a_vanishingly_small_winding():
    """Sanity: for a winding whose axial length and radial build both shrink toward zero,
    the cross-section average must collapse back onto the single-point value -- confirming
    the averaged form is a strict generalization, not a different model."""
    backend = AnalyticReferenceBackend()
    slug = default_slug()
    tiny = CoilWindingGeometry(position_m=0.0, turns=200, coil_length_m=1e-5,
                               radial_thickness_m=1e-5)
    for offset in (0.005, 0.012, 0.02):
        averaged = backend.solve(tiny, slug, offset, 3.0).force_n
        single = _single_point_force(tiny, slug, offset, 3.0)
        assert averaged == pytest.approx(single, rel=1e-3)
