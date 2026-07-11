import pytest

from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry
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
