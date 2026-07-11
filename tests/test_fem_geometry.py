import pytest

from emac_sim.fem.geometry import (
    CoilWindingGeometry,
    SlugGeometry,
    default_sweep_ranges,
    wire_length_m,
)


def default_slug(**overrides) -> SlugGeometry:
    kwargs = dict(magnet_radius_m=0.008, magnet_length_m=0.020, remanence_t=1.2)
    kwargs.update(overrides)
    return SlugGeometry(**kwargs)


def default_coil(**overrides) -> CoilWindingGeometry:
    kwargs = dict(position_m=0.0, turns=200, coil_length_m=0.020, radial_thickness_m=0.010)
    kwargs.update(overrides)
    return CoilWindingGeometry(**kwargs)


def test_slug_geometry_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError):
        default_slug(magnet_radius_m=0.0)
    with pytest.raises(ValueError):
        default_slug(magnet_length_m=-0.01)


def test_coil_geometry_rejects_bad_values():
    with pytest.raises(ValueError):
        default_coil(turns=0)
    with pytest.raises(ValueError):
        default_coil(coil_length_m=0.0)
    with pytest.raises(ValueError):
        default_coil(radial_thickness_m=-0.001)


def test_bore_mean_outer_radius_are_consistent():
    slug = default_slug()
    coil = default_coil(bore_clearance_m=0.0015, radial_thickness_m=0.01)

    bore = coil.bore_radius_m(slug)
    mean = coil.mean_radius_m(slug)
    outer = coil.outer_radius_m(slug)

    assert bore == pytest.approx(slug.magnet_radius_m + coil.bore_clearance_m)
    assert bore < mean < outer
    assert mean == pytest.approx(bore + 0.5 * coil.radial_thickness_m)
    assert outer == pytest.approx(bore + coil.radial_thickness_m)


def test_wire_length_scales_with_turns_and_mean_radius():
    slug = default_slug()
    coil = default_coil(turns=100)
    coil_more_turns = default_coil(turns=200)

    assert wire_length_m(coil_more_turns, slug) == pytest.approx(2.0 * wire_length_m(coil, slug))


def test_default_sweep_ranges_are_symmetric_and_include_zero():
    slug = default_slug()
    coil = default_coil()
    offsets, currents = default_sweep_ranges(coil, slug, n_offsets=11, n_currents=7, max_current_a=6.0)

    assert len(offsets) == 11
    assert len(currents) == 7
    assert offsets[0] == pytest.approx(-offsets[-1])
    assert currents[0] == pytest.approx(-6.0)
    assert currents[-1] == pytest.approx(6.0)
    # odd counts include an exact zero -- important so the LUT captures the
    # known-zero-at-center/zero-current operating points exactly, not just near them.
    mid_offset = offsets[len(offsets) // 2]
    mid_current = currents[len(currents) // 2]
    assert mid_offset == pytest.approx(0.0, abs=1e-12)
    assert mid_current == pytest.approx(0.0, abs=1e-12)
    assert sorted(offsets) == list(offsets)
    assert sorted(currents) == list(currents)


@pytest.mark.parametrize("n_offsets", [2, 3, 4, 5, 6, 7, 11, 21, 30, 31, 41])
def test_default_sweep_ranges_offsets_are_strictly_increasing_for_any_count(n_offsets):
    """ForceLUT requires a strictly increasing axis (np.diff > 0 everywhere) -- the
    two-region grid must never emit a duplicate or out-of-order point, at any n_offsets a
    caller might pass (emac-femgen's --n-offsets is a free CLI integer)."""
    slug = default_slug()
    coil = default_coil()
    offsets, _ = default_sweep_ranges(coil, slug, n_offsets=n_offsets, n_currents=3)
    assert len(offsets) == n_offsets
    assert all(b > a for a, b in zip(offsets, offsets[1:]))


def test_default_sweep_ranges_span_reaches_far_span_factor():
    from emac_sim.fem.geometry import FAR_SPAN_FACTOR

    slug = default_slug()
    coil = default_coil()
    near_span = 1.5 * coil.coil_length_m + 0.5 * slug.magnet_length_m
    offsets, _ = default_sweep_ranges(coil, slug)
    assert offsets[-1] == pytest.approx(FAR_SPAN_FACTOR * near_span)
    assert offsets[0] == pytest.approx(-FAR_SPAN_FACTOR * near_span)
