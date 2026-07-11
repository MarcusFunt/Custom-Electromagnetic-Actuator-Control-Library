"""Regression tests for a real bug found by actually running the FEM pipeline (not just
exercising its plumbing): `default_sweep_ranges`' original span was narrow enough that
`ForceLUT`'s edge-clamping left a "phantom" force of several percent of peak on every coil
a slug had already passed, instead of the ~0 the old analytic q_shape() (a fast Gaussian)
correctly gives. In a 5-coil, 0.05 m-pitch stepper the slug is almost always well outside at
least one coil's old (0.04 m half-span) sweep range, so this wasn't an edge case -- it
directly distorted every net_force() call once a FEM LUT was in play. See
docs/FEM_PIPELINE.md and fem/geometry.py's FAR_SPAN_FACTOR/NEAR_REGION_FRACTION docstrings.
"""

import pytest

from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry, default_sweep_ranges
from emac_sim.fem.reference_backend import AnalyticReferenceBackend
from emac_sim.fem.sweep import sweep_coil
from emac_sim.linear_plant import CoilStation, LinearActuatorParams, net_force


def default_slug() -> SlugGeometry:
    return SlugGeometry(magnet_radius_m=0.008, magnet_length_m=0.020, remanence_t=1.2)


def default_coil(**overrides) -> CoilWindingGeometry:
    kwargs = dict(position_m=0.0, turns=200, coil_length_m=0.020, radial_thickness_m=0.010)
    kwargs.update(overrides)
    return CoilWindingGeometry(**kwargs)


def build_default_lut():
    slug, coil = default_slug(), default_coil()
    backend = AnalyticReferenceBackend()
    offsets, currents = default_sweep_ranges(coil, slug)
    return sweep_coil(coil, slug, backend, offsets, currents), slug, coil, offsets, currents


def test_default_sweep_edge_force_is_a_small_fraction_of_peak():
    """The swept range must extend far enough that clamping to its edge is physically
    indistinguishable from the true (decaying-to-zero) far-field force -- not just "some
    margin past the coil," which is what the original 1x-near-field-scale span gave."""
    lut, *_ = build_default_lut()
    peak = float(abs(lut.force_n).max())
    edge = float(abs(lut.force_n[-1, -1]))
    assert edge / peak < 0.01, (
        f"edge force is {edge / peak:.1%} of peak -- ForceLUT will clamp every out-of-range "
        "query to a non-negligible constant instead of the physically-correct ~0"
    )


def test_no_phantom_force_far_downrange_of_a_passed_coil():
    """Once the slug is far past a coil (the normal operating condition partway through a
    multi-coil run), that coil's LUT-derived force must be negligible, not a lingering
    constant. This is the exact failure mode: before the fix, a coil kept exerting ~6% of
    its peak force on a slug that had moved arbitrarily far away."""
    lut, slug, coil, offsets, _ = build_default_lut()
    peak = float(abs(lut.force_n).max())

    far_offset = 1.2  # meters -- comparable to a full multi-coil stepper's travel
    assert far_offset > offsets[-1], "test offset must actually be outside the swept range"
    far_force = abs(lut(far_offset, 6.0))
    assert far_force / peak < 0.01


def test_next_coil_pitch_force_now_decays_instead_of_clamping_to_the_old_narrow_edge():
    """At a typical adjacent-coil pitch (0.05 m for the shipped example config), the force
    should sit meaningfully below what the OLD narrow-span edge value would have clamped to
    -- i.e. the fix actually changes the answer at a physically relevant distance, not just
    at extreme offsets nobody would query."""
    lut, slug, coil, offsets, _ = build_default_lut()
    backend = AnalyticReferenceBackend()

    old_near_span = 1.5 * coil.coil_length_m + 0.5 * slug.magnet_length_m
    old_edge_force = abs(backend.solve(coil, slug, old_near_span, 6.0).force_n)

    pitch_force = abs(lut(0.05, 6.0))
    true_force = abs(backend.solve(coil, slug, 0.05, 6.0).force_n)

    assert pitch_force < old_edge_force
    assert pitch_force == pytest.approx(true_force, rel=0.1)


def test_interpolation_error_against_true_backend_is_small_near_the_coupling_peak():
    """Direct accuracy check of the swept-grid-vs-true-curve error where it matters most:
    the coupling peak and its immediate falloff, which a poorly-allocated grid (this
    project's first attempt spent too much of its budget at offset=0 itself, a ZERO of the
    coupling, not the peak) badly underserved."""
    lut, slug, coil, offsets, currents = build_default_lut()
    backend = AnalyticReferenceBackend()

    near_span = 1.5 * coil.coil_length_m + 0.5 * slug.magnet_length_m
    worst_rel_err = 0.0
    for k in range(200):
        offset = -near_span + 2.0 * near_span * k / 199
        for current in (-6.0, -3.0, 3.0, 6.0):
            direct = backend.solve(coil, slug, offset, current).force_n
            if abs(direct) < 0.5:
                continue
            interp = lut(offset, current)
            worst_rel_err = max(worst_rel_err, abs(interp - direct) / abs(direct))

    assert worst_rel_err < 0.10


def test_multi_coil_net_force_has_no_residual_push_once_slug_has_cleared_every_coil():
    """System-level check with net_force() itself (not just the LUT in isolation): a slug
    positioned well past every energized coil in a stepper-like layout should feel
    negligible net force -- this is what the phantom-force bug broke in practice (see the
    task-9 diagnosis), since 5 fixed LUTs summed 5 non-vanishing constants."""
    slug = default_slug()
    backend = AnalyticReferenceBackend()

    pitch = 0.05
    n_coils = 5
    coils = []
    for k in range(n_coils):
        coil_geom = default_coil(position_m=k * pitch)
        offsets, currents = default_sweep_ranges(coil_geom, slug)
        lut = sweep_coil(coil_geom, slug, backend, offsets, currents)
        coils.append(CoilStation(position_m=coil_geom.position_m, force_lut=lut))

    p = LinearActuatorParams(coils=tuple(coils))
    currents_a = [6.0] * n_coils   # worst case: every coil still energized

    far_past_every_coil = coils[-1].position_m + 1.0  # 1 m past the last coil
    force = net_force(far_past_every_coil, currents_a, p)

    # Compare against the force right at the peak of one coil, as a sense of scale.
    peak_scale = abs(net_force(0.0104, [6.0] + [0.0] * (n_coils - 1), p))
    assert abs(force) / peak_scale < 0.05
