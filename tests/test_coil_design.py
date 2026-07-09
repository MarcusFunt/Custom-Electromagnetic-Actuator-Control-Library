import math

import pytest

from emac_sim.coil_design import (
    COPPER_RESISTIVITY_20C_OHM_M,
    MU_0,
    build_coil_station,
    copper_resistivity_ohm_m,
    estimate_k_a,
    magnet_mass_kg,
    off_axis_field_cylinder_magnet,
    on_axis_field_cylinder_magnet,
    wind_coil,
)


def test_more_turns_in_a_fixed_envelope_raises_resistance_and_inductance():
    """Turns and coil dimensions aren't independent: packing more turns into the same
    envelope means thinner wire, so both R and L should climb faster than linearly."""
    windings = [
        wind_coil(n, coil_length_m=0.02, radial_thickness_m=0.01, bore_radius_m=0.008)
        for n in (50, 100, 200, 400)
    ]
    resistances = [w.resistance_ohm for w in windings]
    inductances = [w.inductance_h for w in windings]
    wire_diameters = [w.wire_diameter_m for w in windings]

    assert resistances == sorted(resistances)
    assert inductances == sorted(inductances)
    assert wire_diameters == sorted(wire_diameters, reverse=True)

    # R and L both scale ~N^2 for a fixed envelope -- doubling turns should roughly
    # quadruple each, not just double.
    ratio_r = resistances[1] / resistances[0]     # 100 turns / 50 turns
    ratio_l = inductances[1] / inductances[0]
    assert ratio_r > 3.5
    assert ratio_l > 3.5


def test_wind_coil_rejects_invalid_turns():
    with pytest.raises(ValueError):
        wind_coil(0, coil_length_m=0.02, radial_thickness_m=0.01, bore_radius_m=0.008)


def test_bigger_magnet_increases_mass_and_k_a():
    radii = (0.004, 0.006, 0.008)
    masses = [magnet_mass_kg(r, magnet_length_m=0.02) for r in radii]
    k_as = [
        build_coil_station(0.0, turns=200, coil_length_m=0.02, radial_thickness_m=0.01,
                            magnet_radius_m=r, magnet_length_m=0.02, remanence_t=1.3).k_a
        for r in radii
    ]
    assert masses == sorted(masses)
    assert k_as == sorted(k_as)


def test_on_axis_field_decays_with_distance_and_is_positive_near_the_magnet():
    near = on_axis_field_cylinder_magnet(0.0005, magnet_radius_m=0.006, magnet_length_m=0.02,
                                          remanence_t=1.3)
    far = on_axis_field_cylinder_magnet(0.05, magnet_radius_m=0.006, magnet_length_m=0.02,
                                         remanence_t=1.3)
    assert near > 0.0
    assert far > 0.0
    assert far < near


def test_stronger_remanence_gives_more_k_a_and_field():
    weak = on_axis_field_cylinder_magnet(0.0015, 0.006, 0.02, remanence_t=0.4)   # ferrite-like
    strong = on_axis_field_cylinder_magnet(0.0015, 0.006, 0.02, remanence_t=1.3)  # NdFeB-like
    assert strong > weak

    k_a_weak = estimate_k_a(turns=200, mean_radius_m=0.008, magnet_radius_m=0.006,
                             magnet_length_m=0.02, remanence_t=0.4)
    k_a_strong = estimate_k_a(turns=200, mean_radius_m=0.008, magnet_radius_m=0.006,
                               magnet_length_m=0.02, remanence_t=1.3)
    assert k_a_strong > k_a_weak


def test_build_coil_station_is_pure_pm_no_reluctance_term():
    coil = build_coil_station(0.1, turns=150, coil_length_m=0.02, radial_thickness_m=0.01,
                               magnet_radius_m=0.006, magnet_length_m=0.02, remanence_t=1.3)
    assert coil.Cmag == 0.0
    assert coil.k_a > 0.0
    assert coil.position_m == pytest.approx(0.1)
    assert coil.resistance_ohm > 0.0
    assert coil.inductance_h > 0.0


def test_wind_coil_matches_a_direct_hand_calculation():
    """Cross-check the formula itself (not just its monotonic trend) against the same
    arithmetic worked out independently, for one concrete set of numbers."""
    turns, coil_length, radial_thickness, bore_radius = 120, 0.025, 0.008, 0.006
    packing = 0.8

    winding_area = coil_length * radial_thickness
    copper_area = winding_area * packing
    wire_area = copper_area / turns
    expected_wire_diameter = math.sqrt(4.0 * wire_area / math.pi)

    mean_radius = bore_radius + 0.5 * radial_thickness
    mean_turn_length = 2.0 * math.pi * mean_radius
    expected_resistance = COPPER_RESISTIVITY_20C_OHM_M * (turns * mean_turn_length) / wire_area
    # Wheeler's (1928) single-layer air-core solenoid formula
    expected_inductance = 3.937e-5 * mean_radius**2 * turns**2 / (9.0 * mean_radius + 10.0 * coil_length)

    w = wind_coil(turns, coil_length, radial_thickness, bore_radius, packing_factor=packing)
    assert w.wire_diameter_m == pytest.approx(expected_wire_diameter)
    assert w.resistance_ohm == pytest.approx(expected_resistance)
    assert w.inductance_h == pytest.approx(expected_inductance)
    assert w.mean_radius_m == pytest.approx(mean_radius)


def test_wheeler_inductance_matches_long_solenoid_formula_in_the_long_coil_limit():
    """Wheeler's formula should agree with the elementary long-solenoid formula
    (L = mu0*N^2*A/length) once the coil is much longer than its radius -- that's the
    regime the plain formula is actually derived for."""
    turns, radius, length = 200, 0.006, 5.0   # length >> radius
    w = wind_coil(turns, coil_length_m=length, radial_thickness_m=1e-6, bore_radius_m=radius)
    long_solenoid = MU_0 * turns**2 * math.pi * w.mean_radius_m**2 / length
    assert w.inductance_h == pytest.approx(long_solenoid, rel=0.01)


def test_wheeler_and_long_solenoid_diverge_for_a_short_fat_coil():
    """For a short, fat coil (length << radius) -- exactly the regime the optimizer
    explores -- Wheeler's should give a meaningfully SMALLER inductance than the plain
    long-solenoid formula, which overestimates outside its valid regime."""
    turns, radius, length = 200, 0.02, 0.001
    w = wind_coil(turns, coil_length_m=length, radial_thickness_m=1e-6, bore_radius_m=radius)
    long_solenoid = MU_0 * turns**2 * math.pi * w.mean_radius_m**2 / length
    assert w.inductance_h < 0.5 * long_solenoid


def test_copper_resistivity_increases_with_temperature():
    r_cold = copper_resistivity_ohm_m(0.0)
    r_ref = copper_resistivity_ohm_m(20.0)
    r_hot = copper_resistivity_ohm_m(100.0)
    assert r_cold < r_ref < r_hot
    assert r_ref == pytest.approx(COPPER_RESISTIVITY_20C_OHM_M)


def test_wind_coil_at_higher_temperature_has_higher_resistance_same_inductance():
    """Heating the coil changes only resistivity (hence resistance); inductance is a
    purely geometric property and must be unaffected."""
    cold = wind_coil(150, 0.02, 0.008, 0.006, temperature_c=20.0)
    hot = wind_coil(150, 0.02, 0.008, 0.006, temperature_c=100.0)
    assert hot.resistance_ohm > cold.resistance_ohm
    assert hot.inductance_h == pytest.approx(cold.inductance_h)


def test_off_axis_field_matches_on_axis_formula_in_the_rho_to_zero_limit():
    """off_axis_field_cylinder_magnet is a more general (and more expensive) calculation
    that should reduce EXACTLY to on_axis_field_cylinder_magnet's closed form as rho->0 --
    this is the key correctness check for the whole off-axis implementation."""
    mr, ml, br = 0.006, 0.02, 1.3
    for z in (0.0005, 0.002, 0.01, 0.05):
        on_axis = on_axis_field_cylinder_magnet(z, mr, ml, br)
        off_axis_near_zero = off_axis_field_cylinder_magnet(1e-9, z, mr, ml, br)
        assert off_axis_near_zero == pytest.approx(on_axis, rel=1e-4)


def test_off_axis_field_weakens_moving_away_from_the_axis():
    mr, ml, br = 0.006, 0.02, 1.3
    z = 0.0015
    on_axis = off_axis_field_cylinder_magnet(0.0, z, mr, ml, br)
    off_axis = off_axis_field_cylinder_magnet(0.004, z, mr, ml, br)
    assert 0.0 < off_axis < on_axis


def test_off_axis_field_at_the_magnets_own_center_approaches_remanence_for_a_long_magnet():
    """On-axis, deep inside a long magnetized rod, the equivalent-surface-current model
    should give a field approaching the material's own remanence -- the standard result
    for a long magnetized cylinder's interior on-axis field."""
    mr, ml, br = 0.006, 2.0, 1.3
    b_center = off_axis_field_cylinder_magnet(1e-9, -ml / 2.0, mr, ml, br)
    assert b_center == pytest.approx(br, rel=1e-2)


def test_thicker_radial_winding_couples_more_weakly_once_past_the_magnets_radius():
    """The real effect the old on-axis-at-radial-gap approximation couldn't see: a coil
    whose mean radius sits well outside the magnet's own radius couples much more weakly
    than one sitting right at the bore -- confirmed here via estimate_k_a directly."""
    magnet_radius, magnet_length, remanence, turns = 0.006, 0.02, 1.3, 200
    k_a_thin = estimate_k_a(turns, mean_radius_m=magnet_radius + 0.0015,
                            magnet_radius_m=magnet_radius, magnet_length_m=magnet_length,
                            remanence_t=remanence)
    k_a_thick = estimate_k_a(turns, mean_radius_m=magnet_radius + 0.02,
                             magnet_radius_m=magnet_radius, magnet_length_m=magnet_length,
                             remanence_t=remanence)
    assert k_a_thick < k_a_thin


def test_on_axis_field_is_bounded_between_zero_and_remanence():
    remanence = 1.3
    for z in (0.0001, 0.001, 0.01, 0.1, 1.0):
        b = on_axis_field_cylinder_magnet(z, magnet_radius_m=0.006, magnet_length_m=0.02,
                                          remanence_t=remanence)
        assert 0.0 < b < remanence


def test_on_axis_field_at_the_face_approaches_half_remanence_for_a_long_magnet():
    """The classic semi-infinite-rod result: right at the pole face of an very long
    magnetized rod, the on-axis field approaches exactly Br/2. A finite-but-long magnet
    should already be close to that limit."""
    remanence = 1.3
    b = on_axis_field_cylinder_magnet(1e-6, magnet_radius_m=0.006, magnet_length_m=2.0,
                                      remanence_t=remanence)
    assert b == pytest.approx(remanence / 2.0, rel=1e-3)


def test_longer_magnet_gives_more_on_axis_field_near_the_face_holding_radius_fixed():
    lengths = (0.005, 0.02, 0.1)
    fields = [
        on_axis_field_cylinder_magnet(0.001, magnet_radius_m=0.006, magnet_length_m=length,
                                      remanence_t=1.3)
        for length in lengths
    ]
    assert fields == sorted(fields)
