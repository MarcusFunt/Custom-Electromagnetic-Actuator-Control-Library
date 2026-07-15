"""Physical parametrization: turns and coil dimensions into the electrical/magnetic
constants (`CoilStation`'s R, L, k_a, x_c) the simulator actually consumes, and slug/magnet
dimensions into slug mass and the coil-magnet coupling strength. This is what makes
"optimize over turns and coil dimensions" meaningful at all -- without a real relationship
between them, more turns would have no modeled downside, and the search would trivially
diverge to infinity. See docs/DESIGN_OPTIMIZER.md for the derivations and their honest
caveats (this is a deliberately simple estimate, not a magnetostatics solve).

Everything here is SI units throughout (meters, ohms, henries, tesla, kg).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import integrate
from scipy.special import ellipe, ellipk

from .linear_plant import CoilStation
from .plant import COPPER_TEMP_COEFF_PER_C

MU_0 = 4.0e-7 * math.pi                 # H/m, vacuum permeability
COPPER_RESISTIVITY_20C_OHM_M = 1.68e-8  # ohm*m at 20 C
COPPER_RESISTIVITY_OHM_M = COPPER_RESISTIVITY_20C_OHM_M   # back-compat alias, 20 C value
NDFEB_DENSITY_KG_M3 = 7500.0            # typical sintered NdFeB density
COPPER_DENSITY_KG_M3 = 8960.0           # kg/m^3
COPPER_SPECIFIC_HEAT_J_PER_KG_K = 385.0 # J/(kg*K)


def copper_resistivity_ohm_m(temperature_c: float = 20.0) -> float:
    """R(T) tracking: docs/DESIGN.md flags copper's ~0.39%/C resistivity rise as
    "mandatory, not optional" for a real build, since it shifts both the reluctance and
    (via coil resistance) the electrical time constant. Linear model about the 20 C
    reference point -- good over the modest range windings actually operate in; not a
    substitute for the full thermal model (self-heating over time, duty cycle) that
    docs/DESIGN_OPTIMIZER.md notes this project still doesn't have."""
    return COPPER_RESISTIVITY_20C_OHM_M * (1.0 + COPPER_TEMP_COEFF_PER_C * (temperature_c - 20.0))


@dataclass(frozen=True)
class CoilWinding:
    resistance_ohm: float
    inductance_h: float
    wire_diameter_m: float
    mean_radius_m: float
    thermal_mass_j_per_k: float


def wind_coil(turns: int, coil_length_m: float, radial_thickness_m: float,
              bore_radius_m: float, packing_factor: float = 0.8,
              temperature_c: float = 20.0) -> CoilWinding:
    """Given how many turns you wind and the winding's outer envelope (axial length x
    radial build, around a fixed bore), derive the resulting wire gauge, resistance, and
    inductance. `turns` and coil dimensions are NOT independent: more turns in the same
    envelope means thinner wire (less area each), which raises resistance faster than
    linearly (both wire length AND 1/area grow with turns) -- the classic turns-vs-copper-
    loss trade-off. `packing_factor` (~0.7-0.85 for real hand/machine-wound round wire,
    0.785 theoretical max for a square lattice of circles) accounts for the air gaps
    between round conductors. `temperature_c` feeds copper_resistivity_ohm_m() -- default
    20 C; pass a hotter value to see the coil warmed up.
    """
    if turns < 1:
        raise ValueError("turns must be >= 1")
    winding_area_m2 = coil_length_m * radial_thickness_m       # axial x radial envelope
    copper_area_m2 = winding_area_m2 * packing_factor
    wire_area_m2 = copper_area_m2 / turns
    wire_diameter_m = math.sqrt(4.0 * wire_area_m2 / math.pi)

    mean_radius_m = bore_radius_m + 0.5 * radial_thickness_m
    mean_turn_length_m = 2.0 * math.pi * mean_radius_m
    total_wire_length_m = turns * mean_turn_length_m

    resistivity = copper_resistivity_ohm_m(temperature_c)
    resistance_ohm = resistivity * total_wire_length_m / wire_area_m2
    inductance_h = _solenoid_inductance_h(mean_radius_m, coil_length_m,
                                          radial_thickness_m, turns)

    # Thermal mass from the copper itself, not a fabricated constant: total copper volume
    # is (mean turn circumference) x (copper cross-sectional area) -- turns cancels out
    # (total_wire_length_m = turns*mean_turn_length_m, wire_area_m2 = copper_area_m2/turns),
    # so it reduces to the winding's own copper_area_m2 swept once around the mean radius,
    # independent of how that area is divided into turns. This deliberately covers ONLY the
    # copper's own heat capacity, not the bobbin/potting/frame around it -- a real build's
    # thermal mass is at least this much, usually more; treat this as a lower bound, not a
    # calibrated value (unlike thermal_resistance_k_per_w below, which this project has no
    # geometry-derived basis for at all -- see linear_plant.CoilStation).
    copper_volume_m3 = mean_turn_length_m * copper_area_m2
    thermal_mass_j_per_k = copper_volume_m3 * COPPER_DENSITY_KG_M3 * COPPER_SPECIFIC_HEAT_J_PER_KG_K

    return CoilWinding(resistance_ohm=resistance_ohm, inductance_h=inductance_h,
                       wire_diameter_m=wire_diameter_m, mean_radius_m=mean_radius_m,
                       thermal_mass_j_per_k=thermal_mass_j_per_k)


# Wheeler's µH-per-inch constant expressed for SI (meters, henries): 1e-6 H/µH divided by
# 0.0254 m/inch. Every length in both Wheeler formulas below is a ratio of like lengths, so
# this single factor converts either formula from its native inch/µH form to meters/henries.
_WHEELER_SI = 1.0e-6 / 0.0254   # == 3.937e-5

# Above this radial-build-to-mean-radius ratio a winding is firmly "multi-layer" and the
# single-layer formula is used at zero weight; below it the two are blended linearly. 0.2 is
# where Wheeler's own single- and multi-layer fits cross over in accuracy against a direct
# Maxwell mutual-inductance sum (both within a few percent there); see
# tests/test_coil_design.py's blend-vs-numeric checks.
_MULTILAYER_RATIO_THRESHOLD = 0.2


def _wheeler_single_layer_h(mean_radius_m: float, coil_length_m: float, turns: int) -> float:
    """Wheeler's (1928) single-layer air-core solenoid formula, in henries. Unlike the plain
    long-solenoid formula (L = mu0*N^2*A/length, only accurate when length >> radius), this
    fits well across both long and short/fat coils, and reduces to the long-solenoid form
    within ~0.3% in the length >> radius limit (checked in tests). It assumes a single
    cylindrical current sheet -- no radial build -- so it OVER-estimates a thick multi-layer
    winding (the outer layers enclose less flux than the innermost); see
    _wheeler_multilayer_h and _solenoid_inductance_h."""
    return _WHEELER_SI * mean_radius_m ** 2 * turns ** 2 / (9.0 * mean_radius_m + 10.0 * coil_length_m)


def _wheeler_multilayer_h(mean_radius_m: float, coil_length_m: float,
                          radial_thickness_m: float, turns: int) -> float:
    """Wheeler's (1928) multi-layer air-core formula L = 0.8*a^2*N^2/(6a+9b+10c) (a = mean
    radius, b = axial length, c = radial build), in henries. The `+10c` term is what the
    single-layer formula lacks: it accounts for the outer layers linking progressively less
    flux, which is a first-order effect once the radial build c is a sizeable fraction of the
    mean radius a -- exactly the regime the design optimizer explores (radial_thickness_m up
    to 40mm on a ~10mm-radius coil). Agrees with a direct Maxwell mutual-inductance double
    sum to within a few percent there, where the single-layer formula over-estimates by
    30-75%."""
    return (0.8 * _WHEELER_SI * mean_radius_m ** 2 * turns ** 2
            / (6.0 * mean_radius_m + 9.0 * coil_length_m + 10.0 * radial_thickness_m))


def _solenoid_inductance_h(mean_radius_m: float, coil_length_m: float,
                           radial_thickness_m: float, turns: int) -> float:
    """Air-core winding self-inductance (H), blending Wheeler's single-layer and multi-layer
    formulas by the radial-build ratio c/a. Neither formula is uniformly best: the
    single-layer form is the more accurate of the two for a genuinely thin winding (c/a -> 0,
    where the multi-layer fit's constants are slightly off), while the multi-layer form is
    far better once the winding has real radial depth (the single-layer form ignores c
    entirely and over-estimates by 30-75% for the thick coils the optimizer explores). Blend
    linearly in c/a up to _MULTILAYER_RATIO_THRESHOLD, then use the multi-layer form alone.
    Validated against a direct Maxwell mutual-inductance sum across thin/thick and short/long
    geometries (max ~4% error) in tests/test_coil_design.py."""
    single = _wheeler_single_layer_h(mean_radius_m, coil_length_m, turns)
    multi = _wheeler_multilayer_h(mean_radius_m, coil_length_m, radial_thickness_m, turns)
    w = min(1.0, (radial_thickness_m / mean_radius_m) / _MULTILAYER_RATIO_THRESHOLD)
    return (1.0 - w) * single + w * multi


def magnet_mass_kg(magnet_radius_m: float, magnet_length_m: float,
                    density_kg_m3: float = NDFEB_DENSITY_KG_M3) -> float:
    volume_m3 = math.pi * magnet_radius_m * magnet_radius_m * magnet_length_m
    return volume_m3 * density_kg_m3


def on_axis_field_cylinder_magnet(z_m: float, magnet_radius_m: float,
                                   magnet_length_m: float, remanence_t: float) -> float:
    """On-axis B field (tesla) of a uniformly axially-magnetized cylinder, at distance
    z_m from its near pole face, along its axis -- the standard closed-form result for
    this one case. Kept alongside off_axis_field_cylinder_magnet below (which reduces to
    this exact formula in the on-axis limit, checked in tests) as the cheap, closed-form
    special case when you only need the on-axis value."""
    def term(zz: float) -> float:
        return zz / math.sqrt(zz * zz + magnet_radius_m * magnet_radius_m)
    return 0.5 * remanence_t * (term(z_m + magnet_length_m) - term(z_m))


def _loop_field_axial(rho_m: float, z_m: float, loop_radius_m: float) -> float:
    """B_z per unit current (T/A) from a single circular current loop, at cylindrical
    point (rho, z) relative to the loop's center -- the standard Biot-Savart result via
    complete elliptic integrals (e.g. Jackson, Classical Electrodynamics). scipy's
    ellipk/ellipe take the PARAMETER m = k^2, not the modulus k. On-axis (rho=0) reduces
    to the elementary mu0*I*a^2 / (2*(a^2+z^2)^1.5) -- handled as a separate branch since
    the general formula has a removable singularity there."""
    a = loop_radius_m
    if rho_m < 1e-9:
        return MU_0 * a * a / (2.0 * (a * a + z_m * z_m) ** 1.5)
    m = 4.0 * a * rho_m / ((a + rho_m) ** 2 + z_m * z_m)
    m = min(m, 1.0 - 1e-12)
    k_ellip, e_ellip = ellipk(m), ellipe(m)
    denom = math.sqrt((a + rho_m) ** 2 + z_m * z_m)
    cross_term = (a * a - rho_m * rho_m - z_m * z_m) / ((a - rho_m) ** 2 + z_m * z_m)
    return (MU_0 / (2.0 * math.pi)) * (1.0 / denom) * (k_ellip + e_ellip * cross_term)


def off_axis_field_cylinder_magnet(rho_m: float, z_m: float, magnet_radius_m: float,
                                    magnet_length_m: float, remanence_t: float) -> float:
    """B_z (tesla) of a uniformly axially-magnetized cylinder at an arbitrary cylindrical
    point (rho, z), rho measured OFF the magnet's own axis -- z_m in the same "distance
    before the near pole face" convention as on_axis_field_cylinder_magnet (z_m can be
    negative, meaning a point axially INSIDE the magnet's own length; the integral is
    valid there too, e.g. for evaluating the field at the magnet's own center).

    Models the magnet via the standard equivalent-surface-current picture (a uniformly
    magnetized cylinder of remanence Br is magnetically equivalent to a solenoid of
    surface current density Br/mu0), integrating the single-loop Biot-Savart result
    above over the magnet's length. Exact in the rho->0 limit (checked against that closed
    form in tests). NOTE: B_z is NOT the field component estimate_k_a needs -- see
    _loop_field_radial / off_axis_radial_field_cylinder_magnet below for that. B_z is
    kept here because it's independently useful (e.g. for reasoning about reluctance-
    branch coupling, which DOES care about the axial field) and as a cross-check anchor.
    """
    surface_current_per_length = remanence_t / MU_0

    def integrand(z_prime: float) -> float:
        return _loop_field_axial(rho_m, z_m + z_prime, magnet_radius_m)

    value, _ = integrate.quad(integrand, 0.0, magnet_length_m)
    return surface_current_per_length * value


def _loop_field_radial(rho_m: float, z_m: float, loop_radius_m: float) -> float:
    """B_rho per unit current (T/A) from a single circular current loop, at cylindrical
    point (rho, z) relative to the loop's center -- the standard Biot-Savart result via
    complete elliptic integrals. Zero exactly on-axis (rho=0) or at the loop's own plane
    (z=0), by symmetry. THIS is the field component that matters for a voice-coil-style PM
    actuator: force on an AZIMUTHAL (coil) current comes from I*dL x B, and an azimuthal
    dL crossed with a RADIAL B gives an AXIAL force -- B_z (the axial field, what
    off_axis_field_cylinder_magnet above computes) does not enter that cross product at
    all. Using B_z here (as an earlier version of this module did) is not just less
    accurate than using B_rho, it's the wrong physical quantity."""
    a = loop_radius_m
    if rho_m < 1e-9 or abs(z_m) < 1e-12:
        return 0.0
    m = 4.0 * a * rho_m / ((a + rho_m) ** 2 + z_m * z_m)
    m = min(m, 1.0 - 1e-12)
    k_ellip, e_ellip = ellipk(m), ellipe(m)
    denom = rho_m * math.sqrt((a + rho_m) ** 2 + z_m * z_m)
    cross_term = (a * a + rho_m * rho_m + z_m * z_m) / ((a - rho_m) ** 2 + z_m * z_m)
    return (MU_0 / (2.0 * math.pi)) * (z_m / denom) * (-k_ellip + e_ellip * cross_term)


def off_axis_radial_field_cylinder_magnet(rho_m: float, z_m: float, magnet_radius_m: float,
                                           magnet_length_m: float, remanence_t: float) -> float:
    """B_rho (tesla) of a uniformly magnetized cylinder at (rho, z) -- z_m measured from
    the magnet's own AXIAL CENTER (a different, more natural convention than the B_z
    functions' "near face" one): by the magnet's symmetry, B_rho is ODD about the center
    -- exactly zero at z_m=0, growing to an interior maximum somewhere between the center
    and a pole face, then decaying to zero far away. Structurally the same shape as
    q_shape (odd, zero at center, peaked lobes) -- see estimate_k_a, which finds that
    peak directly and uses its location as a physically-derived coupling half-width."""
    half_length_m = magnet_length_m / 2.0
    surface_current_per_length = remanence_t / MU_0

    def integrand(z_prime: float) -> float:
        return _loop_field_radial(rho_m, z_m - z_prime, magnet_radius_m)

    value, _ = integrate.quad(integrand, -half_length_m, half_length_m)
    return surface_current_per_length * value


def _peak_radial_coupling(mean_radius_m: float, magnet_radius_m: float, magnet_length_m: float,
                           remanence_t: float, n_scan: int = 40) -> tuple[float, float]:
    """(offset, B_rho at that offset) where |B_rho| peaks as the coil's axial position
    varies relative to the magnet's center. Found by a coarse grid scan rather than a
    gradient method -- simple and robust, consistent with this model's overall level of
    rigor, and cheap enough (a few dozen field evaluations, done once per coil at design-
    build time, not per simulation timestep) not to matter for the optimizer's runtime."""
    half_length_m = magnet_length_m / 2.0
    search_span_m = 1.5 * half_length_m + magnet_radius_m   # comfortably past the pole face
    best_z, best_b = 0.0, 0.0
    for k in range(1, n_scan + 1):
        z = k * search_span_m / n_scan
        b = off_axis_radial_field_cylinder_magnet(mean_radius_m, z, magnet_radius_m,
                                                   magnet_length_m, remanence_t)
        if abs(b) > abs(best_b):
            best_z, best_b = z, b
    return best_z, best_b


def estimate_k_a(turns: int, mean_radius_m: float, magnet_radius_m: float,
                  magnet_length_m: float, remanence_t: float) -> float:
    """PM-branch thrust constant (N/A): F = k_a * i, the standard motor-design relation
    F = B_gap * L_wire * i, using B_rho (the radial field component -- see
    _loop_field_radial for why that's the physically correct one, not B_z) at wherever it
    peaks for this coil's actual mean radius. Also replaces the earlier on-axis-at-a-
    small-clearance approximation, which had no way to represent a radially-thick winding
    coupling more weakly than a thin one sitting right at the bore -- a real effect once
    radial_thickness_m is a sizeable fraction of magnet_radius_m."""
    _, b_peak = _peak_radial_coupling(mean_radius_m, magnet_radius_m, magnet_length_m, remanence_t)
    wire_length_m = turns * 2.0 * math.pi * mean_radius_m
    return abs(b_peak) * wire_length_m


def build_coil_station(position_m: float, turns: int, coil_length_m: float,
                        radial_thickness_m: float, magnet_radius_m: float,
                        magnet_length_m: float, remanence_t: float,
                        bore_clearance_m: float = 0.0015,
                        packing_factor: float = 0.8,
                        temperature_c: float = 20.0) -> CoilStation:
    """Assemble a full CoilStation from raw physical design knobs. bore_radius is derived
    as magnet_radius + a fixed clearance (not independently free) -- the coil bore has to
    accommodate the slug, so its size isn't really a separate design decision. Both k_a
    and x_c (coupling half-width) now come from the SAME B_rho peak search -- x_c is no
    longer an independent heuristic, it's just where that peak actually occurs."""
    bore_radius_m = magnet_radius_m + bore_clearance_m
    winding = wind_coil(turns, coil_length_m, radial_thickness_m, bore_radius_m,
                        packing_factor=packing_factor, temperature_c=temperature_c)
    peak_offset_m, b_peak = _peak_radial_coupling(winding.mean_radius_m, magnet_radius_m,
                                                   magnet_length_m, remanence_t)
    wire_length_m = turns * 2.0 * math.pi * winding.mean_radius_m
    k_a = abs(b_peak) * wire_length_m
    x_c = peak_offset_m
    return CoilStation(position_m=position_m, x_c=x_c, Cmag=0.0, k_a=k_a,
                       resistance_ohm=winding.resistance_ohm,
                       inductance_h=winding.inductance_h,
                       thermal_mass_j_per_k=winding.thermal_mass_j_per_k)
