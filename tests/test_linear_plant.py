import pytest

from emac_sim.plant import f_current, f_current_pm, q_shape
from emac_sim.linear_plant import (
    CoilStation,
    GateStation,
    LinearActuatorParams,
    coil_current_step,
    coil_force_gradient,
    coil_resistance,
    coil_temperature_step,
    default_coil_stations,
    default_gate_stations,
    kinetic_energy,
    net_force,
    step,
)


def default_params(**overrides) -> LinearActuatorParams:
    return LinearActuatorParams(**overrides)


def test_default_layout_has_five_coils_and_five_gates():
    p = default_params()
    assert len(p.coils) == 5
    assert len(p.gates) == 5
    # entry gate before coil 0, then one gate between each adjacent coil pair
    assert p.gates[0].position_m < p.coils[0].position_m
    for k in range(1, 5):
        assert p.coils[k - 1].position_m < p.gates[k].position_m < p.coils[k].position_m


def test_net_force_is_zero_far_from_every_coil():
    p = default_params()
    currents = [1.0] * len(p.coils)
    assert net_force(-10.0, currents, p) == pytest.approx(0.0, abs=1e-6)


def test_net_force_sums_two_simultaneously_energized_coils():
    p = default_params(coils=(
        CoilStation(position_m=0.0),
        CoilStation(position_m=0.05),
    ))
    x = 0.02
    solo_0 = net_force(x, [1.0, 0.0], p)
    solo_1 = net_force(x, [0.0, 1.0], p)
    both = net_force(x, [1.0, 1.0], p)
    assert both == pytest.approx(solo_0 + solo_1)


def test_net_force_is_zero_with_no_current():
    p = default_params()
    assert net_force(0.01, [0.0] * len(p.coils), p) == pytest.approx(0.0)


def test_free_slide_holds_velocity_with_zero_damping_and_current():
    p = default_params(damping_n_per_mps=0.0)
    x, v = 1.234, 0.5
    for _ in range(1000):
        x, v = step(x, v, [0.0] * len(p.coils), 1e-3, p)
    assert v == pytest.approx(0.5)


def test_damping_alone_monotonically_decelerates():
    p = default_params(damping_n_per_mps=0.2)
    x, v = 0.0, 1.0
    speeds = [v]
    for _ in range(500):
        x, v = step(x, v, [0.0] * len(p.coils), 1e-3, p)
        speeds.append(v)
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[-1] < speeds[0]


def test_kinetic_energy():
    p = default_params()
    assert kinetic_energy(0.0, p) == 0.0
    assert kinetic_energy(2.0, p) == pytest.approx(0.5 * p.mass_kg * 4.0)


def test_zero_pressure_bias_reproduces_the_unpressurized_model():
    p = default_params()
    assert p.pressure_bias_n == 0.0
    assert net_force(p.coils[0].position_m, [0.0] * len(p.coils), p) == pytest.approx(0.0)


def test_pressure_bias_moves_a_slug_resting_exactly_at_a_coil_center():
    """The one genuinely fragile case in the coil-only model: q_shape(0, x_c) == 0, so a
    slug sitting EXACTLY at a coil's center feels zero force from that coil at ANY current
    -- firing it does nothing. A pressure bias makes the net force nonzero there regardless,
    so the slug can never be stuck in true static equilibrium (docs/DESIGN_LINEAR.md)."""
    p = default_params(pressure_bias_n=0.5)
    x0 = p.coils[0].position_m
    # even with the coil that would normally rescue it FULLY energized, force from it is
    # zero at the exact center -- only the pressure bias moves the slug off this point.
    assert net_force(x0, [8.0] + [0.0] * (len(p.coils) - 1), p) == pytest.approx(0.5)

    x, v = step(x0, 0.0, [0.0] * len(p.coils), 1e-3, p)
    assert v > 0.0
    assert x > x0


def test_zero_pressure_bias_leaves_a_slug_at_a_coil_center_stuck():
    """Contrast case: without the bias, a slug exactly at rest at a coil's center truly
    does not move even with that coil fully energized -- this is the fragility the
    pressurized-tube option exists to remove."""
    p = default_params(pressure_bias_n=0.0)
    x0 = p.coils[0].position_m

    x, v = step(x0, 0.0, [8.0] + [0.0] * (len(p.coils) - 1), 1e-3, p)
    assert v == pytest.approx(0.0)
    assert x == pytest.approx(x0)


def test_k_a_zero_reproduces_the_pure_reluctance_model():
    """k_a=0 (no PM contribution) must match plant.f_current alone -- the hybrid model is
    additive, not a replacement, so turning the PM term off must be a true no-op. Cmag is
    given explicitly (nonzero) here -- CoilStation's default is 0.0 (pure-PM slug, no
    iron), so a hybrid/reluctance scenario has to opt in rather than rely on defaults."""
    coil_no_pm = CoilStation(position_m=0.0, Cmag=0.40, k_a=0.0)
    p = default_params(coils=(coil_no_pm,), gates=(GateStation(position_m=-0.025),))

    for i in [0.5, 2.0, 6.0]:
        expected = q_shape(0.01, coil_no_pm.x_c) * f_current(i, coil_no_pm)
        assert net_force(0.01, [i], p) == pytest.approx(expected)
        assert expected != 0.0   # guard against this silently degrading to a 0==0 check


def test_pm_term_is_additive_with_the_reluctance_term():
    coil = CoilStation(position_m=0.0, Cmag=0.40, k_a=0.20)
    coil_no_pm = CoilStation(position_m=0.0, Cmag=0.40, k_a=0.0)
    p_hybrid = default_params(coils=(coil,), gates=(GateStation(position_m=-0.025),))
    p_reluctance_only = default_params(coils=(coil_no_pm,), gates=(GateStation(position_m=-0.025),))

    # negative side of the coil (x < x_coil) is the "approach"/pump direction, where
    # q_shape is positive -- both terms push forward there, so the PM addition should
    # make the net force MORE forward, not less (see q_shape's docstring in plant.py).
    x, i = -0.01, 4.0
    hybrid_force = net_force(x, [i], p_hybrid)
    reluctance_only_force = net_force(x, [i], p_reluctance_only)
    pm_only_contribution = q_shape(x, coil.x_c) * f_current_pm(i, coil.k_a)
    assert reluctance_only_force != 0.0   # guard against this silently degrading to a 0-baseline check

    assert hybrid_force > reluctance_only_force   # the PM term adds, on top of reluctance
    assert hybrid_force == pytest.approx(reluctance_only_force + pm_only_contribution)


def test_pm_term_is_linear_and_signed_unlike_the_attract_only_reluctance_term():
    """The point of embedding a PM: force from that branch is now signed (repel with
    negative current), where the pure reluctance branch is always >= 0 regardless of the
    sign of i (attract-only). This is the physics-layer capability docs/DESIGN_LINEAR.md
    flags as not yet exploited by the supervisor's (still unipolar) current shaping."""
    coil = CoilStation(position_m=0.0, k_a=0.20, i_sat=6.0, Cmag=0.40)

    # reluctance branch alone: zero for i <= 0
    assert f_current(-3.0, coil) == 0.0
    assert f_current(3.0, coil) > 0.0

    # PM branch: linear and signed, proportional to current in both directions
    assert f_current_pm(-3.0, coil.k_a) == pytest.approx(-0.6)
    assert f_current_pm(3.0, coil.k_a) == pytest.approx(0.6)
    assert f_current_pm(-3.0, coil.k_a) == pytest.approx(-f_current_pm(3.0, coil.k_a))

    # so net_force at a negative current is entirely (and only) the PM contribution
    p = default_params(coils=(coil,), gates=(GateStation(position_m=-0.025),))
    x = 0.01
    assert net_force(x, [-3.0], p) == pytest.approx(q_shape(x, coil.x_c) * f_current_pm(-3.0, coil.k_a))


def test_default_slug_has_no_iron_only_a_magnet():
    """Cmag=0.0 is now the default -- this build's slug has no iron, so the reluctance
    branch must contribute nothing regardless of current, leaving only the PM term."""
    p = default_params()
    coil = p.coils[0]
    assert coil.Cmag == 0.0
    for i in [0.5, 3.0, 8.0]:
        assert f_current(i, coil) == 0.0
        assert net_force(coil.position_m - 0.01, [i], p) == pytest.approx(
            q_shape(-0.01, coil.x_c) * f_current_pm(i, coil.k_a)
        )


def test_coil_current_step_reaches_target_exactly_and_holds_it():
    """The idealized current-mode PWM controller solves for the exact constant voltage
    that reaches i_target in precisely one step, when that voltage is within the rail --
    so, unlike a cruder bang-bang model (which chatters around the target rather than
    holding it), it lands on the target exactly and STAYS there. Uses dt >> tau so the
    required voltage (which -> R*i_target as dt/tau grows) is comfortably within the rail
    -- a short dt for a large target can legitimately be rail-limited instead (checked
    separately below), which is correct rail-limited behavior, not a failure to converge."""
    coil = CoilStation(position_m=0.0, resistance_ohm=1.2, inductance_h=0.004)   # tau = L/R = 3.33 ms
    bus_v, target, dt = 12.0, 3.0, 0.05   # dt >> tau

    i = coil_current_step(0.0, i_target=target, coil=coil, bus_voltage_v=bus_v, dt=dt)
    assert i == pytest.approx(target)

    # once at the target, it holds exactly -- no chatter, no drift
    for _ in range(50):
        i = coil_current_step(i, i_target=target, coil=coil, bus_voltage_v=bus_v, dt=dt)
        assert i == pytest.approx(target)

    # and a lower target (explicit down-tracking case) reliably decreases current
    above_target = coil_current_step(target + 1.0, i_target=0.0, coil=coil,
                                      bus_voltage_v=bus_v, dt=1e-4)
    assert above_target < target + 1.0


def test_coil_current_step_is_rail_limited_but_never_overshoots_when_dt_is_short():
    """When the exact tracking voltage would exceed what the rail can supply (a large
    target reached over a dt that's short relative to tau), the controller applies full
    rail voltage for that step instead -- under-shooting the target that tick, same as a
    real voltage-limited driver, but by construction NEVER overshooting past it (unlike
    the old bang-bang model, whose overshoot was the actual bug this replacement fixes)."""
    coil = CoilStation(position_m=0.0, resistance_ohm=1.2, inductance_h=0.004)
    bus_v, target, dt = 12.0, 3.0, 1e-4   # dt << tau=3.33ms: rail-limited on the first step

    i_after_one_step = coil_current_step(0.0, i_target=target, coil=coil, bus_voltage_v=bus_v, dt=dt)
    assert 0.0 < i_after_one_step < target   # confirms this step really was rail-limited

    i = 0.0
    for _ in range(50):
        i_next = coil_current_step(i, i_target=target, coil=coil, bus_voltage_v=bus_v, dt=dt)
        assert i <= i_next <= target   # monotonic approach, never overshoots past target
        i = i_next


def test_coil_current_step_cannot_drive_current_negative():
    """The unipolar (no negative rail) driver model: even commanded to reach a negative
    target, current can only ever decay toward zero, never cross it."""
    coil = CoilStation(position_m=0.0, resistance_ohm=1.2, inductance_h=0.004)
    i = 2.0
    for _ in range(10000):
        i = coil_current_step(i, i_target=-5.0, coil=coil, bus_voltage_v=12.0, dt=1e-4)
        assert i >= 0.0
    assert i == pytest.approx(0.0, abs=1e-6)


def test_coil_current_step_resistance_override_changes_the_tracking_voltage():
    """A hotter coil (higher resistance) needs more voltage to reach the same target in
    the same dt -- confirms resistance_ohm_override actually participates in the solve,
    not just get ignored, and that the override, not coil.resistance_ohm, wins."""
    coil = CoilStation(position_m=0.0, resistance_ohm=1.2, inductance_h=0.004)
    bus_v, target, dt = 48.0, 3.0, 1e-4   # short dt: rail-limited, so R changes the outcome

    i_cold = coil_current_step(0.0, i_target=target, coil=coil, bus_voltage_v=bus_v, dt=dt)
    i_hot = coil_current_step(0.0, i_target=target, coil=coil, bus_voltage_v=bus_v, dt=dt,
                              resistance_ohm_override=2.4)
    assert i_hot < i_cold    # higher R -> slower current rise for the same applied voltage
    assert i_cold == pytest.approx(
        coil_current_step(0.0, i_target=target, coil=coil, bus_voltage_v=bus_v, dt=dt,
                          resistance_ohm_override=1.2))


def test_coil_force_gradient_equals_the_force_per_amp_coupling():
    """coil_force_gradient must be dF/di of the coil's actual force law -- the same coupling
    that both produces the force and (by reciprocity) the back-EMF. For a pure-PM coil this
    is q_shape(offset, x_c)*k_a, independent of current."""
    coil = CoilStation(position_m=0.0, x_c=0.02, Cmag=0.0, k_a=0.25)
    for offset in (-0.015, -0.005, 0.008, 0.02):
        expected = q_shape(offset, coil.x_c) * coil.k_a
        for current in (0.0, 2.0, -3.0):
            assert coil_force_gradient(coil, offset, current) == pytest.approx(expected)


def test_coil_force_gradient_matches_a_finite_difference_of_net_force():
    """The reciprocity claim, checked numerically: dF/di from coil_force_gradient must equal
    a central finite difference of the actual net_force, for both a pure-PM and a hybrid
    (reluctance + PM) coil where the reluctance branch makes the coupling current-dependent."""
    for coil in (CoilStation(position_m=0.0, x_c=0.02, Cmag=0.0, k_a=0.25),
                 CoilStation(position_m=0.0, x_c=0.02, Cmag=0.40, k_a=0.25, i_sat=6.0)):
        p = default_params(coils=(coil,), gates=(GateStation(position_m=-0.03),))
        di = 1e-6
        for offset in (-0.01, 0.006, 0.015):
            for current in (0.5, 3.0, 5.0):
                fd = (net_force(offset, [current + di], p)
                      - net_force(offset, [current - di], p)) / (2.0 * di)
                assert coil_force_gradient(coil, offset, current) == pytest.approx(fd, rel=1e-4)


def test_coil_force_gradient_finite_differences_a_force_lut():
    """When a coil carries a force_lut, the coupling comes from differencing the table, so a
    swept-table coil still gets a physically-consistent back-EMF."""
    # A simple linear-in-current, linear-in-offset table: F = 3.0 * offset * current.
    def lut(offset, current):
        return 3.0 * offset * current

    coil = CoilStation(position_m=0.0, force_lut=lut)
    for offset in (0.005, 0.01, 0.02):
        assert coil_force_gradient(coil, offset, 4.0) == pytest.approx(3.0 * offset, rel=1e-3)


def test_back_emf_zero_reproduces_the_no_emf_current_step():
    """The default back_emf_v=0.0 must be a bit-for-bit no-op relative to the prior model."""
    coil = CoilStation(position_m=0.0, resistance_ohm=1.2, inductance_h=0.004)
    for target in (1.0, 3.0, 6.0):
        with_zero = coil_current_step(0.0, target, coil, 12.0, 2e-4, back_emf_v=0.0)
        default = coil_current_step(0.0, target, coil, 12.0, 2e-4)
        assert with_zero == pytest.approx(default)


def test_back_emf_reduces_the_current_reached_in_one_step():
    """A positive (opposing) back-EMF means less net voltage drives the coil, so after one
    rail-limited step the current is strictly lower than with no EMF."""
    coil = CoilStation(position_m=0.0, resistance_ohm=1.2, inductance_h=0.004)
    no_emf = coil_current_step(0.0, 6.0, coil, 12.0, 2e-4, back_emf_v=0.0)
    with_emf = coil_current_step(0.0, 6.0, coil, 12.0, 2e-4, back_emf_v=6.0)
    assert 0.0 <= with_emf < no_emf


def test_back_emf_exceeding_the_bus_cannot_sustain_current_on_a_unipolar_driver():
    """When the motional back-EMF exceeds what the bus can supply, a single half-bridge
    can't hold the current up -- it decays toward zero (the freewheeling diode blocks
    reverse current), never going negative. This is the physical loading that was previously
    unmodeled: a fast slug's own back-EMF starving its driver."""
    coil = CoilStation(position_m=0.0, resistance_ohm=1.2, inductance_h=0.004)
    i = 4.0
    for _ in range(2000):
        i = coil_current_step(i, 6.0, coil, bus_voltage_v=12.0, dt=1e-4, back_emf_v=30.0)
        assert i >= 0.0
    assert i == pytest.approx(0.0, abs=1e-3)


def test_coil_resistance_matches_plant_formula_and_rises_with_temperature():
    coil = CoilStation(position_m=0.0, resistance_ohm=1.2)
    assert coil_resistance(coil, 20.0) == pytest.approx(1.2)
    assert coil_resistance(coil, 70.0) > coil_resistance(coil, 20.0)


def test_coil_temperature_step_converges_to_steady_state_under_sustained_current():
    """Exact for any dt (plant.thermal_step), so a single dt >> tau step lands at the
    steady state directly rather than needing a long small-dt loop."""
    coil = CoilStation(position_m=0.0, thermal_mass_j_per_k=10.0, thermal_resistance_k_per_w=5.0)
    ambient, current, resistance = 20.0, 3.0, 1.2
    tau = coil.thermal_mass_j_per_k * coil.thermal_resistance_k_per_w
    t = coil_temperature_step(ambient, current, resistance, coil, ambient, dt=50.0 * tau)
    expected_steady_state = ambient + (current * current * resistance) * coil.thermal_resistance_k_per_w
    assert t == pytest.approx(expected_steady_state, rel=1e-9)
    assert t > ambient    # it actually heated up


def test_coil_temperature_step_with_zero_current_stays_at_ambient():
    coil = CoilStation(position_m=0.0, thermal_mass_j_per_k=10.0, thermal_resistance_k_per_w=5.0)
    t = 20.0
    for _ in range(1000):
        t = coil_temperature_step(t, 0.0, 1.2, coil, ambient_c=20.0, dt=1e-3)
    assert t == pytest.approx(20.0)


def test_rl_current_lags_an_ideal_instantaneous_target():
    """The whole point of adding inductance: a step target isn't reached instantly -- the
    actual current after one short tick must be strictly less than the ideal target."""
    coil = CoilStation(position_m=0.0, resistance_ohm=1.2, inductance_h=0.004)
    i_after_one_tick = coil_current_step(0.0, i_target=6.0, coil=coil, bus_voltage_v=12.0, dt=2e-4)
    assert 0.0 < i_after_one_tick < 6.0


def test_bipolar_coil_current_step_can_reach_a_negative_target():
    """Contrast with the unipolar case above: an H-bridge CAN drive current down through
    and past zero to a negative (repel) target -- this is what makes repel-pumping
    actually achievable under current_loop="rl" (docs/DESIGN_LINEAR.md). Over many steps
    (many tau) it converges to -- and holds -- that target exactly, same as the positive-
    target case above."""
    coil = CoilStation(position_m=0.0, resistance_ohm=1.2, inductance_h=0.004)
    i = 2.0
    for _ in range(3000):
        i = coil_current_step(i, i_target=-3.0, coil=coil, bus_voltage_v=12.0, dt=1e-4, bipolar=True)
    assert i == pytest.approx(-3.0)


def test_repel_on_departure_matches_attract_on_approach_in_magnitude():
    """The core physics claim behind repel-pumping: at equal distance from a coil's
    center, attracting on the approach side (x < x_coil, i > 0) and repelling on the
    departure side (x > x_coil, i < 0) produce exactly the same forward force -- q_shape
    flips sign, f_current_pm flips sign, and the product doesn't."""
    coil = CoilStation(position_m=0.0, x_c=0.02, Cmag=0.0, k_a=0.25)
    p = default_params(coils=(coil,), gates=(GateStation(position_m=-0.03),))

    for d, i in [(0.005, 2.0), (0.01, 5.0), (0.015, 8.0)]:
        approach_attract = net_force(-d, [i], p)
        departure_repel = net_force(d, [-i], p)
        assert approach_attract == pytest.approx(departure_repel)
        assert approach_attract > 0.0   # both push forward (+x), not just equal each other

    # and the mismatched pairing (attract while departing) brakes instead -- confirms the
    # sign flip is doing real work here, not a coincidence of the assertions above
    departure_attract = net_force(0.01, [5.0], p)
    assert departure_attract < 0.0


def test_work_energy_theorem_for_a_constant_pm_current():
    """Integrating force over the actual traveled path should match the resulting change
    in kinetic energy -- the basic physical consistency check for net_force/step/
    kinetic_energy working together, independent of any supervisor/energy-shaping math."""
    coil = CoilStation(position_m=0.0, x_c=0.02, Cmag=0.0, k_a=0.2)
    p = default_params(coils=(coil,), gates=(GateStation(position_m=-0.03),),
                       damping_n_per_mps=0.0)

    x, v = -0.015, 0.5
    i = 3.0
    dt = 1e-6
    e0 = kinetic_energy(v, p)
    work = 0.0
    while x < -0.001:
        work += net_force(x, [i], p) * v * dt
        x, v = step(x, v, [i], dt, p)
    e1 = kinetic_energy(v, p)

    assert (e1 - e0) == pytest.approx(work, rel=1e-3)
