"""Physics-accuracy invariants for the host engine -- the checks that make this a
reliable reference rather than a plausible-looking script. These are all fast, FEMM-free,
and deterministic. Real-FEMM cross-checks live in test_fem_femm_backend.py.

They pin four properties that were only informally claimed before:
  1. Both mechanical integrators are genuinely 2nd-order in dt (not just "better than Euler").
  2. The analytic plant's PM thrust constant k_a equals the winding-averaged FEM/analytic
     reference at the same geometry -- the consistency that a single shared kernel now
     guarantees (the two used to diverge 25-75%).
  3. coil_force_gradient is the true dF/di of the force law it advertises (Maxwell
     reciprocity), so the motional back-EMF e=(dF/di)*v is thermodynamically correct.
  4. For the default pure-PM slug, that back-EMF makes electrical and mechanical power
     match exactly -- the energy-conservation guarantee the "rl" loop rests on.
"""
import math

import pytest

from emac_sim import coil_design, linear_plant, plant
from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry
from emac_sim.fem.reference_backend import AnalyticReferenceBackend


def _self_convergence_order(run, h):
    """Reference-free grid-convergence order from three halving steps 4h, 2h, h. For a
    p-th-order method u(h)=u_exact+C*h^p, so ||u(4h)-u(2h)|| / ||u(2h)-u(h)|| = 2^p exactly,
    independent of the (unknown) exact solution. More robust than differencing against a
    fine reference when part of the trajectory is integrated exactly (velocity-Verlet is
    exact for force-free drift), which would otherwise make a Richardson estimate noisy."""
    u4, u2, u1 = run(4.0 * h), run(2.0 * h), run(h)
    d42 = math.hypot(*(a - b for a, b in zip(u4, u2)))
    d21 = math.hypot(*(a - b for a, b in zip(u2, u1)))
    return math.log2(d42 / d21)


def test_pendulum_step_is_second_order_in_dt():
    p = plant.PendulumParams(Q=1e30)

    def run(dt, t_end=1.5, theta0=1.0):
        theta, omega = theta0, 0.0
        for _ in range(int(round(t_end / dt))):
            theta, omega = plant.step(theta, omega, 0.0, dt, p)
        return theta, omega

    order = _self_convergence_order(run, 1e-4)
    assert order == pytest.approx(2.0, abs=0.2), f"pendulum integrator order {order:.2f} != 2"


def test_linear_step_is_second_order_in_dt():
    p = linear_plant.LinearActuatorParams(
        mass_kg=0.05, damping_n_per_mps=0.0,
        coils=(linear_plant.CoilStation(position_m=0.05, k_a=0.5, x_c=0.02),),
        gates=(), current_loop="ideal",
    )

    def run(dt, t_end=0.25):
        x, v = 0.0, 1.0
        for _ in range(int(round(t_end / dt))):
            x, v = linear_plant.step(x, v, [3.0], dt, p)
        return x, v

    order = _self_convergence_order(run, 1e-4)
    assert order == pytest.approx(2.0, abs=0.2), f"linear integrator order {order:.2f} != 2"


@pytest.mark.parametrize("turns,coil_len,radial,mag_r,mag_len,br", [
    (200, 0.010, 0.004, 0.005, 0.010, 1.3),
    (800, 0.050, 0.030, 0.006, 0.020, 1.2),
    (400, 0.030, 0.020, 0.006, 0.020, 1.2),
])
def test_analytic_k_a_matches_winding_averaged_reference(turns, coil_len, radial, mag_r, mag_len, br):
    """build_coil_station's k_a (the analytic plant's peak PM thrust per amp) must equal the
    peak of the FEM/analytic reference backend's force-per-amp curve for the same geometry.
    Both now call coil_design.winding_averaged_force_per_amp, so they agree to quadrature
    precision -- this guards against them ever silently diverging again (they once differed
    by up to 75% because one averaged over the winding and the other sampled a single point)."""
    coil = coil_design.build_coil_station(0.0, turns=turns, coil_length_m=coil_len,
                                          radial_thickness_m=radial, magnet_radius_m=mag_r,
                                          magnet_length_m=mag_len, remanence_t=br)
    cg = CoilWindingGeometry(0.0, turns, coil_len, radial)
    sg = SlugGeometry(mag_r, mag_len, br)
    ref = AnalyticReferenceBackend()
    scale = 1.5 * coil_len + 0.5 * mag_len
    ref_peak = max(abs(ref.solve(cg, sg, 0.02 * scale * k, 1.0).force_n) for k in range(1, 60))
    assert coil.k_a == pytest.approx(ref_peak, rel=0.01)


@pytest.mark.parametrize("Cmag,k_a", [(0.0, 0.3), (0.02, 0.3), (0.05, 0.0)])
def test_coil_force_gradient_is_the_true_dF_di(Cmag, k_a):
    """coil_force_gradient must equal d(net_force)/di of the coil's actual force law, for
    both the PM and reluctance branches -- this is what makes the motional back-EMF
    e=(dF/di)*v the correct Maxwell reciprocal of the force. Checked by central difference."""
    coil = linear_plant.CoilStation(position_m=0.0, x_c=0.02, Cmag=Cmag, k_a=k_a, i_sat=6.0)
    p = linear_plant.LinearActuatorParams(coils=(coil,), gates=())
    for offset in (0.005, 0.015, -0.01):
        for i in (0.5, 2.0, 4.0):
            di = 1e-6
            f_plus = linear_plant.net_force(offset, [i + di], p)
            f_minus = linear_plant.net_force(offset, [i - di], p)
            numeric = (f_plus - f_minus) / (2.0 * di)
            analytic = linear_plant.coil_force_gradient(coil, offset, i)
            assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_pure_pm_back_emf_makes_electrical_and_mechanical_power_match():
    """For the default pure-PM slug the force is F=(dF/di)*i (linear in current), so the
    back-EMF power e*i = (dF/di)*v*i equals the mechanical power F*v exactly, tick by tick.
    That identity is the energy-conservation guarantee behind the 'rl' loop: mechanical work
    is drawn from the electrical source, not conjured. A sign error or a k_a mismatch between
    net_force and coil_force_gradient would break it."""
    coil = linear_plant.CoilStation(position_m=0.0, x_c=0.02, Cmag=0.0, k_a=0.4)
    p = linear_plant.LinearActuatorParams(coils=(coil,), gates=())
    for offset in (0.004, 0.012, -0.02):
        for i in (0.7, 3.3):
            for v in (0.5, 5.0, -2.0):
                F = linear_plant.net_force(offset, [i], p)
                e = linear_plant.coil_force_gradient(coil, offset, i) * v
                assert e * i == pytest.approx(F * v, rel=1e-12, abs=1e-15)
