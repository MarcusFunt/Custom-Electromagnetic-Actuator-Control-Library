import time

import pytest

from emac_sim.fem.backend import ForcePoint
from emac_sim.fem.convergence import (estimate_sweep_cost, mesh_convergence)
from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry

_COIL = CoilWindingGeometry(0.0, turns=400, coil_length_m=0.03, radial_thickness_m=0.02)
_SLUG = SlugGeometry(magnet_radius_m=0.006, magnet_length_m=0.02, remanence_t=1.2)

F_INF = -0.85


def _order2_factory(mesh_size_m):
    """A synthetic backend whose force converges to F_INF as O(mesh^2)."""
    class _B:
        def solve(self, coil, slug, offset_m, current_a):
            return ForcePoint(force_n=F_INF + 40.0 * mesh_size_m ** 2)
    return _B()


def _divergent_factory(mesh_size_m):
    """A backend whose force does NOT settle with mesh (the stress-tensor failure mode)."""
    class _B:
        def solve(self, coil, slug, offset_m, current_a):
            # swings by order-1 amounts between meshes
            return ForcePoint(force_n=-0.85 * (1.0 + 5.0 * mesh_size_m ** 0.5))
    return _B()


def test_convergent_backend_is_reported_converged_and_extrapolates():
    rep = mesh_convergence(_COIL, _SLUG, 0.02, 3.0,
                           [0.003, 0.0018, 0.001, 0.0005], backend_factory=_order2_factory, tol=0.02)
    assert rep.converged
    assert rep.richardson_estimate_n == pytest.approx(F_INF, rel=1e-3)
    # the finest force is close to F_INF; uncertainty is tiny
    assert rep.uncertainty < 0.02
    # recommended mesh is one of the tried meshes and no finer than needed
    assert rep.recommended_mesh_m in {0.003, 0.0018, 0.001, 0.0005}


def test_divergent_backend_is_reported_not_converged():
    rep = mesh_convergence(_COIL, _SLUG, 0.02, 3.0,
                           [0.003, 0.0015, 0.00075], backend_factory=_divergent_factory, tol=0.02)
    assert not rep.converged
    assert rep.uncertainty > 0.02


def test_recommended_mesh_is_coarsest_within_tolerance():
    # loose tol => even the coarsest mesh qualifies (fastest)
    rep = mesh_convergence(_COIL, _SLUG, 0.02, 3.0,
                           [0.003, 0.0015, 0.00075], backend_factory=_order2_factory, tol=0.10)
    assert rep.recommended_mesh_m == 0.003


def test_meshes_are_sorted_coarse_to_fine_regardless_of_input_order():
    rep = mesh_convergence(_COIL, _SLUG, 0.02, 3.0,
                           [0.0005, 0.003, 0.001], backend_factory=_order2_factory, tol=0.02)
    meshes = [p.mesh_size_m for p in rep.points]
    assert meshes == sorted(meshes, reverse=True)


def test_estimate_sweep_cost_projects_grid_size_and_time():
    def slow_factory(mesh_size_m):
        class _B:
            def solve(self, coil, slug, offset_m, current_a):
                time.sleep(0.002)
                return ForcePoint(force_n=-0.5)
        return _B()

    cost = estimate_sweep_cost(_COIL, _SLUG, 0.001, n_offsets=41, n_currents=11,
                               n_geometries=10, sample_solves=3, backend_factory=slow_factory)
    assert cost.n_solves == 41 * 11 * 10
    assert cost.seconds_per_solve == pytest.approx(0.002, abs=0.004)
    assert cost.total_seconds == pytest.approx(cost.seconds_per_solve * cost.n_solves)


def test_empty_mesh_list_raises():
    with pytest.raises(ValueError):
        mesh_convergence(_COIL, _SLUG, 0.02, 3.0, [], backend_factory=_order2_factory)
