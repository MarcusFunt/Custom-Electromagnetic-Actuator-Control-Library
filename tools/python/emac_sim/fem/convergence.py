"""Mesh-convergence and cost analysis for a FEM backend -- the pre-flight check before a
large FEM run.

The whole force-extraction bug was a convergence failure: the force didn't settle as the
mesh shrank. Before committing hours of solves to a fixed mesh, you want two numbers:

  1. Is the force actually mesh-CONVERGED at the mesh I plan to use, and what is the coarsest
     mesh I can get away with? (Coarser = far faster; too coarse = wrong.) -> mesh_convergence
  2. How long will the whole sweep take at that mesh? -> estimate_sweep_cost

Both are written against a `backend_factory(mesh_size_m) -> backend` so they are backend- and
FEMM-agnostic (and unit-testable with a synthetic backend whose error scales with the mesh).
`femm_backend_factory` is the ready-made factory for the real solver.

Timing note: this module uses `time.perf_counter`, so it is a normal script (not usable
inside a resume-safe workflow sandbox). A warm-up solve is always discarded before timing so
FEMM start-up / first-mesh cost doesn't pollute the per-solve estimate.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable, Sequence

from .backend import FEMBackend
from .geometry import CoilWindingGeometry, SlugGeometry, coupling_scale_m

BackendFactory = Callable[[float], FEMBackend]


def femm_backend_factory(mesh_size_m: float) -> FEMBackend:
    """Default factory: a real FEMM backend pinned to `mesh_size_m`. Imported lazily so this
    module stays import-safe without FEMM."""
    from .femm_backend import FemmBackend
    return FemmBackend(mesh_size_m=mesh_size_m)


def _solve_with(backend: FEMBackend, coil, slug, offset_m, current_a):
    """Use `backend` as a context manager if it is one (FEMM opens/closes), else directly."""
    ctx = backend if hasattr(backend, "__enter__") else nullcontext(backend)
    with ctx as b:
        return b.solve(coil, slug, offset_m, current_a).force_n


@dataclass(frozen=True)
class ConvergencePoint:
    mesh_size_m: float
    force_n: float
    solve_time_s: float


@dataclass(frozen=True)
class ConvergenceReport:
    points: list[ConvergencePoint]        # sorted coarse -> fine (descending mesh size)
    tol: float
    offset_m: float
    current_a: float

    @property
    def finest(self) -> ConvergencePoint:
        return self.points[-1]

    @property
    def uncertainty(self) -> float:
        """Relative change between the two finest meshes -- the residual uncertainty in the
        finest force (1.0 if only one mesh was tried)."""
        if len(self.points) < 2:
            return 1.0
        a, b = self.points[-2].force_n, self.points[-1].force_n
        return abs(a - b) / max(abs(b), 1e-30)

    @property
    def converged(self) -> bool:
        return self.uncertainty <= self.tol

    @property
    def recommended_mesh_m(self) -> float:
        """The COARSEST (fastest) mesh whose force is within `tol` of the finest -- the mesh
        to actually run the sweep at. Falls back to the finest if none qualify."""
        best = self.finest.force_n
        for p in self.points:            # coarse -> fine; take the first that is close enough
            if abs(p.force_n - best) / max(abs(best), 1e-30) <= self.tol:
                return p.mesh_size_m
        return self.finest.mesh_size_m

    @property
    def richardson_estimate_n(self) -> float:
        """Extrapolated mesh-independent force from the three finest points, assuming error ~
        C*h^p with a fitted order p (needs 3 geometrically-spaced meshes; else the finest)."""
        if len(self.points) < 3:
            return self.finest.force_n
        h = [p.mesh_size_m for p in self.points[-3:]]
        f = [p.force_n for p in self.points[-3:]]
        d1, d2 = f[1] - f[0], f[2] - f[1]
        if d2 == 0.0 or d1 == 0.0 or (d1 / d2) <= 0.0:
            return self.finest.force_n
        r = h[0] / h[1]                  # mesh ratio (coarse/fine), > 1
        import math
        try:
            p = math.log(d1 / d2) / math.log(r)
        except ValueError:
            return self.finest.force_n
        if not (0.2 < p < 6.0):          # implausible order -> don't trust the extrapolation
            return self.finest.force_n
        return f[2] + d2 / (r ** p - 1.0)

    def __str__(self) -> str:
        lines = [
            f"mesh convergence at offset {self.offset_m*1e3:.2f} mm, I={self.current_a:g} A "
            f"(tol {self.tol*100:.0f}%):",
            f"{'mesh_mm':>10} {'force_N':>12} {'solve_s':>9}",
        ]
        for p in self.points:
            lines.append(f"{p.mesh_size_m*1e3:10.3f} {p.force_n:12.5f} {p.solve_time_s:9.2f}")
        verdict = "CONVERGED" if self.converged else "NOT CONVERGED"
        lines.append(f"-> {verdict} (uncertainty {self.uncertainty*100:.1f}%); "
                     f"recommended mesh {self.recommended_mesh_m*1e3:.3f} mm; "
                     f"extrapolated force {self.richardson_estimate_n:.5f} N")
        return "\n".join(lines)


def mesh_convergence(coil: CoilWindingGeometry, slug: SlugGeometry,
                     offset_m: float, current_a: float, mesh_sizes_m: Sequence[float],
                     backend_factory: BackendFactory = femm_backend_factory,
                     tol: float = 0.02) -> ConvergenceReport:
    """Solve one (coil, slug, offset, current) at each mesh size and report whether the force
    has converged. `mesh_sizes_m` is sorted internally coarse->fine. A fresh backend is built
    per mesh via `backend_factory`."""
    meshes = sorted(set(float(m) for m in mesh_sizes_m), reverse=True)
    if not meshes:
        raise ValueError("mesh_sizes_m must be non-empty")
    points: list[ConvergencePoint] = []
    for m in meshes:
        backend = backend_factory(m)
        t0 = time.perf_counter()
        force = _solve_with(backend, coil, slug, offset_m, current_a)
        dt = time.perf_counter() - t0
        points.append(ConvergencePoint(mesh_size_m=m, force_n=float(force), solve_time_s=dt))
    return ConvergenceReport(points=points, tol=tol, offset_m=offset_m, current_a=current_a)


@dataclass(frozen=True)
class CostEstimate:
    seconds_per_solve: float
    n_solves: int
    total_seconds: float

    @property
    def total_hours(self) -> float:
        return self.total_seconds / 3600.0

    def __str__(self) -> str:
        return (f"~{self.seconds_per_solve:.2f} s/solve x {self.n_solves} solves "
                f"= {self.total_seconds/60.0:.1f} min ({self.total_hours:.2f} h)")


def estimate_sweep_cost(coil: CoilWindingGeometry, slug: SlugGeometry,
                        mesh_size_m: float, n_offsets: int, n_currents: int,
                        n_geometries: int = 1, sample_solves: int = 3,
                        offset_m: float | None = None, current_a: float = 3.0,
                        backend_factory: BackendFactory = femm_backend_factory) -> CostEstimate:
    """Project the wall-clock of a full sweep by timing a few real solves at a representative
    operating point (one warm-up solve is discarded first). `n_geometries` scales it up for a
    factorial study of many cells. Solve time depends mostly on mesh + part size, so a single
    representative point is a good proxy for the per-solve cost."""
    if offset_m is None:
        offset_m = 0.35 * coupling_scale_m(coil, slug)
    backend = backend_factory(mesh_size_m)
    ctx = backend if hasattr(backend, "__enter__") else nullcontext(backend)
    with ctx as b:
        b.solve(coil, slug, offset_m, current_a)              # warm-up, discarded
        t0 = time.perf_counter()
        for _ in range(max(1, sample_solves)):
            b.solve(coil, slug, offset_m, current_a)
        per = (time.perf_counter() - t0) / max(1, sample_solves)
    n = int(n_offsets) * int(n_currents) * int(n_geometries)
    return CostEstimate(seconds_per_solve=per, n_solves=n, total_seconds=per * n)


def _default_mesh_ladder(coil: CoilWindingGeometry, slug: SlugGeometry) -> list[float]:
    """A geometric ladder of mesh sizes bracketing the backend's own default
    (0.15*min(radial_thickness, magnet_radius)), from 1.7x coarser to ~3.7x finer."""
    base = 0.15 * min(coil.radial_thickness_m, slug.magnet_radius_m)
    return [base * f for f in (1.67, 1.0, 0.53, 0.27)]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from pathlib import Path

    from ..config import LinearSimulationConfig, load_config
    from .from_config import geometry_from_config

    parser = argparse.ArgumentParser(
        prog="emac-femcheck",
        description="Pre-flight a FEM run: check mesh convergence for a coil and estimate the "
                    "wall-clock of a full sweep. Requires FEMM.")
    parser.add_argument("--config", required=True, help="linear_stepper TOML config path")
    parser.add_argument("--coil", type=int, default=0, help="coil index to check (default 0)")
    parser.add_argument("--tol", type=float, default=0.02, help="convergence tolerance (default 2%%)")
    parser.add_argument("--n-offsets", type=int, default=41, help="sweep offset points (for cost)")
    parser.add_argument("--n-currents", type=int, default=11, help="sweep current points (for cost)")
    parser.add_argument("--n-geometries", type=int, default=1, help="geometry cells (for cost)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if not isinstance(config, LinearSimulationConfig):
        print("emac-femcheck only supports [sim] kind = \"linear_stepper\" configs", file=sys.stderr)
        return 1
    slug, coils = geometry_from_config(config)
    coil = coils[args.coil]
    offset = 0.35 * coupling_scale_m(coil, slug)

    print(f"coil[{args.coil}] @ {coil.position_m:g} m, slug r={slug.magnet_radius_m*1e3:.1f}mm")
    report = mesh_convergence(coil, slug, offset, 3.0, _default_mesh_ladder(coil, slug), tol=args.tol)
    print(report)
    rec = report.recommended_mesh_m
    cost = estimate_sweep_cost(coil, slug, rec, args.n_offsets, args.n_currents,
                               n_geometries=args.n_geometries)
    print(f"\nfull-sweep cost at the recommended {rec*1e3:.3f} mm mesh: {cost}")
    if not report.converged:
        print("WARNING: not converged at the finest mesh tried -- refine further before the run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
