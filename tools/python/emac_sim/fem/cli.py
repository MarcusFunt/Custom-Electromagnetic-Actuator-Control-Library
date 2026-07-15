"""`emac-femgen`: config in, per-coil FEM/analytic-sweep LUT files out.

Reads a `[sim] kind = "linear_stepper"` config -- the SAME config a normal `emac-sim` run
uses -- builds each coil's axisymmetric geometry from it (fem/from_config.py), sweeps a
backend over a position x current grid (fem/sweep.py), and writes one `.npz`
fem.lut.ForceLUT per coil plus a `manifest.json` mapping coil index -> file, position, and
a ready-to-paste `force_lut_path` TOML snippet. Point a coil's `force_lut_path` at the
written file (see docs/FEM_PIPELINE.md) to have that coil's simulated force come from the
swept table instead of the analytic lobe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import LinearSimulationConfig, load_config
from .backend import FEMBackend
from .from_config import geometry_from_config
from .geometry import default_sweep_ranges
from .reference_backend import AnalyticReferenceBackend
from .sweep import sweep_coil


def _make_backend(name: str) -> FEMBackend:
    if name == "reference":
        return AnalyticReferenceBackend()
    if name == "femm":
        from .femm_backend import FemmBackend  # imported lazily: optional dependency

        return FemmBackend()
    raise ValueError(f"unknown backend: {name!r}")


def generate_luts(config: LinearSimulationConfig, backend_name: str, outdir: Path,
                   n_offsets: int, n_currents: int, max_current_a: float | None,
                   coil_indices: list[int] | None = None,
                   report=lambda msg: None) -> dict:
    """Sweep every requested coil and write its LUT + a manifest under `outdir`. Returns
    the manifest dict (also written to `outdir/manifest.json`)."""
    slug_geometry, coil_geometries = geometry_from_config(config)
    backend = _make_backend(backend_name)
    indices = coil_indices if coil_indices is not None else list(range(len(coil_geometries)))

    outdir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"backend": backend_name, "coils": []}

    for idx in indices:
        coil_geometry = coil_geometries[idx]
        default_offsets, default_currents = default_sweep_ranges(
            coil_geometry, slug_geometry, n_offsets=n_offsets, n_currents=n_currents,
            max_current_a=max_current_a if max_current_a is not None else 6.0,
        )
        n_total = len(default_offsets) * len(default_currents)

        def on_point(i: int, j: int, n_done: int, n_total: int = n_total, idx: int = idx) -> None:
            if n_done % max(1, n_total // 10) == 0 or n_done == n_total:
                report(f"coil[{idx}]: {n_done}/{n_total} points solved")

        report(f"coil[{idx}] @ {coil_geometry.position_m:g} m: sweeping {n_total} points "
               f"with the {backend_name!r} backend")
        lut = sweep_coil(coil_geometry, slug_geometry, backend,
                          offsets_m=default_offsets, currents_a=default_currents,
                          on_point=on_point)

        lut_path = outdir / f"coil_{idx:02d}.npz"
        lut.save(lut_path)
        manifest["coils"].append({
            "index": idx,
            "position_m": coil_geometry.position_m,
            "path": str(lut_path),
        })
        report(f"coil[{idx}]: wrote {lut_path}")

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    report(f"wrote {manifest_path}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emac-femgen",
        description="Sweep an axisymmetric FEM (or analytic-reference) backend over each "
                     "coil in a linear-stepper config and write force LUTs for it.",
    )
    parser.add_argument("--config", required=True, help="linear_stepper TOML config path")
    parser.add_argument("--outdir", default="build/fem_lut", help="directory for .npz LUTs + manifest.json")
    parser.add_argument("--backend", choices=["reference", "femm"], default="reference",
                         help="'reference' (default): analytic Biot-Savart, no FEMM needed. "
                              "'femm': real axisymmetric FEM solve, needs FEMM installed.")
    parser.add_argument("--coil", type=int, action="append", dest="coils",
                         help="coil index to sweep (repeatable); default: every coil")
    parser.add_argument("--n-offsets", type=int, default=41, help="grid points along position")
    parser.add_argument("--n-currents", type=int, default=11, help="grid points along current")
    parser.add_argument("--max-current-a", type=float, default=None,
                         help="sweep currents over +/- this value (default: 6.0 A)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)
    if not isinstance(config, LinearSimulationConfig):
        print("emac-femgen only supports [sim] kind = \"linear_stepper\" configs", file=sys.stderr)
        return 1

    report = (lambda msg: None) if args.quiet else (lambda msg: print(msg))
    generate_luts(
        config, args.backend, Path(args.outdir), args.n_offsets, args.n_currents,
        args.max_current_a, coil_indices=args.coils, report=report,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
