# FEM table-generation pipeline

`docs/PHYSICS_ENGINE_ANALYSIS.md` flags the linear stepper's magnetic coupling as "still
synthetic" -- `plant.q_shape()` is an analytic Gaussian-lobe placeholder, and
`coil_design.estimate_k_a()` reduces a coil's real field profile to a single constant taken
at its peak. This pipeline replaces that shape assumption with a real axisymmetric field
solve, sampled into a lookup table (LUT) that the plant interpolates at simulation time.

It lives in `tools/python/emac_sim/fem/` and is entirely opt-in: no existing config or test
changes behavior unless a coil sets `force_lut_path`.

## Pieces

| Module | Role |
|---|---|
| `fem/geometry.py` | `CoilWindingGeometry` / `SlugGeometry` -- the same physical knobs `coil_design.py` already uses (turns, coil dimensions, magnet dimensions, remanence). |
| `fem/backend.py` | `FEMBackend` protocol: `solve(coil, slug, offset_m, current_a) -> ForcePoint`. |
| `fem/reference_backend.py` | Analytic backend (closed-form Biot-Savart, reusing `coil_design.off_axis_radial_field_cylinder_magnet`) evaluated at the requested offset instead of only at its peak, and AVERAGED over the winding cross-section (a 7x3 Gauss-Legendre quadrature over the coil's axial length and radial build, turn-length weighted) rather than sampled at the single mean-radius/coil-center point. That single point sits at the field maximum, so sampling it there over-states the peak force per amp by ~60% for a coil whose length is comparable to the coupling scale; the average reduces exactly to the single point in the vanishing-winding limit. **Not a real FEM solve** -- no iron, no saturation, vacuum permeability everywhere -- but it traces the coil's actual (non-Gaussian) coupling shape and needs no external tools, so it's both the default backend and what this repo's own tests run against. |
| `fem/femm_backend.py` | Real backend: builds and solves an axisymmetric magnetostatic problem in [FEMM](http://www.femm.info/) via its optional `femm` Python module, and reads force as the **axial Lorentz force on the coil** (not the Maxwell stress tensor over the magnet, which does not converge here -- see `docs/VALIDATION.md`). Raises `FemmNotAvailableError` with a clear message if FEMM isn't installed -- everything else in this package works without it. |
| `fem/validate.py` | Sweeps the analytic and FEMM backends over the same grid and reports their relative disagreement (`compare_analytic_to_femm`) -- quantify the analytic model's accuracy for your geometry before trusting it. See `docs/VALIDATION.md`. |
| `fem/sweep.py` | Sweeps a backend over a (offset, current) grid into a `ForceLUT`. |
| `fem/lut.py` | `ForceLUT`: `.npz` save/load + edge-clamped bilinear interpolation, callable as `(offset_m, current_a) -> force_n`. Clamps rather than extrapolates -- a sweep is only trustworthy inside the region it actually sampled. |
| `fem/from_config.py` | Builds `CoilWindingGeometry`/`SlugGeometry` straight from a `LinearSimulationConfig`. |
| `fem/cli.py` | `emac-femgen`: the command-line entry point below. |

## Generating tables

Add a `[slug]` section and per-coil geometry fields (`turns`, `coil_winding_length_m`,
`radial_thickness_m`, `bore_clearance_m`, `packing_factor`, `winding_temperature_c`) to a
`linear_stepper` config -- see `examples/configs/linear_stepper_5coil_fem.toml`. Then:

```powershell
emac-femgen --config examples/configs/linear_stepper_5coil_fem.toml --outdir build/fem_lut
```

Options: `--backend {reference,femm}` (default `reference`), `--coil N` (repeatable, default
every coil), `--n-offsets`/`--n-currents` (grid resolution), `--max-current-a` (sweep span),
`--quiet`. Writes one `coil_NN.npz` per coil plus a `manifest.json`.

A real FEMM sweep solves one full mesh per grid point -- seconds, not microseconds -- which
is exactly why this is a table-generation step, never called per simulation timestep.

## Using a table

Point a coil's `force_lut_path` at the written `.npz` (relative to the current working
directory):

```toml
[[coils]]
position_m = 0.00
force_lut_path = "build/fem_lut/coil_00.npz"
```

`config.LinearSimulationConfig.to_actuator_params()` loads it into
`linear_plant.CoilStation.force_lut`; `linear_plant.net_force()` then calls the LUT directly
for that coil instead of `q_shape`/`f_current`/`f_current_pm` -- the table already **is**
the coil's full force law, so there's no remaining role for the synthetic lobe once one
exists. A coil that never sets `force_lut_path` is completely unaffected (`force_lut`
defaults to `None`).

## Installing FEMM

FEMM is a separate Windows application, not a pip package: install it from
<http://www.femm.info/>, which places `femm.py` on your Python path (or `pip install
pyfemm` if you're using the PyPI wheel). Without it, `--backend femm` raises
`FemmNotAvailableError` with this same pointer; `--backend reference` needs nothing beyond
this repo's existing dependencies.

## Sweep range: why it's wider than it looks, and non-uniformly spaced

`fem/geometry.py`'s `default_sweep_ranges` doesn't use a plain uniform grid. Two properties
matter for the result to be trustworthy for actual analysis (not just plumbing that runs):

- **The grid must extend far enough that `ForceLUT`'s edge-clamping is physically correct.**
  A coil-magnet coupling decays roughly like 1/r^3 away from the coil; if the swept range
  stops too early, every out-of-range query (e.g. a coil the slug has already passed) gets
  clamped to a non-negligible constant instead of the true near-zero value -- a persistent
  "phantom force" that doesn't show up as a crash or a NaN, only as quietly wrong physics.
  `FAR_SPAN_FACTOR` was chosen (and is regression-tested, see
  `tests/test_fem_analysis_regression.py`) so the edge value is <0.1% of peak.
- **The grid must stay dense enough near the coupling peak** (which sits partway into the
  falloff, not at the coil's own center -- offset=0 is actually a ZERO of the coupling) that
  linear interpolation between grid points stays accurate where curvature is highest. A
  single geometrically-spaced grid concentrated around offset=0 was tried first and
  rejected for exactly this reason. `_two_region_grid` instead spends most of its point
  budget on a uniform fine grid covering the peak and initial falloff, and only a sparse
  tail beyond that. The default resolution is `--n-offsets 41` and the fine region spans
  `FINE_SPAN_FACTOR = 1.0` coupling scales: the winding-averaged reference force rises more
  steeply out of the offset=0 null than the earlier single-point estimate, so the fine
  budget is both larger and concentrated where that curvature is (worst-case linear-
  interpolation error ~5% at the default geometry, vs ~16% with the old 31-point/1.5x span).

If you widen a coil's geometry drastically or need tighter accuracy, prefer raising
`--n-offsets` over hand-tuning the span constants -- the two-region allocation already
targets the region that matters.

## Using it in the design optimizer and sensitivity sweeps

`emac-optimize` and `design_sensitivity.py`'s sweeps -- the actual design-space *analysis*
tools in this repo -- always built coils via `coil_design.build_coil_station` (the analytic
k_a/x_c estimate through `plant.q_shape`'s synthetic lobe), never touching the FEM pipeline.
That's fixed: `optimize_design.build_params`/`simulate_design`/`optimize`, and every
`design_sensitivity.py` sweep function, now take a `force_law` argument:

- `"analytic"` (default, unchanged behavior) -- `coil_design.build_coil_station`.
- `"fem_reference"` -- each coil's force law comes directly from
  `fem.reference_backend.AnalyticReferenceBackend`, called live per force query. No LUT
  file needed for this: the reference backend is a closed-form evaluation (~0.1 ms), cheap
  enough to call during a search rather than needing a pre-swept table. Winding electrical/
  thermal properties (resistance, inductance, thermal mass) are identical between the two --
  only the force law differs, so results are directly comparable.

```powershell
emac-optimize --force-law fem_reference --maxiter 15 --popsize 12
```

**This is not a rounding-error difference.** Evaluating the SAME design (5 coils, 150
turns, 10 A cap) with the velocity governor disabled (the optimizer's actual objective --
see `V_TGT_FULL_THRUST`, which is genuinely open-loop, unlike a `stepper_supervisor`-
tracked `emac-sim` run) gives 4.43 m/s under `"analytic"` vs. 3.68 m/s under
`"fem_reference"` -- a 17% difference. A sensitivity sweep over `i_max_a` on a different
baseline shows the two force laws diverging consistently across the whole swept range (not
just at one point), and a short comparative search converged to different winning designs
under each (`n_coils=6, turns=189` at 9.91 m/s analytic vs. `n_coils=6, turns=183` at 12.04
m/s FEM-reference) -- the coupling model can change which design looks best, not just by
how much. `emac-sim`'s closed-loop `stepper_supervisor` demo doesn't show anywhere near
this much difference between force laws, because its velocity feedback control actively
compensates for whatever the force law gives it -- the optimizer's open-loop objective is
where the FEM pipeline's effect on real analytical conclusions is actually visible.

`"fem_reference"` runs meaningfully slower than `"analytic"` (roughly 10x in one measured
search) -- each force evaluation is now a closed-form elliptic-integral call rather than a
trivial exponential. Budget `--maxiter`/`--popsize` accordingly for a full search; a
sensitivity sweep (tens to low hundreds of evaluations) stays fast either way.

## Visualizing it: the EMAC Optimizer Dashboard's FEM coupling view

The `fem_coupling_analysis` MCP tool (docs/MCP_SERVER.md) computes both coupling curves for
one coil of a design and writes `build/fem_lut/latest_analysis.json` in the same
load-a-file pattern `run_optimization` uses. Open that file in
`tools/web/optimizer_dashboard.html` to see:

- force vs. slug offset, analytic (dashed) overlaid on FEM reference (solid), color-coded by
  current level -- the same divergence the numbers above describe, but as a curve you can
  see the SHAPE of, not just a single peak-force percentage (the two peaks can be nearly
  identical while the curves' widths differ enough to change a simulated trajectory
  substantially -- peak-force alone understates this),
- a stat card with peak force under each model and the percentage divergence, with a warning
  banner if it crosses 10%,
- an axisymmetric coil/slug cross-section schematic (to scale, both halves of the revolve
  shown) so the geometry the curves came from is visible alongside them.

No server or build step -- open the HTML file directly, same as every other view in the
dashboard.

## Known limitations

- PM branch only. The reluctance branch (`Cmag`, soft-iron slugs) isn't modeled here yet --
  both backends assume a pure-PM slug, matching this build's default (`c_mag_n_per_a2 =
  0.0`). A soft-iron/hybrid slug would need real B-H-curve materials in the FEMM backend,
  which is a larger extension.
- The reference backend has all the idealizations listed in its docstring (vacuum
  permeability, an idealized uniformly-magnetized cylinder, no eddy currents). Treat it as a
  shape-accurate stand-in for testing/plumbing, not a substitute for the FEMM backend when
  accuracy actually matters.
- One LUT per coil, built around that coil's OWN geometry in isolation -- mutual coupling
  between adjacent energized coils isn't captured (each coil's contribution is still summed
  independently in `net_force`, same as the analytic model).
