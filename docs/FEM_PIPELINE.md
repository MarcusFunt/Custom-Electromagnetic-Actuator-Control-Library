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
| `fem/reference_backend.py` | Analytic backend (closed-form Biot-Savart, reusing `coil_design.off_axis_radial_field_cylinder_magnet`) evaluated at the requested offset instead of only at its peak. **Not a real FEM solve** -- no iron, no saturation, vacuum permeability everywhere -- but it traces the coil's actual (non-Gaussian) coupling shape and needs no external tools, so it's both the default backend and what this repo's own tests run against. |
| `fem/femm_backend.py` | Real backend: builds and solves an axisymmetric magnetostatic problem in [FEMM](http://www.femm.info/) via its optional `femm` Python module, and reads force off the Maxwell stress tensor. Raises `FemmNotAvailableError` with a clear message if FEMM isn't installed -- everything else in this package works without it. |
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
  tail beyond that.

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
  only the force law differs, so results are directly comparable. **Neither `"analytic"` nor
  `"fem_reference"` is a real FEM solve** -- both are closed-form approximations (no iron, no
  saturation, vacuum permeability everywhere); see the reference backend's own module
  docstring. Don't mistake the name `"fem_reference"` for "this ran through FEMM."
- `"femm"` -- each coil's force law comes from an actual FEMM axisymmetric magnetostatic
  solve (`fem.femm_backend.FemmBackend`). One sweep per design is shared across every coil
  (`optimize_design._femm_force_lut`/`_femm_geometry_key`): `FEMBackend.solve()` never reads
  a coil's `position_m`, and every coil in one design has identical winding/magnet geometry,
  so the coils differ only in *where* the shared table gets evaluated (`net_force`'s
  `offset = x - coil.position_m`). Results are memoized per unique geometry, so repeated
  calls with the same knobs don't re-sweep (the cache key includes `i_max_a`, since it sets
  the swept current span -- see below). Uses a coarser grid (15 offsets x 5 currents,
  `optimize_design._FEMM_VERIFY_N_OFFSETS`/`_FEMM_VERIFY_N_CURRENTS`) than `emac-femgen`'s
  31x11 LUT-file default, trading interpolation precision for wall-clock time, but the
  current axis is always swept out to the design's own `i_max_a` (not the 6&nbsp;A generic
  default) -- `ForceLUT` clamps out-of-range queries rather than extrapolating, so a table
  that stopped short of a design's actual operating current would silently understate its
  force at every point past that edge. Requires FEMM
  installed; raises `FemmNotAvailableError` otherwise. **Cannot drive `optimize()`'s search
  itself** -- `differential_evolution` calls the force law potentially millions of times
  across a run, and each FEMM solve takes seconds; `optimize(force_law="femm")` raises
  `ValueError` pointing at `verify_with_femm` instead (see below). `simulate_design`/
  `build_params`/`simulate_design_detailed`/`sensitivity_sweep` accept it directly for
  single-design calls, where the cost is one sweep, not millions.

```powershell
emac-optimize --force-law fem_reference --maxiter 15 --popsize 12
```

### Verifying the winning design against real FEMM

Because the search itself can't use `"femm"`, `optimize()` (and the `emac-optimize` CLI, and
the `run_optimization` MCP tool) instead take a separate `verify_with_femm` option that
re-simulates ONLY the winning design under `force_law="femm"` after the search finishes,
reporting it as a distinct `femm_speed` -- never substituted for the search's analytic
number, so a result can't silently look FEMM-verified when it isn't:

- `None` (default) -- auto: verify if FEMM is importable, otherwise leave `femm_speed=None`
  with a printed/JSON note (`"FEMM not installed -- ... is analytic-only"`) rather than
  failing or staying silent about which number you're looking at.
- `True` -- require FEMM; raises `FemmNotAvailableError` if it's missing.
- `False` -- skip verification entirely.

```powershell
emac-optimize --maxiter 15 --popsize 12                 # verify-femm defaults to "auto"
emac-optimize --maxiter 15 --popsize 12 --verify-femm no # analytic-only, no FEMM attempt
```

The CLI prints both numbers when verification succeeds:

```
search reported 4.21 m/s at low fidelity; re-verified (analytic) at high fidelity: 4.18 m/s
FEMM-verified exit speed: 4.05 m/s
```

The MCP `run_optimization` tool's JSON result and `build/optimize_results/latest.json`
snapshot carry the same two numbers as `best_speed_m_s` (analytic/fem_reference) and
`femm_speed_m_s` (real FEMM, `null` if not verified) plus a `femm_note` explaining why when
`femm_speed_m_s` is `null`.

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

### Field lines (real FEMM only)

Pass `field_lines=true` to `fem_coupling_analysis` to overlay REAL traced magnetic field
lines on the cross-section schematic -- not an approximation, and not the analytic/
fem_reference path: `fem.femm_backend.FemmBackend.field_lines` runs an actual FEMM solve at
one operating point and RK4-integrates along FEMM's own `mo_getb(r,z)` postprocessor
(confirmed present and working via `pyfemm`), seeded evenly around the magnet's real
boundary faces (bottom cap, outer side, top cap -- never the r=0 symmetry axis, which isn't
a physical surface). Useful for two things: seeing what the FEM solve actually looks like
(rather than trusting a scalar force number), and building intuition for control-scheme
design on the electromagnets driving the PM slug -- where the field points and how strong
it is at a given offset/current is exactly what governs how you'd want to sequence or shape
coil currents.

Parameters: `field_line_offset_m` (default `0.0` -- slug centered on the coil, the
strongest-coupling operating point) and `field_line_current_a` (default the design's
`i_max_a`). Default `field_lines=false` -- this is opt-in and adds real FEMM wall-clock
time (a handful of seconds; each line traces up to `_FIELD_LINE_MAX_POINTS` RK4 steps in
each direction, deliberately capped low since the schematic only ever shows a padded
near-field view, not FEMM's full 6x-margin solved domain -- points beyond the visible area
are wasted compute). Requires FEMM installed: without it, the result's `"field_lines"` is
`null` and `"field_lines_note"` explains why (same null-plus-note pattern as
`run_optimization`'s `femm_speed_m_s`/`femm_note` -- see "Verifying the winning design
against real FEMM" above) rather than raising or silently omitting the field.

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
