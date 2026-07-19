# EMAC Phase 0 Host Simulator

This repository is currently a Phase 0 host-only simulator for an event-first
electromagnetic actuator control library. It models two geometries built from the
same shared primitives (see `docs/DESIGN_LINEAR.md` section 1 for exactly what's
shared and what isn't):

- a **soft-iron, attract-only magnetic pendulum**, one bottom coil + one bottom
  photogate (`docs/DESIGN.md`) -- the original Phase 0 target, and
- a **linear one-way stepper** (coilgun-style): a permanent-magnet slug pulled
  through a tube by N air-core coils firing in sequence, sensed by photogates
  between them (`docs/DESIGN_LINEAR.md`).

Both validate the estimator/supervisor control loop before firmware is written.
The linear stepper additionally has a **design-space optimizer**
(`docs/DESIGN_OPTIMIZER.md`) that searches driver/winding/magnet/topology knobs
to maximize slug exit speed, using a physically-grounded parametric model
(turns and dimensions genuinely trade off against resistance/inductance/thrust,
not just "more is better").

The immediate goal is a configurable **virtual actuator lab**: define fictional
pendulum/stepper, gate, coil, driver, and controller values in a TOML file, run
the sim on a PC, inspect the plots/visualizer, and use the results to guide
later hardware.

The future hardware target remains the one documented in `docs/DESIGN.md`:
ESP32-S3, hardware timer capture for the photogate, and a unipolar coil power
stage for a soft-iron bob.

## Documentation

| Doc | Covers |
|---|---|
| [`docs/DESIGN.md`](docs/DESIGN.md) | Firmware/hardware design spec: event-first architecture, ESP32-S3 target, control-law derivations. |
| [`docs/DESIGN_LINEAR.md`](docs/DESIGN_LINEAR.md) | Linear one-way stepper: physics, electrical/thermal dynamics, estimator, supervisor. |
| [`docs/DESIGN_OPTIMIZER.md`](docs/DESIGN_OPTIMIZER.md) | Design-space optimizer: the physical winding/magnet model, knobs, and how to run it. |
| [`docs/MCP_SERVER.md`](docs/MCP_SERVER.md) | Model Context Protocol interface: drive the optimizer from an LLM client with live progress/fault-rate reporting, plus the `tools/web/optimizer_dashboard.html` GUI. |
| [`docs/PHYSICS_ENGINE_ANALYSIS.md`](docs/PHYSICS_ENGINE_ANALYSIS.md) | Host physics engine's numerical methods (integrator, event interpolation), known limits, and roadmap. |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | How the engine's accuracy is *measured* (analytic coupling vs real FEMM to ~1-2%, integrator convergence order, energy/back-EMF closure), the numbers, and how to reproduce them for your own geometry. |
| [`docs/FEM_PIPELINE.md`](docs/FEM_PIPELINE.md) | FEM axisymmetric table-generation pipeline: geometry builder, FEMM/analytic-reference backends, `emac-femgen` CLI, the `emac-femqc`/`emac-femcheck` analysis tools, and the LUT hook into the plant that replaces the synthetic coupling lobe. |

## Setup

```powershell
python -m pip install -e .[dev]
```

## Run a Configurable Simulation

```powershell
emac-sim --config examples/configs/pendulum_softiron_1gate.toml --outdir build/phase0
```

Override the configured duration for a quick smoke run:

```powershell
emac-sim --config examples/configs/pendulum_softiron_1gate.toml --t-end 3 --no-plots
```

The legacy fixed-demo command still works and uses the built-in Phase 0 defaults:

```powershell
emac-phase0 --no-plots
```

`emac-sim` dispatches on the config's `[sim] kind` -- the same command also runs the
linear one-way stepper, just by pointing it at a different config:

```powershell
emac-sim --config examples/configs/linear_stepper_5coil.toml --outdir build/linear
```

## Run the Visual Simulator

Generate a standalone browser visualizer from the default Phase 0 scenario:

```powershell
emac-visual --outdir build/visual
```

Generate it from a fictional-hardware config:

```powershell
emac-visual --config examples/configs/pendulum_softiron_1gate.toml --outdir build/visual
```

If the console script is not on `PATH`, use the module entrypoint:

```powershell
python -m emac_sim.visual --config examples/configs/pendulum_softiron_1gate.toml --outdir build/visual
```

`emac-visual`'s interactive tube-canvas animation for the linear stepper isn't built yet
(`docs/DESIGN_LINEAR.md` section 6) -- use `emac-sim`'s static plots for that geometry.

## Run the Design Optimizer

Search driver voltage, coil turns/dimensions, current waveform, single-ended vs. H-bridge
switching, coil count, and magnet properties to maximize the linear stepper's slug exit
speed (see `docs/DESIGN_OPTIMIZER.md` for what each knob means and how the search is
scoped):

```powershell
emac-optimize --maxiter 25 --popsize 15
```

Add `--force-law fem_reference` to search against the FEM reference backend's real coupling
shape instead of the default analytic estimate -- see `docs/FEM_PIPELINE.md`'s "Using it in
the design optimizer and sensitivity sweeps" section for how much this can actually change
both the reported speed and the winning design.

The legacy entrypoint still works:

```powershell
python tools/python/run_phase0.py
```

## Drive the Optimizer from an LLM Client (MCP)

```powershell
python -m pip install -e .[mcp]
claude mcp add emac -- emac-mcp
```

Exposes `run_optimization`, `simulate_design_detailed`, `sensitivity_sweep`, and
`get_latest_result` as MCP tools, with live progress and per-generation fault-rate
warnings so a long search's health is visible while it's still running. See
`docs/MCP_SERVER.md`.

## Generate FEM Force Tables for the Linear Stepper

Replace the linear stepper's synthetic coupling lobe with a swept axisymmetric field
table -- real FEM via [FEMM](http://www.femm.info/) if installed, or a shape-accurate
analytic-reference backend if not:

```powershell
emac-femgen --config examples/configs/linear_stepper_5coil_fem.toml --outdir build/fem_lut
emac-sim --config examples/configs/linear_stepper_5coil_fem.toml --outdir build/linear_fem
```

See `docs/FEM_PIPELINE.md` for the geometry knobs, backend choices, and how a coil's
`force_lut_path` hooks into the plant.

## GUI: One App to Run, Sweep, and Visualize

```powershell
emac-gui
```

`emac-gui` starts a small local web app (stdlib only -- no new dependencies) and opens it in
your browser. It unifies what used to be three separate pages into one **EMAC control lab**:

- **Runs & results** -- every search and sweep you've run, turned into interactive charts.
  *Design searches* (the RL hardware BO, the FEMM Bayesian optimizations) show the
  **speed↔efficiency scatter** with the Pareto-optimal set highlighted, a **convergence** curve,
  and per-parameter exploration plots; *frontiers* (the RL λ-sweeps) plot speed vs efficiency with
  an optional **reluctance-literature overlay**; the *factorial sweep* shows the
  **analytic-vs-FEMM agreement** and a switchable design-trend chart. Every run has a
  **Reproduce** button that pre-fills the Run tab (scoped small for a fast re-validation) so you
  can re-run it to confirm the result -- all points hover for the exact geometry, and export to
  CSV/SVG. **Click any candidate** to play a **launch animation** of that geometry, rendered
  inline as a looping **animated GIF** -- the slug accelerating through the multi-stage tube with
  the coils firing (color = current polarity), a velocity-vs-position trace, and a live t/v/coil/
  current readout. It runs on the fast analytic force by default, or is rebuilt on a **real FEMM
  force table** for that exact coil/slug with one click, and the GIF is downloadable for slides
  and papers.
- **Run a tool** -- pick any command (`emac-sim`, `emac-optimize`, `emac-femgen`, `emac-femqc`,
  `emac-femcheck`, `emac-rl-train`, the hardware BO×RL search, or the FEMM studies), set its
  options in a form, run it, and watch its output stream live -- the same as a terminal, but
  wired into the app.
- **Sweep & estimate** -- configure a FEM force-table sweep (how *large*: offset/current/
  geometry counts; how *detailed*: mesh fineness), get a real wall-clock **time estimate**
  before you commit (it times a couple of solves and projects the whole run), then start it
  and watch a **live progress bar with ETA** as the points solve.
- **Force tables** -- load a directory of tables, triage the whole batch in one quality-control
  table, then drill into any table: stat tiles (peak thrust, k_a, x_c, coupling width),
  **interactive** coupling curves and a force heatmap (hover for exact values), the analytic-
  vs-FEM overlay, the quality checks, and the source geometry -- with **CSV / SVG export**.
- **Optimizer run** -- live convergence chart and best-design spec sheet from a running search
  (`build/optimize_results/latest.json`).

Niceties: it **remembers** your last config/directory/theme, raises a **toast** when a
background job finishes, has **keyboard shortcuts** (`1`-`6` switch views), and collapses to
an icon rail on narrow windows.

### Reaching it from another device

```powershell
emac-gui --lan            # serve on 0.0.0.0; prints the http://<your-ip>:8765/ to open
```

By default the GUI binds `127.0.0.1`, so it is reachable only from the machine it runs on.
`--lan` binds all interfaces so a phone or laptop on the same Wi-Fi can open it — the startup
banner prints the exact URL. **There is no password**: anyone who can reach that address can use
the GUI to run the whitelisted tools on the host machine and read this project's files, so only
do this on a network you trust. On Windows you may also have to allow Python through the
firewall the first time (Windows Defender will prompt, or add an inbound rule for the port).

The older standalone pages (`tools/web/optimizer_dashboard.html`, the FEMM-trends
`studies/femm_trends/dashboard.html`) still open directly in a browser with no server, but
`emac-gui` supersedes them for interactive use -- it's the one that can also *run* things, and
its **Runs & results** page reads the same on-disk logs those dashboards were built from.

## Verify

```powershell
python -c "import emac_sim"
python -m pytest
```

## Next Milestone

Phase 0B is the configurable virtual-hardware milestone: TOML configs for fictional
pendulums, gates, coils, drivers, and controller settings; visual comparison of runs;
and later sweeps over coil/sensor/controller parameters.

Phase 1 is the first hardware milestone: ESP32-S3 capture input, unipolar coil
power stage, timing-budget proof, and a sustained one-gate swing.
