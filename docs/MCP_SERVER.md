# `emac-mcp` — Model Context Protocol interface

Companion to `docs/DESIGN_OPTIMIZER.md`: that document covers the search itself; this one
covers driving it from an LLM client (Claude Code, Claude Desktop, or any other MCP client)
instead of the `emac-optimize` CLI, with live progress and per-generation fault-rate
reporting so a long search's health is visible while it's still running, not just at the end.

Implemented in `tools/python/emac_sim/mcp_server.py`, built on
[`mcp`](https://pypi.org/project/mcp/)'s `FastMCP`.

## Setup

```powershell
python -m pip install -e .[mcp]
```

Register it with a client, e.g. Claude Code:

```powershell
claude mcp add emac -- emac-mcp
```

Or run it directly for manual/stdio testing: `emac-mcp` (or `python -m emac_sim.mcp_server`).

## Tools

| Tool | Use for |
|---|---|
| `run_optimization(maxiter, popsize, seed, dt, t_end, bounds_overrides, fault_warning_threshold, force_law)` | Runs the differential-evolution search (docs/DESIGN_OPTIMIZER.md's 11 knobs). Streams progress via the MCP progress channel and raises an explicit warning as soon as a generation's fault rate crosses `fault_warning_threshold` (default 90%), so a badly-bounded search is visible within the first generation or two. |
| `get_latest_result()` | Returns the current contents of `build/optimize_results/latest.json` without running anything — useful for reattaching to a run in progress. |
| `simulate_design_detailed(knobs, dt, t_end, bootstrap_timeout_s, max_samples, force_law)` | Full closed-loop time series (position/velocity/current/temperature, gate crossings) for one fully-specified design — run it on `best_knobs` to see *how* the winning design reaches its exit speed. |
| `sensitivity_sweep(knob, baseline, bounds_overrides, n_points, dt, t_end, force_law)` | One-at-a-time main-effect curve for a single knob (wraps `design_sensitivity.sweep_knob`); `baseline` defaults to the latest optimization result. |
| `fem_coupling_analysis(knobs, coil_index, n_offsets, n_currents)` | Force-vs-slug-offset curves for one coil, comparing the default analytic coupling shape against `fem.reference_backend`'s real (non-Gaussian) shape, at a few current levels. `knobs` defaults to the latest optimization result. Writes `build/fem_lut/latest_analysis.json`. |

`force_law` (on `run_optimization`/`simulate_design_detailed`/`sensitivity_sweep`) selects
`"analytic"` (default, unchanged behavior) or `"fem_reference"` -- see
docs/FEM_PIPELINE.md's "Using it in the design optimizer and sensitivity sweeps" for how much
this can change both the reported speed and the winning design. `"fem_reference"` runs
meaningfully slower per evaluation (a closed-form elliptic-integral call instead of a trivial
exponential) -- budget `maxiter`/`popsize` accordingly.

`run_optimization` intentionally does not expose `optimize_design.optimize()`'s `workers`
option: the per-evaluation fault/best-so-far instrumentation used for progress reporting
lives in in-process shared state that would not survive being pickled into worker
processes. Use the `emac-optimize --workers N` CLI directly if you want multiprocess
throughput over live progress visibility.

## Live results file

Every `run_optimization` call overwrites `build/optimize_results/latest.json` after
**every generation**, not just at the end — `{"status": "running" | "done", "generation",
"maxiter", "evals_total", "fault_fraction_overall", "best_speed_m_s", "best_knobs",
"elapsed_s", "eta_s", "history": [...]}`.

## GUI: `tools/web/optimizer_dashboard.html`

A self-contained, dependency-free HTML page (open it directly in a browser -- no server, no
build step) that visualizes that live file, or any saved `simulate_design_detailed` /
`sensitivity_sweep` / `design_sensitivity.interaction_sweep` / `fem_coupling_analysis` result.
It auto-detects which of the five JSON shapes it's been given.

**Watching a long run live:** click "Open results file..." and select `build/optimize_results/
latest.json`. In Chrome/Edge (the File System Access API) the page keeps a handle to that file
and polls it every second, so the SAME open tab updates automatically as new generations
complete -- no reloading, no re-running anything. Other browsers (Safari, Firefox) don't support
that API; there, re-open or drag-and-drop the file again to refresh, or use the "Paste JSON..."
box. Nothing is uploaded anywhere -- the file is read locally and stays in the page.

What it shows for a running or finished search:
- a status pill (Running / Done) and a progress bar (generation / maxiter) with elapsed time
  and ETA,
- an explicit warning banner if the overall fault rate is high (bounds likely infeasible),
- the convergence curve (best speed so far) overlaid with each generation's fault fraction, so
  a search that's stuck at 0 m/s or thrashing on infeasible candidates is visible immediately
  instead of after a multi-minute run completes,
- the current best design's full spec sheet, grouped the same way as
  `optimize_design._print_design` (driver / topology / coil winding / slug-magnet),
- a sortable-by-eye generation history table.

Drop a `simulate_design_detailed` result in instead to see the winning design's position/
velocity/current/temperature time series and gate crossings, a `sensitivity_sweep` result to
see a knob's main-effect curve, a `design_sensitivity.interaction_sweep` result (the raw
Python module, not an MCP tool here) to see a pairwise heatmap, or a `fem_coupling_analysis`
result to see one coil's force-vs-offset curve under both the analytic and FEM-reference
coupling shapes (overlaid, color-coded by current level) alongside an axisymmetric coil/slug
cross-section schematic and a peak-force divergence stat -- see docs/FEM_PIPELINE.md.

**Reading the charts.** Every chart carries labelled value axes and gridlines, and hovering
anywhere on one snaps a crosshair to the nearest sample and reads out each series' value at
that point. Series that share a time axis but not a unit -- position against velocity, coil
current against temperature -- are drawn on independent left/right axes, so both traces use
the full plot height instead of the smaller one flattening against the baseline. Chart labels
hold a constant size regardless of window width.

**Theme.** The header has an auto / light / dark switch; `auto` follows the OS setting. The
choice persists in `localStorage`.

**Keyboard.** With the slug-animation track focused: `Space` plays/pauses, `<-` / `->` step one
sample (hold `Shift` for ten), `Home` / `End` jump to either end of the trajectory.
