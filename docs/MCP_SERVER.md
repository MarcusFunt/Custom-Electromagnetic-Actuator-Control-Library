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
| `run_optimization(maxiter, popsize, seed, dt, t_end, bounds_overrides, fault_warning_threshold)` | Runs the differential-evolution search (docs/DESIGN_OPTIMIZER.md's 11 knobs). Streams progress via the MCP progress channel and raises an explicit warning as soon as a generation's fault rate crosses `fault_warning_threshold` (default 90%), so a badly-bounded search is visible within the first generation or two. |
| `get_latest_result()` | Returns the current contents of `build/optimize_results/latest.json` without running anything — useful for reattaching to a run in progress. |
| `simulate_design_detailed(knobs, dt, t_end, bootstrap_timeout_s, max_samples)` | Full closed-loop time series (position/velocity/current/temperature, gate crossings) for one fully-specified design — run it on `best_knobs` to see *how* the winning design reaches its exit speed. |
| `sensitivity_sweep(knob, baseline, bounds_overrides, n_points, dt, t_end)` | One-at-a-time main-effect curve for a single knob (wraps `design_sensitivity.sweep_knob`); `baseline` defaults to the latest optimization result. |

`run_optimization` intentionally does not expose `optimize_design.optimize()`'s `workers`
option: the per-evaluation fault/best-so-far instrumentation used for progress reporting
lives in in-process shared state that would not survive being pickled into worker
processes. Use the `emac-optimize --workers N` CLI directly if you want multiprocess
throughput over live progress visibility.

## Live results file

Every `run_optimization` call overwrites `build/optimize_results/latest.json` after
**every generation**, not just at the end — `{"status": "running" | "done", "generation",
"maxiter", "evals_total", "fault_fraction_overall", "best_speed_m_s", "best_knobs",
"elapsed_s", "eta_s", "history": [...]}`. Reload it at any time, including mid-run, in the
**EMAC Optimizer Dashboard** artifact (ask Claude to open it, or regenerate it from this
doc) to see:

- a progress bar (generation / maxiter) with elapsed time and ETA,
- the convergence curve (best speed so far) overlaid with each generation's fault
  fraction, so a search that's stuck at 0 m/s or thrashing on infeasible candidates is
  visible immediately instead of after a multi-minute run completes,
- the current best design's full spec sheet, grouped the same way as
  `optimize_design._print_design`.

Drop a `simulate_design_detailed` result into the same dashboard to see the winning
design's velocity/position trace and gate crossings, or a `sensitivity_sweep` result to see
a knob's main-effect curve — the dashboard auto-detects which of the three JSON shapes it's
been given.

The dashboard itself isn't a file committed to this repo -- it's generated on demand (ask
your MCP client to build one against the JSON shapes documented above, or regenerate it from
this doc if an existing one goes stale). That keeps a client-side visualization out of the
Python package rather than pinning a specific one as canonical.
