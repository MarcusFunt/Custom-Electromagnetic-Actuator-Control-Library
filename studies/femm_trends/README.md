# FEMM design-trends study

A real-FEMM design-of-experiments study of the linear one-way stepper: **66 coil/magnet
geometries** (each a real axisymmetric FEMM solve, ~16k solves total) × **32 driver/control
settings** = **2,112 designs**, each also evaluated under the cheap "analytic" coupling model
for comparison. Objective: slug exit speed, velocity governor disabled (pure speed max).

Producing it required fixing three real bugs in `fem/femm_backend.py` (sign, air-domain,
mesh) — see [`BUG_REVIEW.md`](BUG_REVIEW.md); those fixes are committed separately in the
package source.

## Open the GUI
Double-click **[`dashboard.html`](dashboard.html)** (or drag it into any browser). It's a
single self-contained file — no server, no build, no dependencies — with a live filter
sidebar: toggle any level of any of the 10 knobs and every view (summary tiles, main-effect
curves, interaction heatmap, moderation, 2-D design maps, analytic-vs-FEMM, feasibility)
recomputes instantly.

## Headline findings
- **Bipolar (±) drive is the dominant lever** (mean 4.02 vs 1.37 m/s unipolar) and multiplies
  every other knob: voltage helps **3.7×** more, remanence **16.7×** more, current cap 3.7×;
  turns even flips from hurting to helping.
- **Voltage saturates** (+3.8 → +0.5 m/s per 100 V from 12→260 V); **turns has a sweet spot
  (~450)**; light-and-strong magnets beat big ones.
- **The analytic model overpredicts FEMM by ~15% median** — worst (+57%) exactly at the
  high-performance corner (thin windings, small magnets).

Full write-up: [`TRENDS_REPORT.md`](TRENDS_REPORT.md). Detailed numbers:
[`DETAILED_TRENDS.txt`](DETAILED_TRENDS.txt) · headline text: `TRENDS_ANALYSIS.txt`.

## The dataset — `results/`
66 files `cell_NNNN.jsonl`, one per geometry, 64 rows each (32 driver settings × 2 force
laws). One JSON object per row:

| field | meaning |
|---|---|
| `cell_id` | geometry index (0–971 in the full factorial; 66 sampled) |
| `turns`, `coil_length_m`, `radial_thickness_m`, `magnet_radius_m`, `magnet_length_m`, `remanence_t` | geometry knobs |
| `bus_voltage_v`, `driver_bipolar`, `pump_envelope`, `i_max_a`, `n_coils` | driver/control knobs |
| `force_law` | `"femm"` (real solve) or `"analytic"` (cheap model) |
| `exit_speed_mps` | slug speed past the last gate (the response) |
| `sim_error` | null unless the sim raised |

`cell_qc.jsonl` has per-geometry QC (peak force, force-symmetry error, tail decay, solve
failures). Data quality: 0 sim errors, 99.9% of designs moved, 3/66 cells had ≤8% solve
failures (NaN-filled).

## Regenerate everything (reads `results/` only, needs numpy [+ matplotlib for figures])
```
python build_dashboard.py     # -> dashboard.html
python make_figures.py         # -> figures/fig1..6.png
python detailed_trends.py      # -> stdout (redirect to DETAILED_TRENDS.txt)
```
`study_viz.py` is the shared loader/helpers. Not included here: the FEMM run harness
(`run_study.py`, `study_lib.py`) that generated `results/` — it needs FEMM installed.
