# FEMM design-trends study

A real-FEMM design-of-experiments study of the linear one-way stepper: **66 coil/magnet
geometries** (each a real axisymmetric FEMM solve, ~16k solves total) × **32 driver/control
settings** = **2,112 designs**, each also evaluated under the cheap "analytic" coupling model
for comparison. Objective: slug exit speed, velocity governor disabled (pure speed max).

Producing it required fixing three real bugs in `fem/femm_backend.py` (sign, air-domain,
mesh) — see [`BUG_REVIEW.md`](BUG_REVIEW.md); those fixes are committed separately in the
package source.

## Requirements — FEMM is a core requirement (to *re-run* the study)
**`run_study.py` / `study_lib.py` drive REAL axisymmetric FEMM solves — FEMM is a hard,
core dependency, not optional.** The harness has no analytic fallback (that is what the
committed `results/` dataset and the repo's reference backend are for): without FEMM every
solve fails and the run produces nothing. To reproduce the dataset you need, on **Windows**:

- the **FEMM application** — <http://www.femm.info/>
- **`pip install pyfemm`** — the Python bindings the harness drives over ActiveX/COM
- the project package — `pip install -e .[dev]` from the repo root (`numpy`/`scipy`)

The **analysis tools** (`dashboard.html` + `build_dashboard.py`, `make_figures.py`,
`detailed_trends.py`, `study_viz.py`) need only `numpy` (+ `matplotlib` for figures) — they
read the committed `results/` and never touch FEMM. So you can explore all the trends with
zero FEMM setup; you only need FEMM to regenerate the raw data.

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

## Regenerate the analysis (no FEMM — reads `results/`, needs numpy [+ matplotlib])
```
python build_dashboard.py      # -> dashboard.html
python make_figures.py         # -> figures/fig1..6.png
python detailed_trends.py      # -> stdout (redirect to DETAILED_TRENDS.txt)
```
`study_viz.py` is the shared loader/helpers.

## Re-run the FEMM study itself (needs FEMM — see Requirements above)
```
python run_study.py            # ~9 h serial, real FEMM; writes into ./study/, resumable
```
`study_lib.py` is the bridge from the corrected FEMM backend into the exit-speed sim.
Knobs via env vars: `STUDY_BUDGET_S` (wall-clock, default 9 h), `STUDY_WORKERS` (default 1 —
serial; pyfemm COM isn't safe for concurrent instances), `STUDY_MAX_CELLS`, `STUDY_SUBDIR`.
A fresh run writes to `studies/femm_trends/study/results/`, so it will not overwrite the
canonical `results/` committed here.
