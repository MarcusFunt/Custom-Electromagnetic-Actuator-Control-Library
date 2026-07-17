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

## Search for a *fast* design with Bayesian optimization — `bo_search.py` (needs FEMM)

The factorial sweep above spends equal, expensive FEMM budget on hopeless-slow and fast
designs alike — most of its 972 cells land under ~3 m/s while only a thin tail reaches ~20.
`bo_search.py` instead **concentrates real-FEMM effort in the fast end of the design space**
with Bayesian optimization (a Gaussian-process surrogate + Expected-Improvement acquisition,
via [`scikit-optimize`](https://scikit-optimize.github.io/)), and **warm-starts from whatever
the sweep has already produced** so that compute is reused, not discarded.

**What it optimizes.** BO searches the **6 expensive geometry knobs** (`turns`,
`coil_length_m`, `radial_thickness_m`, `magnet_radius_m`, `magnet_length_m`, `remanence_t`)
over the study's proven FEMM-meshable ranges. Each candidate geometry builds **one** real
axisymmetric FEMM force LUT (the expensive step — the LUT is position-independent, so it
depends only on those 6 knobs), then a **cheap inner sweep over the 32 driver/control combos**
(`bus_voltage_v × driver_bipolar × pump_envelope × i_max_a`, at `n_coils=5`) reuses that one
LUT to find the best driver. The geometry's score is that best inner value. Objective is
**exit speed (m/s)** by default (`--objective`, also `energy` = ½·m·v² and `momentum` = m·v).

**Run it** (needs FEMM, and **exclusive** FEMM — see below):
```
python bo_search.py --n-calls 250          # ~250 real-FEMM geometry evals; warm-starts from ./study/results
```
It writes `bo/bo_eval_log.jsonl` (the source of truth) and a live, GUI-compatible snapshot at
`build/optimize_results/latest.json`, so **emac-gui → Optimizer** shows the convergence curve
and best design as it runs. Useful flags: `--objective {speed,energy,momentum}`,
`--n-coils N`, `--sim-t-end S` (inner-sim horizon; default 3.0 = the study's value, lower is
faster and only drops sub-~1 m/s designs), `--budget-s` (wall-clock cap), `--no-warm-start`,
`--warmstart-dir DIR` (repeatable), `--timeout` (hard per-eval seconds).

**Resume.** Just run it again with the same `--outdir` — it replays `bo_eval_log.jsonl` via
`tell()` to reconstruct the optimizer, then continues toward `--n-calls` (a *total* budget,
not "N more").

**Warm-start** reads the **corrected** current sweep (`./study/results/`) by default, reducing
each geometry to its best `force_law="femm"` exit speed over drivers. The legacy committed
`results/` (pre-fix stress-tensor extraction) is **excluded on purpose** — its objectives are
distorted and would mislead the surrogate; add it explicitly with `--warmstart-dir results`
only if you know why you want it.

**Robustness / EXCLUSIVE FEMM.** Concurrent FEMM instances hang (observed, multi-hour), so
run BO only when no other FEMM is active (i.e. stop the factorial sweep first). Each geometry
is evaluated in a **subprocess with a hard timeout**: a wedged FEMM COM call (uninterruptible
in-thread) is killed with the process, `femm.exe` is force-killed, that geometry scores 0, and
the search moves on. The subprocess runs FEMM from a short cwd (`C:\femmwork\bo`) to dodge the
Windows MAX_PATH solver hang, same as `run_study.py`.

**Test without FEMM.** `--backend reference` swaps the analytic reference backend in for the
FEMM LUT (no FEMM, no subprocess) — that path is what `tests/test_bo_search.py` exercises in
CI. It's for plumbing/tests only, not a substitute for a real solve.

## Reluctance projectiles — `--slug-type reluctance`

Everything above optimizes a **permanent-magnet (PM)** slug (Lorentz force, bipolar drive). The
coilgun literature is mostly **reluctance** guns — a passive **soft-iron** slug pulled toward
the coil center (`F ∝ I²·dL/dx`, attract-only, saturating). Pass `--slug-type reluctance` to
optimize that device class instead:

```
python bo_search.py --slug-type reluctance --n-calls 50
```

What changes under the hood (PM stays the default, untouched):
- **FEMM slug** is nonlinear-B-H steel (`1018 Steel`), not NdFeB — so real saturation (the
  dominant limiter for reluctance guns) is captured.
- **Current sampling** is 6 non-negative magnitudes `[0 … 90 A]` (the `∝I²` force isn't linear,
  so the PM 3-point linear reduction doesn't apply); the force is even in current.
- **Drive** is effectively unipolar — reluctance can't be repelled, so `driver_bipolar`/the
  departure-repel kick just don't engage.
- **Warm-start** only ingests sweep rows of the same `slug_type` (the committed PM sweep won't
  seed a reluctance run — it starts cold).

The **analytic** reluctance force model (`coil_design.reluctance_force_model`, a coenergy /
inductance-gradient estimate) is deliberately **coarser** than the PM one — real FEMM is the
accuracy reference, same posture as the PM analytic model. Elsewhere: `optimize_design
--slug-type reluctance` (fast analytic search), `emac-femgen --slug-type reluctance` (generate a
reluctance LUT), and the **GUI** sweep/optimizer forms have a slug-type selector; the Analyze
tab auto-detects a reluctance table from its LUT metadata (QC stops expecting a linear-in-current
force, and the force-vs-current chart shows the even/saturating shape).
