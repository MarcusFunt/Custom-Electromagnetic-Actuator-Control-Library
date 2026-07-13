# EMAC linear stepper — real-FEMM design trends

**Run:** 9.04 h, one FEMM instance, **66 coil/magnet geometries** each swept over **32
driver/control settings** = **2,112 real-FEMM designs** (each also run through the cheap
"analytic" model for comparison). Geometries are a seeded-random slice of a 972-cell
factorial, so the sample is balanced across every variable. Exit speed = slug speed past the
last gate, velocity governor disabled (a pure speed-maximization objective).

Each design's coil–magnet force comes from an actual axisymmetric FEMM magnetostatic solve
(≈245 solves per geometry, ~16,000 FEMM solves total), not the synthetic coupling lobe the
optimizer normally uses. **This required fixing 3 real bugs in the FEMM backend first** (§4)
— without them every FEMM design reads 0 m/s.

---

## 1. The headline: it's all about bipolar drive

**Whether the driver can push current *both* directions (H-bridge) or only positive
(single half-bridge) is by far the largest lever — and it multiplies the payoff of almost
every other knob.** Bipolar averages **4.02 m/s vs 1.37 m/s** unipolar (2.9× overall).

Physically: a unipolar driver can only *attract on approach* — it throws away the departure
half of each coil's coupling lobe. A bipolar driver adds *repel-on-departure* thrust (and
can actively force current to zero for a clean cut), so it uses twice the trajectory to
convert force into speed. That's why the moderator effects below are so large.

### Your exact example, quantified
> *"voltage has a bigger impact if you use both positive and negative drive instead of only positive"*

**Confirmed — and it's a 3.7× effect:**

| drive | exit speed gain per +100 V |
|---|---|
| unipolar (positive-only) | **+0.28 m/s** |
| bipolar (±) | **+1.03 m/s** |

→ **Raising the bus voltage buys you ~3.7× as much speed with a bipolar drive as with a
unipolar one.** Under unipolar drive, voltage barely matters past ~40 V (the current can't
do anything useful on the departure side no matter how fast you slew it).

### The same pattern holds for every "more force" knob
Each of these helps far more once the drive is bipolar (this is the real, actionable trend —
these knobs are *near-worthless on a unipolar driver*):

| knob | unipolar effect | bipolar effect | ratio |
|---|---|---|---|
| **magnet remanence** | +0.15 m/s per T | +2.43 m/s per T | **16.7×** |
| **current cap `i_max`** | +0.006 m/s per A | +0.021 m/s per A | **3.7×** |
| **bus voltage** | +0.28 m/s per 100 V | +1.03 m/s per 100 V | **3.7×** |
| **turns** | −0.03 m/s per 100 t | +0.05 m/s per 100 t | **sign flip** |

The turns row is the subtle one: **more turns actively *hurts* on a unipolar driver but
helps on a bipolar one.** More turns = more force per amp but more resistance/inductance;
unipolar can't exploit the extra force (no departure thrust) so it only pays the electrical
penalty, while bipolar converts it.

### One interaction that isn't about polarity
- **Voltage × pulse shape:** voltage helps **1.5× more** with a `square` current pulse
  (+0.79 m/s/100 V) than a smooth `rcos` pulse (+0.52). Square delivers more average current
  per peak, so it has more to gain from voltage headroom — a genuine speed-vs-smoothness knob.

---

## 2. Main effects (ranked by standardized strength)

Δ exit speed per one standard deviation of each knob, across all 2,112 FEMM designs:

| rank | knob | std. effect | direction |
|---|---|---|---|
| 1 | **driver_bipolar** | **+1.32** | bipolar ≫ unipolar |
| 2 | **coil_length_m** | +0.75 | longer coils help |
| 3 | **bus_voltage_v** | +0.63 | more volts help |
| 4 | radial_thickness_m | −0.50 | thicker winding hurts |
| 5 | magnet_length_m | −0.47 | longer/heavier magnet hurts |
| 6 | remanence_t | +0.40 | stronger magnet helps |
| 7 | magnet_radius_m | −0.39 | fatter/heavier magnet hurts |
| 8 | i_max_a | +0.34 | higher current cap helps |
| 9 | pump_envelope | +0.28 | `square` > `rcos` |
| 10 | **turns** | **+0.04** | ~no net effect |

Two findings worth calling out:
- **The three magnet/winding "size" knobs all have *negative* main effects.** Bigger magnets
  and thicker windings add mass and resistance faster than they add usable thrust — for raw
  exit speed, *light and strong* beats *big*. (Remanence, which adds field without mass, is
  the right way to get more magnetic force.)
- **Turns is a near-perfect wash (+0.04).** The turns↔copper-loss trade-off in the winding
  model cancels almost exactly — good validation that the physics model isn't trivially
  "more is better," and a signal that turns is a free knob to spend on packaging/thermal
  rather than speed.

**Best design found: 19.47 m/s** — 260 V, bipolar, square pulse, 70 A cap, 450 turns, 32 mm
coil, 5 mm winding, a *small light* magnet (4 mm radius × 22 mm), N52 remanence (1.25 T).
It maxes every helpful knob and minimizes magnet mass, exactly as the trends predict.

---

## 3. Does the cheap model mislead? (analytic vs real FEMM)

The optimizer ships with a synthetic "analytic" coupling model. Running the identical
2,112 designs through both:

- **The analytic model overpredicts exit speed by a median +15%** (IQR +4% … +34%) — it
  assumes a wider/stronger coupling lobe than the real FEMM field, so it credits designs with
  impulse they don't actually get.
- **8 designs the analytic model calls "moving" actually stall under FEMM** (and 4 the
  reverse). Near the feasibility edge, the cheap model's optimism flips the yes/no answer.

Takeaway: the analytic model is fine for *ranking* driver knobs (the polarity/voltage trends
above are qualitatively identical under both), but it's optimistic by ~15% on absolute speed
and unreliable for marginal-feasibility geometries — use FEMM tables when the answer is close.

---

## 4. Bug review — 3 real bugs found (all fixed + tested)

All three are in `tools/python/emac_sim/fem/femm_backend.py`, all hit the documented
`emac-femgen --backend femm` command, and **none were caught by the test suite** (it skips
FEMM entirely and marks `solve()` `# pragma: no cover`). I found them by running FEMM
end-to-end, fixed all three, added a FEMM-gated regression test, and verified **232 passed,
2 skipped**.

| # | bug | effect | fix |
|---|---|---|---|
| 1 | **Force sign inverted** vs the reference backend & plant convention | FEMM LUT drives the slug *backward* → **0 m/s for every design** | negate the returned force |
| 2 | **Air domain doesn't grow with offset** — slug drawn outside the FEM boundary at far offsets | phantom ~0.9 N force where truth is ~0.07 N, on every coil the slug has passed | `half_extent += abs(offset_m)` |
| 3 | **`mesh_size_m` is a silent no-op** (`automesh=1` overrides it) | force noise stuck at ~3.3%, uncontrollable | pass `automesh=0` |

Details and evidence in `BUG_REVIEW.md`. A 4th, minor one: `max_current_a` in the coil
configs is parsed but never used.

*(Operational note for future FEMM runs: the solver silently **hangs on a hidden dialog** if
its temp-file path exceeds Windows' 260-char MAX_PATH — run FEMM from a short working dir.)*

---

## 5. Deeper analysis (figures + detailed numerics)

Full figure set in `figures/` and the per-level numbers in `DETAILED_TRENDS.txt`.
Additional findings the detailed pass surfaced:

- **Voltage saturates.** Marginal gain per +100 V (bipolar): **+3.8 → +1.4 → +0.5 m/s**
  across 12→40→110→260 V. Most of the benefit is below ~110 V; past that you're paying for
  little. (`fig1`, §3 of the detailed report.)
- **Turns has a sweet spot (~450).** More turns help up to ~450, then *reverse* as
  resistance/inductance dominate — 900 turns is slower than 450. The near-zero *average*
  turns effect is really a rise-then-fall. (`fig1`.)
- **Magnet length has an interior optimum (~22 mm)**; magnet radius and winding thickness are
  monotonically bad for speed (mass/resistance).
- **The cheap analytic model is most wrong exactly where you'd build.** Median overprediction
  hits **+57% at thin windings, +28% at small magnets, +21% at long coils** — the very
  high-performance corner. It's most trustworthy on the slow, heavy designs. (`fig5`.)
- **Unipolar drive is a feasibility risk, not just slow:** 18.7% of unipolar designs stall
  (<0.5 m/s) vs 0.4% bipolar; low voltage (12 V) stalls 16.9%. (`fig6`.)
- **Elasticities** (per +1% of a knob): coil length +0.62%, remanence +0.44%, voltage +0.26%,
  current cap +0.23%, turns +0.02%; winding thickness −0.36%, magnet length −0.36%, magnet
  radius −0.29%.

### The analysis tool suite (all read `results/`, no FEMM needed)
| script | output |
|---|---|
| `build_dashboard.py` | the interactive GUI → `dashboard.html` |
| `make_figures.py` | the 6 figures below → `figures/` |
| `detailed_trends.py` | per-level tables, moderation, diminishing returns, feasibility, model-error, best-per-constraint, elasticities → `DETAILED_TRENDS.txt` |
| `study_viz.py` | shared loading/labels/OLS helpers |

`TRENDS_ANALYSIS.txt` is the headline text report; its generator (`analyze_study.py`) and the
FEMM run harness that produced `results/` (`run_study.py`, `study_lib.py`, needs FEMM) are not
committed here — ask if you want them.

**Figures:** `fig1` main-effect curves (10 knobs) · `fig2` polarity moderation (8 knobs) ·
`fig3` interaction heatmap (all 45 two-way terms) · `fig4` 2-D design maps · `fig5`
analytic-vs-FEMM (scatter + where-it-errs) · `fig6` feasibility, speed distribution, speed-vs-mass Pareto.

## 6. Caveats
- **PM-only slug** (no soft-iron/reluctance branch — the FEM pipeline doesn't model it yet).
- **Coil lengths capped at 32 mm**: much larger coils need FEM domains FEMM struggles to
  mesh; the trends are strongest within the swept ranges.
- **Per-geometry force noise ~0.5–2%** (Maxwell stress tensor). It's *common-mode* within a
  geometry, so the driver interactions (the headline) are essentially immune; the
  across-geometry main effects carry a little noise but are averaged over 66 cells.
- **Exit-speed objective** with the governor off — this is a *max-speed* study, not a
  tracking/efficiency one. A closed-loop tracking controller would compensate for much of the
  force-law difference (that's why the analytic-vs-FEMM gap matters most here, open-loop).
