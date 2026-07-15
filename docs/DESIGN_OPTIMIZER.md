# `emac` — design-space optimizer

Companion to `docs/DESIGN_LINEAR.md`. That document covers the *control* problem (given
fixed hardware, how do you drive it well); this one covers the *hardware* problem: given a
set of physical knobs -- driver voltage, coil turns and dimensions, current-waveform
shape, single-ended vs. H-bridge switching, number of coils, and slug/magnet properties --
what combination makes the slug go as fast as possible?

Implemented in `tools/python/emac_sim/coil_design.py` (physical parametrization) and
`optimize_design.py` (the search itself); run via `emac-optimize` or
`python -m emac_sim.optimize_design`.

## 0. Why this needed a physical model first

Before this, `CoilStation`'s `resistance_ohm`, `inductance_h`, and `k_a` were independent,
hand-set placeholder constants -- there was no relationship connecting "wind more turns"
to any downside. Optimizing "over turns and coil dimensions" against that would be
meaningless: more turns would only ever help (more `k_a`), so the search would trivially
diverge to whatever upper bound you set. `coil_design.py` exists to make that trade-off
real: turns and coil dimensions are **not** independent knobs -- given a fixed winding
envelope, more turns means thinner wire, which raises resistance faster than linearly (see
section 1). That's what makes "optimize over turns" a genuine optimization rather than a
one-directional slider.

## 1. The physical model (and its honest limits)

**Coil winding** (`wind_coil`): given `turns`, an axial `coil_length_m`, a radial
`radial_thickness_m` (the winding's outer envelope, around a bore sized to the slug), and a
`packing_factor` (~0.7-0.85 for real round-wire winding, 0.785 theoretical max):

```
copper_area = coil_length * radial_thickness * packing_factor
wire_area   = copper_area / turns              -> wire_diameter = sqrt(4*wire_area/pi)
mean_radius = bore_radius + radial_thickness/2
resistance  = copper_resistivity_ohm_m(temperature_c) * (turns * 2*pi*mean_radius) / wire_area
inductance  = blend of Wheeler's single-layer and multi-layer (1928) formulas   (see below)
```

Both R and L scale roughly with `turns^2` for a fixed envelope (wire length grows with
turns, wire area shrinks with turns, and resistance is length/area) -- confirmed
numerically in `tests/test_coil_design.py` (doubling turns more than triples both).
Resistivity is temperature-dependent (`copper_resistivity_ohm_m`, linear about the 20 C
reference, ~0.39%/C -- `docs/DESIGN.md` flags this as "mandatory, not optional" for a real
build); `wind_coil(..., temperature_c=...)` defaults to 20 C. Inductance blends Wheeler's
single-layer (`~a^2 N^2/(9a+10b)`) and multi-layer (`0.8 a^2 N^2/(6a+9b+10c)`, with
`a`=mean radius, `b`=length, `c`=radial build) air-core formulas by the radial-build ratio
`c/a` (`coil_design._solenoid_inductance_h`). The single-layer form alone -- what this model
used previously -- ignores `c` entirely and OVER-estimates a thick multi-layer winding by
30-75% (the optimizer explores `radial_thickness_m` up to 40mm on a ~10mm-radius coil, well
into that regime); the blend tracks a direct Maxwell mutual-inductance sum to within ~4%
across thin/thick and short/long geometries (`tests/test_coil_design.py`). Getting `L` right
matters because it sets each coil's `L/R` time constant, which governs how fast the `"rl"`
current loop can build current during a pump window.

**Self-heating now feeds back into every candidate the search evaluates.**
`build_params` sets `LinearActuatorParams.thermal_model=True`: each coil's temperature
integrates from its own `i^2*R` dissipation over the run (`plant.thermal_step`, a one-node
thermal model -- see `docs/DESIGN_LINEAR.md` section 2.4), and the resulting hotter,
higher-resistance winding feeds back into the `"rl"` current loop, so a design can no
longer look fast purely because it never paid a heating penalty. `thermal_mass_j_per_k`
is derived from the winding's own copper volume (a lower bound -- real bobbin/potting
mass adds more); `thermal_resistance_k_per_w` has no such derivation and stays a
placeholder constant (8 K/W) like every other uncalibrated value here, pending a real
measurement of how a physical coil actually sheds heat. **Remaining limit:** this models
self-heating WITHIN one run, not a duty-cycle notion of repeated runs back-to-back --
firing this actuator once, cooling, then firing again isn't distinguished from doing so
continuously.

**Magnet / slug** (`magnet_mass_kg`, `estimate_k_a`, `build_coil_station`): the slug is a
cylindrical magnet of given radius, length, and remanence (`Br`). Mass is volume x density
(NdFeB ~7500 kg/m^3 default). The PM-branch thrust constant follows the standard voice-
coil relation `F = B_gap * L_wire * i`, `L_wire = turns * 2*pi*mean_radius` -- **using the
RADIAL field component B_rho, not the axial one B_z**. This matters: force on an azimuthal
(coil) current comes from `I * dL x B`, and an azimuthal `dL` crossed with a radial B gives
an axial force -- B_z doesn't enter that cross product at all. An earlier version of this
model used B_z (evaluating the on-axis field formula at the radial clearance "as if" it
were an axial standoff), which wasn't just less accurate, it was the wrong physical
quantity, and it showed: k_a came out *negative* for several reasonable coil geometries.

`_loop_field_radial` / `off_axis_radial_field_cylinder_magnet` compute B_rho by modeling
the magnet as an equivalent solenoid (a uniformly magnetized cylinder of remanence Br is
magnetically equivalent to a surface current density Br/mu_0) and integrating the exact
single-loop Biot-Savart result (elliptic integrals, via `scipy.special.ellipk/ellipe`) over
the magnet's length. B_rho is **odd about the magnet's own axial center** by symmetry --
zero at the center, growing to an interior maximum somewhere between the center and a pole
face, then decaying to zero far away -- structurally the same shape as `q_shape` (odd,
zero at center, peaked lobes). `_peak_winding_averaged_coupling` finds that peak with a
coarse grid scan (simple and robust, consistent with this model's overall rigor; cheap
enough -- done once per distinct coil geometry at build time, cached and reused across the
identical coils of one design, not per simulation step -- not to matter for the optimizer's
runtime), and `build_coil_station` uses **both** the peak value (for `k_a`) **and** its
location (for `x_c`) -- x_c is no longer an independent heuristic, it's wherever that
field profile actually peaks. Crucially, the field it scans is **winding-averaged** over the
whole (r, z) coil cross-section (`winding_averaged_force_per_amp`, the same kernel the FEM
reference backend uses), not sampled at a single mean-radius point -- the single point
over-stated `k_a` by 25-75% for coils whose length approaches the coupling scale. `on_axis_field_cylinder_magnet` (the simpler on-axis-only
formula) is kept alongside as a cross-check anchor -- `off_axis_field_cylinder_magnet`
(the B_z off-axis calculation, still available, just not what k_a uses) is checked against
it in the rho->0 limit in tests.

**Remaining limits:** the equivalent-surface-current model assumes an ideal, uniformly
magnetized cylinder (real sintered magnets have manufacturing tolerances and can't
actually be magnetized perfectly uniformly); this is still not a substitute for a real
magnetic simulation or bench measurement, just a materially better estimate than the B_z
version it replaced.

None of this replaces real calibration once hardware exists (the whole project's stated
approach, repeated at every layer). It exists so the optimizer has *something* physically
grounded to trade off, rather than nothing.

### 1.1 A numerical-stability fix this model needed to be trustworthy

`linear_plant.coil_current_step()` originally modeled the "rl" current loop as a simple
two-state (full-on/full-off) bang-bang controller. That's a reasonable cartoon of a real
current-mode driver *most* of the time, but it broke down for a real region of this
optimizer's own search space: with few turns and a thick winding, coil resistance can be
very low (fractions of a milliohm); combined with a high bus voltage, the bang-bang
controller's implied steady-state current (`V/R`) can be **orders of magnitude** above the
actual target -- so a full-rail "on" period overshoots massively before the next decision
point can react. This was caught during development by a dt-refinement check: a design's
reported exit speed kept **dropping as dt shrank** (109 -> 74 -> 45 -> 12 -> 0.16 m/s)
instead of converging -- the classic signature of an integration relying on discretization
error rather than resolving real physics. No amount of sub-stepping fixed it, because the
failure wasn't an accuracy problem, it was the control law's target being wrong by
thousands of times.

The fix: `coil_current_step` now solves for the **exact constant voltage** that would
drive current from where it is to the target over exactly one step (inverting
`rl_current_step`'s own closed form), then clamps that to what the rail can actually
supply. When the exact voltage is achievable, this reaches the target precisely, at any
dt, with no overshoot possible by construction; when it isn't, applying the rail limit for
the whole step is the correctly rail-limited (not overshooting) response -- modeling an
idealized current-mode PWM loop, which is both more physically accurate (a real one is
tuned to converge smoothly, not bang-bang) and immune to this failure mode regardless of
how extreme R, L, or bus voltage get. Re-checked against the same design that exposed the
bug: now converges to ~4.15-4.19 m/s at every dt from 2e-4 down to 5e-6.

**Why this matters for trusting any result from this optimizer:** the search explores
turns/dimensions/voltage combinations broadly enough that it *will* find low-resistance,
high-voltage corners of the space. `optimize()`'s existing high-fidelity re-verification
step (see section 4) is what originally caught this -- if a reported result and its
re-verified speed disagree by more than ~10-20%, that's worth treating as a red flag and
investigating rather than trusting, the same way this one was found.

### 1.2 A second dt-stability fix: dead-reckoning couldn't see its own pump coming

Even after 1.1's fix, high-`i_max_a` designs could still fail catastrophically at coarse
`dt` -- not by overshooting current, but by the slug reversing direction entirely after the
very first coil, flying backward out of the tube. Root cause was in `linear_supervisor.py`,
not the electrical model: `StepperSupervisor._run_step` schedules a coil's approach-pump
pulse to cut off just *before* the slug reaches that coil's center (`t1 = t_arrival -
phase_advance_s`), using `LinearStepperEstimator.time_to_reach()` -- pure constant-velocity
extrapolation from the *previous* gate's measured speed. That's fine when the upcoming pump
barely changes velocity, but breaks down exactly when it's supposed to: a strong pump
measurably accelerates the slug *during* the same approach it's trying to predict the timing
of, so the slug reaches the coil's center sooner than a constant-velocity guess assumes. The
cutoff, scheduled from the too-late estimate, doesn't arrive until after the crossing --
`q_shape`'s sign has already flipped by then, so the same "attract" current that was pulling
the slug forward now pulls it backward, decelerating or (at high enough current) fully
reversing it.

The fix (`StepperSupervisor._predict_arrival`) corrects the dead-reckoning with a SUVAT
constant-acceleration estimate instead of a constant-velocity one: `v_eff = (v0 + v1) / 2`,
where `v1` comes from energy conservation on the energy this pulse is actually about to
deliver (`v1 = sqrt(v0^2 + 2*dE/mass)`). One subtlety mattered: `dE` here must be the energy
*actually deliverable* at the `i_max`-clamped `i_peak`, not the raw commanded `dE_cmd` (which
can be enormous when `v_tgt` asks for far more speed than the design can reach) -- using the
raw value overcorrected the other way, predicting arrival too *early* and firing the
departure-side repel kick before the slug had even reached the coil. Both the pump's own
cutoff and the departure kick it schedules now share this corrected estimate. Re-verified
against the design that exposed it: all 8 gates now fire, with closely-converging gate
speeds, across `dt` from `2e-5` to `1e-3` (previously only the entry gate fired at any
`dt` >= `5e-4`, with the slug rocketing backward out of the tube afterward).

### 1.3 A third and fourth attempt, both wrong, before the actual root cause

The deliverable-energy fix above wasn't the end of it -- two more attempts at patching
`_predict_arrival` followed, and **both of them were themselves bugs**, not fixes, each
caught by re-verifying every previously-fixed design after making the change (a discipline
this section exists to argue for doing every time, not just when something looks wrong).

**Attempt 3 (wrong): cap the assumed energy at the slug's current kinetic energy.** A
fresh search turned up a design -- light slug, 20 coils, high current -- whose reported
79.5 m/s collapsed to 4.3 m/s under dt-refinement, traced to `_predict_arrival` assuming
16x the slug's own kinetic energy would be delivered in one lobe pass (implausible, since
`K_pump`'s calibration assumes a full lobe-spanning pulse that the resulting short `T_p`
couldn't actually sustain). The fix applied at the time capped the correction's energy at
the slug's own current KE. It re-verified clean against the two designs known at the time.
**It was still wrong**: it also throttled the single most common and legitimate case in
this whole model -- a slug accelerating from near rest, where a strong first pump is
*supposed* to inject many multiples of the slug's (near-zero) kinetic energy. This surfaced
as an unexplained cliff in a sensitivity sweep (`rcos` collapsing to ~1.2 m/s at high
current while `trapezoid`/`square` kept climbing on the SAME baseline design) -- caught only
because a direct question ("why would raising current make it worse -- isn't that a bug?")
prompted tracing the actual trajectory instead of accepting the sweep at face value.

**Attempt 4 (also wrong): throttle by electrical feasibility instead of kinetic energy.**
The right-sounding fix: iterate the correction, checking whether the resulting pump window
`T_p` leaves the coil's own `L/R` time constant enough time to actually reach the assumed
current (discounting by `1 - exp(-T_p/tau_elec)` when it doesn't). This fixed the `rcos`
cliff AND both earlier designs -- and **broke the project's actual best-known design**
(~98 m/s), collapsing it to ~6 m/s. Re-verifying every previously-fixed design after a
"fix" is what caught this one; without that check it would have shipped.

**The actual root cause: one shared estimate was being asked to be safe in two opposite
directions at once.** `_predict_arrival`'s corrected arrival time feeds TWO different
decisions with OPPOSITE failure directions:
- the approach pump's own cutoff (`_run_step`) must not fire *late* -- lingering past
  center flips `q_shape`'s sign and turns the same attract current into a brake (the
  original 1.1/1.2 bug);
- the departure-repel kick and end-of-travel's kick (`on_gate`'s scheduling) must not fire
  *early* -- firing while genuinely still approaching turns "repel" into a brake instead
  (1.2's second symptom, 1.3, and 1.4, all really the same failure wearing different hats).

Every attempt so far tried to find ONE correction accurate enough to be safe both ways.
There isn't one: any correction aggressive enough to reliably beat the late-cutoff failure
is, by the same aggressiveness, liable to occasionally predict too early for the departure
kick, in some corner of an 11-dimensional design space. **The actual fix is architectural,
not numerical**: stop sharing the estimate. `_run_step`'s own cutoff keeps the full
accel-corrected `_predict_arrival` (aggressive, early-biased -- exactly what the approach
pump needs). `on_gate`'s departure-kick and end-of-travel scheduling now use PLAIN, naive
constant-velocity dead reckoning (`est.time_to_reach`) instead -- which systematically
predicts arrival LATE (the whole reason the approach-pump correction was needed in the
first place), the safe direction for a kick that must never fire before the slug truly
arrives. The cost is a little lost repel-assist (the kick fires somewhat after the ideal
instant); the benefit is that it can no longer fire while still approaching, in any design.

Re-verified against all four designs this investigation touched, at `dt` from `2e-4` down
to `5e-6`: the original 1.1/1.2 design (~16.3-16.7 m/s), the attempt-3 design (~76.3-76.9
m/s), the current best design (~98.4-98.9 m/s, matching its pre-regression numbers), and
the `rcos`-cliff design (~66-72 m/s, no more cliff) all converge cleanly.

**The general lesson:** when a single quantity feeds two decisions, check whether they
actually need the same accuracy/direction of error before reusing it -- "more accurate" is
not automatically "safer" if the two consumers are wrong in opposite directions. The
`optimize()` high-fidelity re-verification step (section 4) and re-checking every
previously-fixed design after every subsequent change are what caught all four of these;
treat any large gap between `search_reported` and the re-verified `speed` in an
`optimize_result*.json` as a prompt to dt-refine that specific design before trusting it.

## 2. The eleven knobs

| Knob | Meaning | Default bounds |
|---|---|---|
| `bus_voltage_v` | driver supply voltage | 3 - 400 V |
| `driver_bipolar` | single half-bridge (False) vs. H-bridge (True) | boolean |
| `pump_envelope` | current shape when a station's PM branch fires: `rcos` (smooth force, 0.5x avg current) / `trapezoid` (0.8x) / `square` (1.0x, unsmoothed) | categorical |
| `n_coils` | number of stations | 2 - 30 |
| `turns` | per coil (all coils share one design -- see section 4) | 10 - 1500 |
| `coil_length_m` | axial winding length (also sets pitch -- coils are packed edge-to-edge) | 0.005 - 0.08 m |
| `radial_thickness_m` | winding's radial build | 0.002 - 0.04 m |
| `magnet_radius_m` | slug magnet radius | 0.002 - 0.025 m |
| `magnet_length_m` | slug magnet length | 0.005 - 0.08 m |
| `remanence_t` | magnet material grade, ferrite (~0.3T) to N52 NdFeB (~1.42T) | 0.3 - 1.42 T |
| `i_max_a` | driver current rating | 1 - 150 A |

Every bound is a placeholder for **your** actual constraints (driver rating, available
space, magnet grades on hand, budget) -- override them via `Bounds(...)` in Python or the
`--max-tube-length-m` CLI flag (more flags are easy to add the same way; only the tube
length constraint has one so far). Self-heating within one run IS modeled now (section 1
above), but there is still no duty-cycle (repeated runs back-to-back) or cost model: a
design that "wins" here might still cook itself under continuous/repeated use, or cost far
more than a slightly-slower alternative. Treat the output as a starting point for a real
design, not a final answer.

**`driver_bipolar` is a bigger lever than it might look.** Repel-pumping (`docs/
DESIGN_LINEAR.md`'s departure-side thrust) only works with an H-bridge -- a single
half-bridge cannot source negative current at all, full stop, regardless of what the
control law wants. In one measured case this alone was a 2.2x speed difference at the
same everything else. If the optimizer converges on `driver_bipolar=False`, that means the
extra H-bridge complexity genuinely isn't worth it for the rest of that design; if it
converges on `True`, that's a real, physically load-bearing recommendation, not noise.

## 3. What "speed" means here

The objective is the slug's velocity as it clears the last gate, with the supervisor's
velocity governor (`target_velocity_m_s`) effectively disabled (set far above anything
achievable) -- see `docs/DESIGN_LINEAR.md` section on "governed vs. true ceiling". This is
a pure speed-maximization search, not a tracking or efficiency one. A design that FAULTs
(bootstrap never got a gate response -- too weak to move the slug at all) or clears zero
gates scores 0.0, same as one that violates the tube-length budget -- both push the search
away from that region rather than crashing it.

## 4. Simplifications in the search itself

- **All coils share one winding design.** Turns/dimensions are one shared value across
  every station, not optimized per-coil independently -- far more tractable, and more
  realistic for a manufacturable build (winding N different custom coils is a much bigger
  ask than winding N identical ones).
- **Coils are packed edge-to-edge** (pitch = `coil_length_m`, no gap) -- ignores bobbin/
  former wall thickness and mounting clearance a real build would need.
- **Bore radius is derived, not free**: `magnet_radius_m + 1.5mm` fixed clearance -- the
  coil bore has to fit around the slug, so it isn't really an independent design choice.
- **The search runs `current_loop="rl"` at a coarser `dt`/shorter bootstrap timeout than
  the supervisor's defaults**, specifically so resistance/inductance differences between
  candidates are actually visible (`"ideal"` mode ignores R and L entirely -- see
  `optimize_design.py`'s module docstring for why that would make the whole point of
  exposing turns/dimensions moot) while keeping thousands of evaluations tractable. The
  best design found is re-simulated once at high fidelity (`dt=2e-5`, full bootstrap
  patience) before being reported, so the final number you see isn't itself coarse.

## 5. Running it

```powershell
emac-optimize --maxiter 25 --popsize 15
```

Takes a few minutes at the defaults (roughly `maxiter * popsize * 11` worst-case
evaluations, each a short closed-loop simulation). Increase `--maxiter`/`--popsize` for a
more thorough search; `--workers N` parallelizes across processes. `--dt` and `--t-end`
control the search-phase simulation fidelity/duration (the final report is always
re-verified at high fidelity regardless).

```python
from emac_sim.optimize_design import optimize, Bounds

bounds = Bounds(bus_voltage_v=(3.0, 60.0), i_max_a=(1.0, 30.0))   # e.g. cap to what you can actually source
knobs, speed, result = optimize(bounds=bounds, maxiter=25, popsize=15)
```

## 6. Sensitivity and interaction analysis (`design_sensitivity.py`)

`optimize()` finds one best point. It doesn't say *why* it's best, or how each knob's effect
depends on the others -- questions like "does more current help more with an H-bridge or a
single half-bridge" need the relationship between knobs, not just an optimum.
`design_sensitivity.py` maps that out locally around a baseline design (by default,
whatever `optimize()` found):

- **`sweep_knob(knob, baseline, ...)`** -- one-at-a-time (OAT): vary one knob across its
  bounds (or its fixed options, for the two categorical knobs), holding everything else at
  the baseline's value. Returns a main-effect curve.
- **`full_sensitivity_report(baseline, ...)`** -- runs `sweep_knob` for every knob in
  `ALL_KNOBS`, as one dict.
- **`interaction_sweep(knob_a, knob_b, baseline, ...)`** -- varies two knobs across a grid
  simultaneously, holding the rest fixed. This is what an OAT sweep of either knob alone
  *can't* show: whether one knob's effect depends on the other's level.

All results are plain JSON-serializable dicts (`{"value": ..., "speed": ...}` points for
sweeps, `{"values_a", "values_b", "grid"}` for interactions) -- cacheable, plottable,
reusable without re-simulating.

**Worked example -- the H-bridge/current question.** `interaction_sweep("i_max_a",
"driver_bipolar", baseline)` on the reference 8-coil design: under a single half-bridge,
sweeping `i_max_a` from 1 A to 150 A moves exit speed from ~1.0 to only ~5-6.5 m/s (flat
past ~25 A); under an H-bridge, the same sweep climbs from ~2.0 to ~21.0 m/s, nearly
monotonically. The reason is architectural, not incidental: only a bipolar driver can run
the departure-side repel-pump (docs/DESIGN_LINEAR.md) alongside the approach-side attract
pump, so an H-bridge's extra current has two lobes to spend on, a half-bridge's has one and
saturates fast. **Current is an H-bridge knob first, and a half-bridge knob a distant
second.**

Other findings from the same baseline's full sensitivity report worth knowing before
trusting a design: `turns` collapses past ~260 turns at a fixed 48V bus (resistance scales
roughly `turns^2`, so the design becomes current-starved), and `radial_thickness_m` /
`magnet_radius_m` are both monotonically *worse* as they grow (a larger winding or magnet
pushes the coil's mean radius farther from the slug, weakening `k_a` faster than the extra
size helps). All these are **local** relationships around one baseline, not global claims --
a knob that looks flat here might matter a great deal at a different point in the design
space. See `tests/test_design_sensitivity.py` for the module's contract tests, and
`tests/test_linear_supervisor.py` for the arrival-prediction fix (section 1.2)'s regression
test.
