# `emac` — linear one-way stepper: design addendum

Companion to `docs/DESIGN.md`, whose section 6 ("Scaling to the coil ring") describes
generalizing the single-coil pendulum to an N-magnet **ring** (a motor). This addendum
covers the other natural generalization: an N-coil **finite line** instead of a wrapping
ring -- a slug with **no iron, just a permanent magnet**, sliding in a tube wound with
**air-core** coils, pulled forward by coils that fire in sequence ahead of it, sensed by
photogates positioned between adjacent coils. Concretely, the reference build is
**5 coils + 5 gates**. Force is purely from the PM branch (linear, signed) -- the model
keeps a reluctance term available too (see section 2.1), off by default, for anyone who
wants to model a hybrid or iron-only slug instead. Section 3 covers the coil's own
electrical dynamics (inductance), now genuinely simulated rather than assumed ideal.

This is a **continuous one-way stepper** (coilgun / staged reluctance launcher), not an
oscillator: there is no target amplitude, no bounded energy well, no braking-toward-center
objective. That difference is why it gets its own estimator/supervisor rather than reusing
`Tier1Estimator`/`EnergySupervisor` -- see section 1.

> Once you're asking "what hardware makes this fastest" rather than "how do I control
> fixed hardware", see `docs/DESIGN_OPTIMIZER.md`: a search over driver voltage, coil
> turns/dimensions, current waveform, single-ended vs. H-bridge switching, coil count, and
> slug/magnet properties, built on a physical model connecting turns and dimensions to
> resistance/inductance/thrust-constant.

## 0. Implementation status

Implemented in `tools/python/emac_sim/linear_plant.py`, `linear_estimator.py`,
`linear_supervisor.py`, `linear_sim.py`, plus `linear_cli.py` for text/plot reporting.
Selected via `[sim] kind = "linear_stepper"` in a TOML config (see
`examples/configs/linear_stepper_5coil.toml`); `emac-sim --config ...` dispatches to it
transparently -- the same command that runs the pendulum. `emac-visual`'s interactive
tube-canvas animation is **not yet built** (see section 6). Coil electrical dynamics
(inductance) are modeled but off by default -- see section 2.3, `[driver] current_loop`.

## 1. What's shared with the pendulum, and what isn't

| Primitive | Status |
|---|---|
| `q_shape(u, half_width)` -- odd, zero-at-center coupling lobe | **Reused verbatim, for BOTH force terms below.** A coil's axial pull on a coaxial ferromagnetic/PM slug is the same law as the pendulum's bottom-coil torque, just evaluated per-station. The reluctance and PM mechanisms could in principle have different spatial coupling profiles; there's no calibration data yet to justify separating them, so both share one `q_shape` anchor per coil. |
| `f_current(i, coil)` -- attract-only quadratic-with-saturation (reluctance branch) | **Reused verbatim**, unchanged signature. `CoilStation` exposes `.Cmag`/`.i_sat` so it satisfies `f_current`'s duck-typed contract as-is. `Cmag` defaults to 0.0 (no iron in the current slug) but the term stays available for a hybrid/iron-only slug. |
| `f_current_pm(i, k_a)` -- linear and SIGNED (PM branch, `docs/DESIGN.md` 3.3) | **New, added to `plant.py` as a shared primitive** (not linear-stepper-specific) since the pendulum's own design doc already anticipates a PM-bob branch. With `Cmag=0.0`, this is the ONLY force term. `k_a` is a placeholder pending real calibration. |
| `PulseCmd` / `current_at(t, cmd)` -- time-windowed raised-cosine current envelope | **Reused verbatim** from `supervisor.py`. Already coordinate-free; now a TARGET for the current loop under `"rl"` rather than the literal current (section 2.3). |
| `_q_window_integral` (pump-charge-per-A^2 integral) | Lightly generalized to take a bare half-width float instead of a `PendulumParams`, so each coil station can use its own `x_c`. |
| Damped velocity-Verlet mechanical integration | Same pattern, separate implementation in `linear_plant.step()`. Originally semi-implicit Euler (first-order); upgraded to a kick-drift-kick velocity-Verlet core with exact half-step exponential damping (second-order for the conservative part) -- see `docs/PHYSICS_ENGINE_ANALYSIS.md`. |
| `numerics.hermite_event_fraction` -- cubic Hermite event-crossing interpolation | **New, shared primitive** (`numerics.py`, not linear-stepper-specific). Replaced constant-velocity linear interpolation for BOTH the pendulum's bottom crossing and the linear stepper's gate crossings -- uses both endpoint velocities the integrator already computed, exact for a constant-acceleration segment. See `docs/PHYSICS_ENGINE_ANALYSIS.md`. |
| `rl_current_step(i, v, r, l, dt)` -- exact RL circuit update | **New, added to `plant.py` as a shared primitive.** Coordinate-agnostic (just current/voltage/R/L), so either model could use it once it needs real electrical dynamics; only the linear stepper does so far (section 2.3). |
| Estimator (decaying-sinusoid vs. constant-velocity coast) | **Not shared.** A slug with no restoring force has nothing to decay toward; `LinearStepperEstimator` dead-reckons by coasting at the last measured velocity instead. |
| Supervisor (energy-shaping vs. forward commutation) | **Not shared.** There's no "pump vs. brake, auto-selected by sign of error" law here -- just an ordered sequence of stations to pump through, plus a startup bootstrap. |
| Simulator orchestration shape | Reused as a *pattern* (true plant -> detect event -> estimator -> supervisor -> apply current), not a shared base class -- the event predicate (ordered gate-index advance vs. bottom sign-change) and per-event contracts differ enough that forcing a common `Simulator` base now would mean inventing an abstraction for two call sites. |

The one true shared seam is **config + CLI dispatch**: `config.parse_config()` reads
`[sim] kind` (`"pendulum"`, the default so every existing config is unaffected, or
`"linear_stepper"`) and returns the matching config type; `cli.main()` dispatches on that
type to either the existing pendulum code path or `linear_cli.run()`. Same command,
same TOML-in/report-out mental model, two cooperating backends built from shared
primitives -- the same "construction-time strategy swap" philosophy `DESIGN.md` states
for pendulum-vs-ring, applied here to pendulum-vs-linear-stepper.

## 2. Physics

```
m * x_ddot = -c_visc * x_dot
             + sum_k[ q_shape(x - x_coil_k, x_c_k) * (f_current(i_k) + f_current_pm(i_k, k_a_k)) ]
```

The direct degenerate limit of the pendulum ODE: drop the `-m*g*L*sin(theta)` gravity
term (the tube is assumed horizontal -- gravity is not the restoring mechanism here),
replace angular inertia with mass, and sum the same per-coil odd lobe across N stations
instead of evaluating one lobe at the bottom. At most one station is normally energized
at a time in this implementation (sequential-only; no blended cross-coil handoff yet --
see section 6).

### 2.1 Pure permanent-magnet slug (no iron), air-core coils

The current assumption: the slug has **no iron at all, only a magnet** -- not a
ferromagnetic body with a magnet added, but a slug whose entire coupling to the coils is
via that magnet's own field. Combined with **air-core** coils (wound directly around the
tube, no iron core of their own -- so no coil-side B-H saturation either), this puts the
actuator cleanly in the PM branch `docs/DESIGN.md`'s top-of-doc fork already describes
for the pendulum ("Is the bob a permanent magnet or soft iron?"):

- **PM term** (`f_current_pm`): the magnet's fixed field interacts with the coil's field
  linearly in current, and -- unlike a reluctance actuator -- it's **signed**: positive
  current attracts, negative current *repels*. With no iron, this is the ONLY force term.
- **Reluctance term** (`f_current`): kept in the model (`CoilStation.Cmag`,
  defaulting to **0.0**) for anyone who wants to model a hybrid (ferromagnetic + embedded
  magnet) or pure-iron slug instead -- setting `Cmag` nonzero adds it back in, unchanged.
  It is NOT currently part of this build's assumption.

`net_force()` always computes both terms and adds them; `Cmag=0.0` is simply what makes
the reluctance term vanish for every current, not a special code path. `k_a` (the PM
gain) is a placeholder pending real calibration, like every other constant in this model.

**Repel-pumping is now built.** `PulseCmd` carries an explicit `polarity` field
(`"attract"` default, or `"repel"`, mirroring `docs/DESIGN.md`'s `polarity_t {ATTRACT,
REPEL, REGEN}`), and `supervisor.current_at()` sign-flips the envelope for `"repel"`.
Every PM-branch station (`k_lin > 0`) gets **both** an approach-side attract pump
(`StepperSupervisor._run_step`, unchanged in spirit from before) **and** a departure-side
repel kick (`StepperSupervisor._fire_departure`, scheduled by `on_gate` once dead-
reckoning says the station's center has been passed, fired from `tick()`): `q_shape < 0`
on departure, and a negative (repel) current there makes the force product positive --
forward thrust, the same sign convention that makes attract work on approach. Reluctance-
only stations (`k_quad > 0`, `k_lin == 0`) still only ever get the approach pump --
attraction can't be signed regardless of current direction, so there's nothing for a
repel kick to exploit there.

**Actually driving a negative current needs `driver_bipolar=True`** (`LinearActuatorParams`,
default `False`): a single half-bridge can only ever source `i >= 0`, matching the
original reference driver, so repel-pumping has no electrical effect under it in `"rl"`
mode (a negative target just decays toward zero -- see section 2.3). Under `"ideal"` mode
there's no hardware model at all, so a signed target is reached instantly regardless of
`driver_bipolar`. In one measured case, `driver_bipolar=True` alone was worth a ~2-3x
speed difference at otherwise-identical everything else (see `docs/DESIGN_OPTIMIZER.md`)
-- exactly the H-bridge-vs-half-bridge tradeoff this section originally flagged as
unbuilt.

**The PM-branch pump-sizing approximation flagged here previously is fixed.**
`_station_k_pump()` now returns `(k_quad, k_lin)` separately -- `k_quad` for the
reluctance branch's `dE ~ i²` scaling (unchanged), `k_lin` for the PM branch's `dE ~ i`
scaling (linear, as the physics actually requires, scaled by whichever current envelope
the PM branch uses -- see `supervisor.envelope_average_linear`). `_i_peak_for_energy`
inverts the combined quadratic `dE = k_quad*i² + k_lin*i` exactly for either pure branch
(and sensibly, if approximately, for a hybrid with both nonzero). The old
quadratic-only formula was a real mis-sizing for a pure-PM slug, not just an
approximation to tighten -- it's no longer in use.

### 2.2 Gate placement (5 coils, 5 gates)

```
gate0            coil0   gate1   coil1   gate2   coil2   gate3   coil3   gate4   coil4
  |----pitch/2-----|-------|-------|-------|-------|-------|-------|-------|-------|
```

One entry gate before coil 0, then one gate at the midpoint of each adjacent coil pair:
`n_coils` gates for `n_coils` coils (`linear_plant.default_gate_stations()`). On a
*finite* line (unlike the ring, which has no boundary and tiles cleanly), one gate must
sit at a boundary rather than strictly between two coils. The entry boundary was chosen
over an exit boundary because the harder, more novel problem here is bootstrap detection
(section 4): an entry gate gives the earliest possible confirmation that the bootstrap
kick produced real motion. The cost: end-of-travel (clearing the last coil) must be
*inferred* by dead-reckoning forward from the last gate, not sensed directly (section 5).

**Consequence for the supervisor:** because gate[j] always precedes coil[j] in this
layout, "which coil to target" is just "the coil with the same index as the gate that
just fired" -- no separate gate-to-coil lookup table is needed. A different placement
scheme would need one.

### 2.3 Electrical dynamics: coil inductance

Every prior section describes the MECHANICAL model. Until now, current was treated as
instantaneously commandable -- `supervisor.current_at()`'s raised-cosine profile *was*
the actual coil current, with no electrical dynamics in between. That's no longer the
only option: `LinearActuatorParams.current_loop` selects between it (`"ideal"`, still the
default) and a real per-coil RL circuit (`"rl"`).

**The circuit.** Each `CoilStation` now carries `resistance_ohm` and `inductance_h`
(previously present in config but never wired to the physics). Air-core windings have a
fixed, current-independent L (no B-H saturation to shrink it, unlike an iron-core coil),
so a plain first-order RL model is a good fit:

```
L * di/dt = v_applied - i * R
```

`plant.rl_current_step(i, v_applied, r, l, dt)` is the EXACT closed-form update for this
(assuming piecewise-constant `v_applied` over one tick) -- unconditionally stable
regardless of how `dt` compares to the L/R time constant `tau = L/R`, unlike explicit
Euler. It's a shared, geometry-agnostic primitive in `plant.py`, not linear-stepper-
specific, in case the pendulum ever needs the same treatment.

**Driving it.** Real current-mode drivers regulate current with a comparator/PWM loop,
not an open-loop voltage command. `linear_plant.coil_current_step()` models an idealized
version of that loop: it solves for the exact CONSTANT voltage that would drive current
from where it actually is to the target over exactly one control tick (inverting
`rl_current_step`'s own closed form), then clamps that to what the driver rail can
actually supply -- `[0, bus_voltage_v]` for a single half-bridge (`driver_bipolar=False`,
the reference driver, `i >= 0` only) or `[-bus_voltage_v, bus_voltage_v]` for an H-bridge
(`driver_bipolar=True`). When the exact voltage is within the rail this reaches the
target exactly after one step, at any `dt`, with no overshoot possible by construction;
when it isn't, applying the rail limit for the whole step is the correctly rail-limited
response. `driver_bipolar=False` still means a negative target can't actually be reached
(same as before) -- repel-pumping (section 2.1) needs `driver_bipolar=True` to take
effect under `"rl"`.

This replaced an earlier two-state (full-on/full-off) bang-bang model, which chattered
around the target under normal conditions but broke down outright for a real corner of
this project's own design-optimizer search space: a low-resistance coil (few turns, thick
winding) under a high bus voltage can have a bang-bang steady-state current (`V/R`)
orders of magnitude above the intended target, discovered via a dt-refinement check where
reported exit speed kept *dropping* as `dt` shrank instead of converging. See
`docs/DESIGN_OPTIMIZER.md` section 1.1 for the full story -- no amount of sub-stepping
fixed it, because the failure was the control law's steady-state target being wrong by
orders of magnitude, not an integration-accuracy problem.

**What this reveals, with the reference L/R (1.2 ohm / 4 mH -> tau ~ 3.3 ms):** the
coil's own electrical response is much FASTER than the commutation timescale (pump
windows tens of ms wide), so under `"rl"` the actual current tracks the idealized
raised-cosine target closely with only a small lag, never overshooting it (by
construction, now that the driver targets the exact tracking voltage rather than bang-
banging toward it). The one thing that's true regardless of relative timescales: current
cannot jump to a target in zero time from a cold start (see
`tests/test_linear_plant.py::test_rl_current_lags_an_ideal_instantaneous_target` and
`test_plant.py`'s `rl_current_step` tests). If you want `"rl"` to visibly matter more than
a small ripple -- e.g. to see a real "hard cut isn't instant" decay tail dominate behavior
-- either slow the commutation down, speed up the current target's ramp (shorter pump
windows), or increase `inductance_h` relative to `resistance_ohm`.

**What "ideal" is still for:** fast, simple sweeps where electrical dynamics aren't the
question -- e.g. everything in this document's earlier sections, and the CLI examples
run so far, used it. Switch to `"rl"` specifically when you want to see the effect of
inductance itself.

### 2.4 Thermal dynamics: winding self-heating

Everything above (and every result reported by `docs/DESIGN_OPTIMIZER.md`'s search) treats
`CoilStation.resistance_ohm` as a fixed constant. It isn't, physically: copper resistivity
rises ~0.39%/C (`plant.COPPER_TEMP_COEFF_PER_C`, already used at BUILD time by
`coil_design.wind_coil`'s `temperature_c` parameter), and a coil dissipates real power
(`i^2*R`) every time it's energized. Left unmodeled, an optimizer is free to recommend a
design that only looks fast because it never pays a heating penalty -- more turns, more
current, and a bus voltage far above what a continuously-operating winding could actually
sustain, with nothing in the simulation to say otherwise.

`LinearActuatorParams.thermal_model` (default `False`, reproducing the fixed-resistance
model exactly) turns on a genuine one-node thermal model, per coil:

```
C_th * dT/dt = i^2*R(T) - (T - T_ambient) / R_th
R(T) = resistance_ohm * (1 + COPPER_TEMP_COEFF_PER_C * (T - ambient_temperature_c))
```

`plant.thermal_step` is the exact closed-form update for this (same first-order-linear-ODE
shape, and same reason, as `rl_current_step`'s RL circuit: unconditionally stable
regardless of how `dt` compares to the thermal time constant `C_th*R_th`, which is usually
much larger than the electrical `tau` but nothing guarantees that across an arbitrary
design search). `linear_plant.coil_temperature_step` wraps it per-coil;
`linear_plant.coil_resistance` computes `R(T)` and feeds it back into `coil_current_step`
(via its `resistance_ohm_override` parameter) so `"rl"` mode's electrical dynamics actually
see the hotter, higher-resistance coil, not just a number that gets computed and discarded.
Temperature is tracked under `"ideal"` current mode too (dissipation happens regardless of
how the current got there), it just doesn't feed back into anything under `"ideal"`, which
has no hardware model at all.

**`thermal_mass_j_per_k` is derived from the winding's own copper, not fabricated.**
`coil_design.wind_coil` computes it as (copper volume) x (copper density) x (specific
heat), where copper volume is the mean turn circumference times the winding's copper
cross-sectional area -- turns cancels out of that product entirely (see
`tests/test_coil_design.py::test_wind_coil_thermal_mass_matches_a_direct_copper_volume_calculation`).
This is a **lower bound**: it covers only the copper itself, not the bobbin, potting
compound, or frame around it, all of which add real thermal mass in a physical build.
**`thermal_resistance_k_per_w` has no such derivation** -- it depends on convection,
airflow, and mounting, none of which this model has any basis to estimate from turns and
dimensions alone -- so it stays a placeholder constant (`CoilStation.thermal_resistance_k_per_w`,
default 8 K/W) like every other uncalibrated value here, pending a real measurement.

**Wired into the search.** `optimize_design.py`'s `build_params` sets `thermal_model=True`
(and `ambient_temperature_c=20.0`, matching `coil_design.build_coil_station`'s own build-
time reference temperature) for every candidate the search evaluates -- see
`docs/DESIGN_OPTIMIZER.md`. A design that only looked fast because it never paid a heating
penalty no longer gets a free pass; `thermal_resistance_k_per_w`'s placeholder value
(8 K/W, uncalibrated -- see above) still bounds how much this can currently mean in
absolute terms, but the qualitative effect (a design too weak in R_th for its own
dissipation shows some real speed penalty within the run) is genuine.

## 3. Estimator: position, velocity, and the stall caveat

`LinearStepperEstimator` re-anchors position to each gate's known location and
dead-reckons forward by coasting at the last measured velocity (`v = w_eff / pulse_width`,
the linear analog of the pendulum's `theta_dot = dalpha / pulse_width`) -- no decaying
sinusoid, since there's no restoring term for one to decay toward.

Gate order must be strictly increasing. A one-way stepper has no direction ambiguity to
resolve, so there is no analog of the pendulum's alternating-parity trick: an
out-of-order, skipped, or repeated gate index is treated as an anomaly
(`STALL_SUSPECT`), not a signal to interpret.

This directly implements `DESIGN.md` section 6's explicit caveat: gates-between-magnets
give excellent direction + speed **at running speed** but **nothing at stall** (the
classic sensorless-BLDC low-speed problem). If the next expected gate is overdue by more
than `stall_factor` times the last inter-gate interval, status degrades from `TRACKING`
to `STALL_SUSPECT` rather than continuing to trust an increasingly stale dead-reckoned
guess.

## 4. Supervisor: forward commutation and startup bootstrap

Because `q_shape` is zero-and-odd at each coil's own center, "pump on approach, cut at
center" -- the pendulum's core trick -- applies unchanged at every station. Direction is
never in question (always forward), so the ring's `commTable[sector][dir]` collapses to
a plain ordered list of stations: after gate[j] fires, pump coil[j], cut at its predicted
center-crossing time (with a phase-advance lead to compensate coil L/R electrical rise
time, the linear analog of the ring's `theta_adv = beta * omega`); a PM-branch station
also gets a departure-side repel kick once dead-reckoning says its center has been passed
(section 2.1) -- reluctance-only stations remain attract-only in both directions, since
attraction can't be signed regardless of current.

**The center-crossing prediction is accel-corrected, not naive dead reckoning.**
`StepperSupervisor._predict_arrival` (called by both `_run_step`'s pump cutoff and the
departure-kick scheduling above) estimates the crossing time using a SUVAT constant-
acceleration average velocity, `(v0 + v1) / 2`, where `v1` comes from energy conservation
on the energy this pulse is actually about to deliver -- not `LinearStepperEstimator`'s
own plain constant-velocity `time_to_reach()`. A strong pump measurably accelerates the
slug *during* the very approach it's predicting the timing of, so a constant-velocity
guess systematically predicts arrival too late; the stale cutoff then leaves the coil
still energized once the slug has already crossed its center, where the same "attract"
current pulls backward instead (`q_shape`'s sign has flipped) -- enough to fully reverse
a high-thrust design at a coarse simulation `dt`. The correction's own energy assumption is
in turn capped at the slug's current kinetic energy (at most doubling it in one lobe pass)
-- an uncapped correction can still overshoot the other way for a light, high-current
design, predicting arrival implausibly early and firing the departure kick too soon. See
`docs/DESIGN_OPTIMIZER.md` sections 1.2 and 1.3 for both failure modes and why the
correction has to use the energy *actually deliverable* at the current-limited `i_peak`,
not the raw commanded energy.

**Startup is structurally easier than the pendulum's.** `DESIGN.md` section 4.6: a
soft-iron pendulum bob *always* rests at bottom-center (gravity puts it there), which is
exactly the coil's dead zone -- hence the mandatory mechanical offset or dedicated kicker
coil. A resting slug in the tube will similarly detent at whichever coil is nearest (the
same zero-force-at-center problem, via passive reluctance-seeking instead of gravity) --
**but** with 5 stations, firing any *non-nearest* coil has immediate nonzero coupling.
No extra hardware is required: `StepperSupervisor`'s bootstrap FSM fires station 0 for a
short forced pulse; if no gate response arrives within a timeout, it advances to station
1, then 2, and so on, never repeating a station consecutively. A resting slug can
coincide with at most **one** station's exact zero-force center, so this escapes any
single-coil detent within at most two attempts. Exhausting all 5 stations with no gate
response at all means `FAULT` (no slug in the tube, or it's jammed).

### 4.5 Optional: a pressurized tube removes the detent case entirely

The station-hunting bootstrap above exists because of one specific fragility: `q_shape(0,
x_c) == 0` exactly, so a slug resting **precisely** at a coil's own center feels zero
force from that coil at *any* current -- firing it does nothing, and you must fire a
*different* station instead. Pressurizing the tube behind the slug (a constant forward
force from a regulated gas/spring reservoir, independent of any coil) removes this case
outright: with a nonzero bias, the net force at that exact point is `pressure_bias_n`,
never zero, so the slug is never in true static equilibrium anywhere in the tube --
including sitting dead-center on a coil.

Modeled as `LinearActuatorParams.pressure_bias_n` (default `0.0`, exactly reproducing the
unpressurized model): a constant term added to `net_force()` regardless of coil currents.
This is deliberately the simplest reasonable model -- a regulated supply or a large
reservoir behind the slug is genuinely close to constant-force over the tube's short
travel. A small fixed-volume charge would instead follow an isothermal-expansion decay
(`P*V = const`, so force falls as the slug advances and the volume behind it grows); that
refinement isn't built, since a constant bias already captures the property that matters
here (guaranteed non-equilibrium) without added complexity.

**What this does and doesn't change:** with `pressure_bias_n > 0`, the slug is
guaranteed to accelerate forward from *any* starting position and *any* coil state --
including fully de-energized -- so gate crossings and forward progress no longer strictly
depend on the bootstrap FSM correctly identifying a non-degenerate station. The
station-hunting logic in `StepperSupervisor` is left in place regardless (it still gives
a controlled, faster start than pressure alone, and remains the load-bearing mechanism
when `pressure_bias_n == 0`), but `FAULT` becomes reachable only through a genuine fault
(no slug present, a jammed tube, or damping so large relative to the bias that terminal
velocity is negligible) rather than an unlucky rest position.

## 5. End of travel

Configurable via `[actuator] end_of_travel`:

- **`"coast"` (default):** after the last coil's pump-and-cut fires, stop commutating and
  let the slug exit -- except that a PM-branch last coil still fires its departure-repel
  kick first (section 2.1, `_end_of_travel`'s "coast" branch calls `_fire_departure`
  instead of just going silent): the same free extra thrust every earlier station gets,
  not a special case. Falls back to true silence only when there's no PM branch to repel
  with (a pure-reluctance build).
- **`"brake_hold"`:** mirrors the pendulum's "attract on departure" braking, fired after
  dead-reckoning says the slug has passed the last coil's center (there is no gate past
  it to sense this directly -- a consequence of the entry-gate placement in section 2).

## 6. Open questions / deliberately deferred

- **Interactive tube-canvas visualizer** (`emac-visual` support), mirroring the
  pendulum's animated HTML report -- not built yet; `emac-sim`'s static plots
  (`linear_cli.write_plots`) are the current reporting surface.
- **Per-station hardware variance:** the reference config gives every coil/gate identical
  properties except position; `LinearCoilConfig`/`LinearGateConfig` already support
  per-station overrides in TOML if a real build needs them.
- **Blended multi-coil handoff** across adjacent stations (to smooth cogging at
  handover, per `DESIGN.md` section 6's ring discussion) -- current implementation is
  strictly sequential, one active station at a time.
- **Damping feed-forward** in the velocity-shaping law: `EnergySupervisor` feeds forward
  the pendulum's per-swing damping loss; `StepperSupervisor` does not yet, since viscous
  loss over one station-to-station hop is small relative to the coil's applied impulse
  compared to the pendulum's many free periods. Worth revisiting if measured tracking
  error against `target_velocity_m_s` is larger than expected.
- **Gate placement alternative:** 4 internal gates + one exit gate (senses end-of-travel
  directly, at the cost of a blinder bootstrap window) was considered and rejected in
  favor of the entry-gate scheme above; revisit if bootstrap confirmation time turns out
  not to be the binding constraint in practice.
