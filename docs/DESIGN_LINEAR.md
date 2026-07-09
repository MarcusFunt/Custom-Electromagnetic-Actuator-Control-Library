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
| Semi-implicit Euler integration | Same pattern, separate 2-line implementation in `linear_plant.step()` -- not worth extracting a shared `integrators` module for two lines of code. |
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

**What this does and doesn't unlock.** The physics layer (`linear_plant.net_force`)
already computes the correct signed force for a negative current -- see
`tests/test_linear_plant.py`'s PM-branch tests. But `StepperSupervisor`'s commutation
logic and `supervisor.current_at()`'s raised-cosine shaping still only ever schedule
`i_peak >= 0` (unipolar, single-FET-style current sourcing, matching the reference
driver) -- so even though the WHOLE actuator is now bidirectional in principle, nothing
yet drives it that way. Exploiting repel/regen would need: (a) `PulseCmd`/`current_at` to
support a signed envelope or an explicit polarity field (mirroring `docs/DESIGN.md`'s
`polarity_t {ATTRACT, REPEL, REGEN}`), and (b) `StepperSupervisor`'s braking logic
reworked to use repel-on-approach instead of (or alongside) the current attract-on-
departure `brake_hold`. Neither is built -- flagged as a natural next step, not assumed.

**Known approximation:** `_station_k_pump()`'s calibration (used to size pump/brake pulse
current from a commanded energy) is derived assuming a pure quadratic (`F ∝ i²`)
reluctance response. With `Cmag=0.0` this assumption is simply wrong in form (the real
response is linear in `i`, not quadratic), so delivered energy for a commanded `dE_cmd`
will be off by more than in the old hybrid case -- worth revisiting (re-deriving
`_station_k_pump`-equivalent sizing for the PM branch specifically) if tracking accuracy
against `target_velocity_m_s` matters more than it has so far.

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
not an open-loop voltage command. `linear_plant.coil_current_step()` models the simplest
version of that: hysteretic bang-bang, full bus voltage while actual current is below
the target (still `current_at()`'s raised-cosine profile -- now a TARGET rather than the
literal current), zero (synchronous freewheel through the winding's own resistance) once
at or above it. This driver has **no negative rail**: it can only ever reach `i >= 0`,
matching the unipolar reference driver and `StepperSupervisor`'s current unipolar
commutation -- a negative target (repel; not currently commanded) would just decay toward
zero rather than actually go negative. A bipolar (H-bridge) driver would need its own
signed variant; not built (see 2.1's PM-branch discussion of what else repel/regen needs).

**What this reveals, with the reference L/R (1.2 ohm / 4 mH -> tau ~ 3.3 ms):** the
coil's own electrical response is much FASTER than the commutation timescale (pump
windows tens of ms wide), so under `"rl"` the actual current tracks the idealized
raised-cosine target closely, chattering slightly around it rather than lagging it
substantially -- a naive bang-bang controller with no hysteresis band can even briefly
*overshoot* a slowly-rising target, not just lag it. The one thing that's true regardless
of these relative timescales: current cannot jump to a target in zero time from a cold
start (see `tests/test_linear_plant.py::test_rl_current_lags_an_ideal_instantaneous_target`
and `test_plant.py`'s `rl_current_step` tests). If you want `"rl"` to visibly matter more
than a small ripple -- e.g. to see a real "hard cut isn't instant" decay tail dominate
behavior -- either slow the commutation down, speed up the current target's ramp
(shorter pump windows), or increase `inductance_h` relative to `resistance_ohm`.

**What "ideal" is still for:** fast, simple sweeps where electrical dynamics aren't the
question -- e.g. everything in this document's earlier sections, and the CLI examples
run so far, used it. Switch to `"rl"` specifically when you want to see the effect of
inductance itself.

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
center" -- the pendulum's core trick -- applies unchanged at every station. There's no
direction/polarity choice here (always forward, always attract-only), so the ring's
`commTable[sector][dir]` collapses to a plain ordered list of stations: after gate[j]
fires, pump coil[j], cut at its predicted center-crossing time (with a phase-advance lead
to compensate coil L/R electrical rise time, the linear analog of the ring's
`theta_adv = beta * omega`).

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

- **`"coast"` (default):** after the last coil's pump-and-cut fires, stop commutating;
  the slug exits the tube. Matches the coilgun/staged-launcher framing and needs no
  additional control logic.
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
