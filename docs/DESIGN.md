# `emac` — ElectroMagnetic Actuator Control library: design

A portable, **event-first** control library for a magnetic pendulum driven by a
single electromagnet at the bottom of its swing, sensed by a photogate — designed
so the *same* pipeline scales, unchanged, to an N-coil ring "motor" later.

**Repository status, July 2026:** this repo is currently a **Phase 0 host-only
Python simulator package**. It is installable for simulation and test work, but it
does not yet contain the portable C++ firmware core or hardware abstraction layer.
The next working milestone is Phase 1 hardware on the documented soft-iron +
ESP32-S3 target: capture input, unipolar coil power stage, timing-budget proof,
and sustained one-gate swing.

Core thesis, one sentence:

> **A sensor edge is the atomic unit of truth — not an angle.** Angle, velocity,
> energy, phase, and rotor sector are all *reconstructions* layered on a
> timestamped event stream.

That single choice is what lets a 1-gate pendulum, a 2-gate directional pendulum,
and an N-gate ring all be the same code with different strategy objects plugged in.

---

## 0. Two hardware decisions that change the physics

> **Chosen configuration for this build:** soft-iron (attract-only) bob on an
> **ESP32-S3** (float FPU, WiFi telemetry/config, dual-core). Consequences carried
> through the rest of the doc:
> - Power stage is a **single half-bridge** (low-side N-FET + synchronous freewheel) —
>   no H-bridge needed, since force is unidirectional.
> - **Braking is timing-only** (attract on departure); no repel, no regeneration.
> - **Start-from-rest is not free** — you must design in a deliberate mechanical
>   offset (see §4.6). Decide δ *before* you build the rig.
> - The coil's `f(i) = k_r·i²` map **drifts with temperature** (copper R ~+0.39 %/°C),
>   so R(T) tracking / warm recalibration is mandatory, not optional.
> - ESP32-S3 specifics: use the **MCPWM capture** submodule for sub-µs gate timestamps,
>   **LEDC or MCPWM** for coil PWM, pin the fast control loop to **core 1** and
>   WiFi/telemetry to **core 0**, stream **CBOR over UDP/WebSocket**. Use `float`.

Everything downstream forks on these. Decide them first.

1. **Is the bob a permanent magnet or soft iron?**
   - **Permanent-magnet bob (PM):** force is *bidirectional* and roughly linear in
     current, `F ∝ i`. You can attract **and** repel (reverse the H-bridge), so you
     can actively brake and even regenerate. Reversible ⇒ easier braking and startup.
   - **Soft-ferromagnetic bob (soft iron):** force is *attract-only* and roughly
     quadratic, `F ∝ i²`, and falls off much more steeply with gap. Reversing current
     does nothing. Energy is managed purely by **timing** the attractive pulse.
     Braking and start-from-rest are harder (see §4.6, §5.1).

2. **Target platform.** Default assumption in this doc: a portable **C++ core**
   (Arduino / ESP32 / STM32-class MCU) for the hard real-time loop, plus a **Python
   companion** for simulation, calibration fitting, and offline tuning. On an FPU part
   (Cortex-M4F/M7, ESP32, Teensy) use `float`; on M0/AVR use a `real_t` fixed-point
   typedef (§7.5).

---

## 1. The stack (seven narrow layers)

Mirrors the SimpleFOC / ODrive / VESC discipline (sensor → observer → controller →
modulator → driver), but with an **event pipeline** instead of a continuous encoder:

```
  ISR context        │        control-loop context
 ┌───────────┐  SPSC  ┌───────────┐   ┌──────────────┐   ┌─────────────┐
 │EventSource│─ring──►│ Estimator │──►│  Supervisor  │──►│  Current    │──► PowerStage ──► coil
 │ (capture) │Crossing│ θ,ω,E,φ,t*│ x̂ │ energy-shape │ q*│  Controller │duty  H-bridge /
 └───────────┘        └─────┬─────┘   └──────┬───────┘   └─────────────┘     N-phase
                            │  uses           │ uses
                     ┌──────┴─────┐    ┌──────┴──────┐
                     │ PlantModel │    │  ForceMap   │  (calibrated q(θ)·f(i),
                     │ ODE, E(x)  │    │ force+inverse│   flashed as a table)
                     └────────────┘    └─────────────┘
```

| Layer | In → Out | Owns |
|---|---|---|
| **IEventSource / Capture** | HW timer-capture on gate pins → `Crossing` | Sub-µs edge timestamp + pulse width. Integer only, no float, no malloc. |
| **EventQueue** | `Crossing` (ISR) → `Crossing` (loop) | Lock-free SPSC ring; decouples ISR from control tick. Needs real memory ordering (§7.1). |
| **IEstimator** | `Crossing`s + Δt → `State` | Fuse sparse events with dead-reckoning; energy; direction. |
| **PlantModel** | params → `dxdt`, `E(x)` | Pendulum/rotor ODE for predict + feed-forward. |
| **IForceMap** | (θ,i) → force, and inverse (θ,q)→i | Calibrated `q(θ)·f(i)`; PM (signed, ∝i) vs soft-iron (≥0, ∝i²). |
| **ISupervisor** | `State`, targets → force setpoint `q*` + window | Energy shaping (pendulum) / sector torque shaping (ring); safety envelope. |
| **ICurrentController** | `q*` → duty | force→current (ForceMap inverse)→duty; slew-limit for low vibration. |
| **IPowerStage** | duty → coil current | H-bridge / half-bridge / N-phase; PWM, dead-time, fault flags. |

Pendulum-vs-ring is a **construction-time strategy swap** (which concrete
`IEstimator` + `ISupervisor` you inject), not a fork of the control code.

**Three execution contexts:**

| Context | Rate | Does | Budget |
|---|---|---|---|
| ISR | per edge | latch capture register, compute pulse width, `queue.push()`. Integer only. | < 2 µs |
| Fast tick | 1–10 kHz (PWM-sync) | `est.update()` (drain queue + dead-reckon), `cc.update()` (force→current→duty) | ≤ 40 µs @10 kHz |
| Slow tick | 50–200 Hz | `sup.desiredForce()` (energy policy, mode FSM), telemetry. Transcendentals OK. | ~10 ms @100 Hz |

Use **hardware input-capture**, never `attachInterrupt`/`pulseIn`. A GPIO software
ISR carries 1–5 µs entry jitter; a timer-capture peripheral latches the counter *at
the edge* in hardware (STM32 84 MHz → 12 ns), independent of servicing latency. That
hardware timestamp is what makes pulse-width → speed trustworthy (§3).

---

## 2. Part A — reconstructing where the pendulum is, from one photogate

### 2.1 The event

```cpp
struct Crossing {
    uint16_t sensor_id;   // 0 for the single-gate pendulum; sector id for the ring
    uint32_t t_capture;   // HW timer ticks at the edge (NOT loop time)
    uint8_t  edge;        // 0 = falling (beam broken), 1 = rising (beam restored)
    uint32_t pulse_width; // on rising: t_rise − t_prev_falling (ticks). 0 if unknown
    uint8_t  seq;         // monotonic per-sensor counter; catches dropped ISRs
};
```

The bottom instant is the **midpoint** `t_mid = (t_fall + t_rise)/2` — θ = 0 there,
exactly. `pulse_width = Δt_block = t_rise − t_fall` is the single most valuable
number: it encodes **speed at the bottom**.

### 2.2 Speed from pulse width

While the beam is blocked the bob sweeps a fixed **effective angular width** Δα (as
seen from the pivot). Over that short block the speed is ~constant, so:

```
θ̇_bottom ≈ Δα / Δt_block          Δα ≈ (w_bob + a_aperture) / L
```

Example: `w_bob = 15 mm`, `a = 3 mm`, `L = 300 mm` → Δα ≈ 0.060 rad (3.4°). A 1 µs
timer against Δt_block ≈ 2 ms gives ~0.05 % speed resolution. Polling at 1 kHz would
give ~50 % — this is why hardware capture matters.

**Calibrate Δα once, empirically — do not compute it from CAD.** It absorbs bob
shape, beam divergence, mounting slop, and comparator threshold. Release the bob from
a protractor-measured angle θ₀; energy conservation gives the true bottom speed
`θ̇_true = √(2(g/L)(1−cos θ₀))`, and on the first pass `Δα = θ̇_true · Δt_block`.
Caveat: this ignores the quarter-period damping loss between release and the first
crossing — fractional error ≈ (π/2)/(2Q), so it is <0.1 % only for a high-Q rig
(Q ≳ 800). For a low-Q rig, correct for the known damping or use an electromagnet
hold-and-release instead of a hand release, and average several θ₀.

### 2.3 Energy & amplitude — the primary state

At the bottom all energy is kinetic (θ = 0), so **one pulse width gives total energy**:

```
E      = ½ m L² θ̇_bottom²                                  (measured directly)
θ_max  = arccos( 1 − L θ̇_bottom² / (2g) )     (small angle: θ_max ≈ θ̇_bottom / ω₀)
```

Track `E_k` and its per-swing change `ΔE_k = E_k − E_{k−1}`. Under free swing,
`ΔE_k < 0` **is the live damping loss** — the exact feed-forward the controller must
replace to hold amplitude (§4.2). Energy/amplitude is more robust than a reconstructed
angle; expose it as the controller's primary feedback.

### 2.4 Dead-reckoning between events

Integrate the plant ODE forward from the last anchor, re-anchoring at every bottom
crossing (the most observable point: θ = 0 exactly, θ̇ just measured):

```
θ̈ = −(g/L) sin θ − (b/mL²) θ̇ + q(θ) f(i)/(mL²)
```

- Use a **symplectic** integrator (semi-implicit Euler or velocity-Verlet), *not*
  explicit Euler — explicit Euler injects energy and corrupts a conservative oscillator.
- `h = 1–2 ms` gives 500–1000 steps/period at L = 0.3 m.
- Carry the full `sin θ`, not `≈ θ`: the period lengthens with amplitude
  (≈ +1.6 % at A = 0.5 rad, ≈ +6.6 % at A = 1.0 rad); if you linearize, dead-reckoned
  phase drifts.
- For θ_max ≲ 20° you can skip integration and evaluate a closed-form decaying sinusoid.

You re-anchor **twice per period**, so drift never exceeds one half-swing. The
dominant residual is *phase near the apex* — exactly where the magnet is off and the
controller is least sensitive. **That is why a single bottom gate is sufficient.**

### 2.5 The observer: ship Tier 1, keep 2–3 as upgrades

**Tier 1 — decaying-sinusoid tracker (DEFAULT).** One gate gives two scalars per
half-swing (peak speed, timestamp); a decaying oscillator has ~3 slowly-varying
parameters (amplitude Θ, frequency ω, damping ζ). You're over-determined every
period — a perfect match, and far less code than a Kalman filter for the same numbers.

```
θ(t) ≈ Θ · e^(−ζω₀ (t−t₀)) · sin(ω(t−t₀) + φ)
ω_k    = π / T_half,k             (directly measures the amplitude-dependent slowdown)
ζω₀    ≈ ln(v_{k−1}/v_k) / T_half (log-decrement; linear in ln A ⇒ viscous;
                                    curved ⇒ add quadratic drag; linear in A ⇒ Coulomb)
```

**Tier 2 — PLL on crossing timestamps.** A phase accumulator advancing at ω̂,
PI-corrected each crossing. Very smooth, low-jitter phase (the low-vibration
controller loves this) and coasts cleanly through a single missed event.

**Tier 3 — EKF/UKF** on `[θ, θ̇, ω₀, (E)]`: predict with the ODE, sparse
two-component update `z = [θ=0, |θ̇| = Δα/Δt_block]` only when a `Crossing` arrives.
Over-engineering for one gate; keep it in the Python companion as the reference model
and the path to the array/second gate.

### 2.6 Direction — and why you mostly don't need it (yet)

A single bottom gate gives **|speed| and timing only**; left→right and right→left
pulses are identical. **Important:** for a bottom coil, pump/brake windows are placed
by *time* relative to the predicted crossing `t*`, and an attractive force pulls
toward center regardless of which side the bob approaches from — so **single-gate
amplitude control is direction-agnostic.** Direction is optional metadata for the
pendulum; it becomes load-bearing for the ring (§6) and for PM repel-on-approach
braking.

If you do want it: a free pendulum strictly alternates, so keep a parity bit seeded
at startup. Guard it against missed events by comparing the measured half-period to the
**amplitude-corrected** predicted `T(A)` (not a fixed constant), and don't flip parity
on a step where a crossing was clearly skipped. The clean upgrade is an **offset second
gate** at a small angle Δβ: firing order gives direction outright, inter-gate time
`θ̇ ≈ Δβ/Δt` gives an independent speed sample and a live Δα cross-check — and it is
the natural stepping-stone to the ring.

### 2.7 Output contract & confidence

```cpp
struct State {
    float theta, theta_dot;   // rad, rad/s at query time
    float amplitude, energy;  // rad, J (from ½mL²v² at last bottom)
    float omega, phase;       // rad/s; phase 0 at bottom moving +, wraps 2π/period
    uint32_t t_next_bottom;   // predicted next crossing (ticks) — controller pre-shapes on this
    int8_t direction;         // +1 / −1 (optional for pendulum)
    float confidence;         // 0..1
    enum { IDLE, SEARCHING, TRACKING, LOW_AMP } status;
    bool valid;
};
```

`confidence = c_time · c_apex · c_status`, with `c_time = exp(−(t_now−t_last)/τ)`
(staleness), `c_apex = 1 − k|sin φ|` (LOW at the apex, HIGH at the bottom),
`c_status ∈ {TRACKING:1, LOW_AMP:0.3, SEARCHING:0.1, IDLE:0}`. The estimate is
freshest and most trustworthy at the bottom — exactly where the magnet acts — so the
controller commits torque only when confidence is high.

Update energy/amplitude **unconditionally** (single clean measurement); update ω/ζ
**only on clean, non-missed** events. Debounce edges (Schmitt comparator,
`t_deb ≪ Δt_block,min`); on a missed event, coast and inflate uncertainty — never
blind-fire. Do all Δt as `uint32_t` modular subtraction so timer wrap is free.

---

## 3. The magnetic model + calibration (the part no OSS ships for you)

### 3.1 Plant equation

```
I θ̈ = −m g L sin θ − b θ̇ + τ_mag(θ,i),     I = m L²  (point-mass bob)
ω₀ = √(g/L),   T(A) ≈ T₀(1 + A²/16 + 11A⁴/3072 + …),   Q = I ω₀ / b,   ζ = 1/(2Q)
```

Store `I` and `L_eff = g/ω₀²` as primitives and **derive** `m_eff = I/L_eff²` — for
a real bob + string, effective mass comes from calibration, not a kitchen scale.

### 3.2 Separable torque map — and the crucial sign subtlety

For control/estimation we need torque about the pivot. Assume separability (excellent
near the axis):

```
τ_mag(θ, i) = q(θ) · f(i)
```

**`q(θ)` is an ODD function with a zero at the bottom-center.** This is the single
most important — and most counter-intuitive — fact in the whole design:

- With the coil directly below, at θ = 0 the attractive force is purely **radial**
  (straight down the string) → it does **zero tangential work** at the exact bottom.
- Off-center, the attractive force pulls the bob back toward θ = 0, so the *torque* is
  restoring: negative for θ > 0, positive for θ < 0. Odd, lobed, peaking a few degrees
  either side of bottom, ~0 for |θ| > θ_c where **θ_c ≈ (coil_radius + bob_radius)/L**.

Consequence for pumping (this corrects a natural but wrong intuition):

```
ΔE_pass = f(i) · ∫_window q(θ) dθ
```

Because `q` is odd, the integral over a **symmetric** window `[−θc, +θc]` is **zero** —
the approach lobe pumps and the departure lobe brakes an equal amount. **Energizing
the coil over the whole symmetric window delivers no net energy, just heat.** You add
energy only by using the **approach half** and cutting current at the bottom:

```
pump:  window [t* − T_p, t*],  Q_win_pump = ∫_{−θc}^{0} q(θ) dθ > 0,  cut at t*
```

The useful coupling `q(θ)·θ̇` peaks at a small offset `θ_opt` (a few degrees before
bottom, where q has grown but θ̇ hasn't dropped) — **not** at θ = 0. Center your pump
window on that offset, not on t* itself, or you'll pump almost nothing. Ideally derive
`T_p` and the offset from the measured `q(θ)` LUT rather than a fixed fraction.

### 3.3 The current law `f(i)`, per branch

- **PM bob:** `f = k_a · i` — linear, sign-reversible. `i > 0` attract, `i < 0` repel.
  Can pump, actively brake, and regenerate.
- **Soft-iron bob:** `f = k_r · i² · sat(i)` — quadratic, unsigned, saturating
  (`k_r i²/(1 + i²/i_sat²)`). Manage energy by timing only. Keep the peak current on
  the *linear* part (below `i_sat`) — past saturation, extra current buys no force but
  still costs I²R.

### 3.4 Calibration protocol (produces the plant descriptor)

Run in order; the tracks cross-check.

1. **Ring-down, coil off → ω₀, L_eff, Q, b, damping class.** Displace to A₀, log
   bottom-crossing times. Half-period → ω (extrapolate to zero amplitude to strip the
   large-angle term). Beam-block duration → peak-speed proxy; slope of `ln A_n` vs `t_n`
   is `−ω₀/(2Q)` → Q, then `b = I ω₀ / Q`.
2. **Force sweep → `f(i)`.**
   - *Static:* mount the **coil** on a pivoted arm resting on an HX711 load cell
     (measure the Newton's-third-law reaction), hold the bob rigidly at known angles on
     a drilled jig, step current, record. **Specify the lever arm** to convert force →
     torque. Expected forces at few-degree gaps are often a fraction of a newton — pick
     the load cell accordingly (e.g. 100 g vs 1 kg cell). A string-hung bob can't be
     held at angle; use a rigid fixture.
   - *Dynamic (in-situ, preferred cross-check):* fire a single short pulse of known `i`
     at a known phase on the swinging pendulum, read the amplitude jump from gate timing.
     This folds in driver delay, coil L/R rise-time, and eddy lag automatically. **Note
     it returns the product `f(i)·Q_win`, not `f(i)` alone** — fine for control, but
     don't describe it as isolating `f(i)`.
3. **Phase-swept pulse → `q(θ)`.** Sweep the pulse *center phase* at fixed `i`; a
   narrow pulse at phase θ_p samples `q(θ_p)·θ̇(θ_p)`. **SNR caveat:** the signal
   vanishes near the apex (θ̇ → 0) and near bottom-center (q → 0), so restrict this to
   the mid-range where θ̇ is safely nonzero, deconvolve by the estimator's `θ̇(θ)`, and
   average many passes per phase. Don't try to measure q near the apex from energy jumps.
4. **Single calibrated impulse → I, m_eff.** `I·Δθ̇ = f(i) q_peak Δt` from the velocity
   jump; `m_eff = I/L_eff²`. Static torque map and dynamic jump must be self-consistent.
5. **Store descriptor:** `{L_eff, I, m_eff, b (or b_visc,b_quad), Q, q[] LUT,
   f-params (k_a or k_r,i_sat), θ_c}`. Device *evaluates* it; host *fits* it (§7.6).

---

## 4. Part B — efficient, low-vibration drive

### 4.1 Objective

Per steady-state cycle: `minimize ∫ i²R dt  s.t.  E → E_tgt, |ampl error| < ε,
band-limited force, |di/dt| ≤ slew_max`. Work internally in energy; expose amplitude:

```
E_tgt = m g L (1 − cos θ_a,tgt) ≈ ½ m g L θ_a,tgt²
```

### 4.2 Energy-shaping law (with the damping fix)

Under applied torque τ, `dE/dt = τ θ̇`. The collocated passivity law (Åström–Furuta
swing-up; same idea as dfki energy shaping and penduino drive/brake):

```
τ(t) = −k · θ̇ · (E − E_tgt)   ⇒   dE/dt = −k θ̇² (E − E_tgt)
```

θ̇² ≥ 0, so E is driven monotonically to E_tgt, and the law **auto-selects pump vs
brake**: below target ⇒ torque with motion (pump); above ⇒ torque opposes motion
(brake); at target ⇒ coast. The θ̇ factor fixes the sign and makes injected power
vanish smoothly at turnaround.

**Damping correction (important):** pure proportional energy shaping leaves a
steady-state **droop** — it holds amplitude *below* target by ≈ ΔE_damp/k_E, because it
doesn't account for the per-swing loss. Add the measured loss as feed-forward (you
already compute `ΔE_k` in §2.3):

```
ΔE_cmd = ΔE_damp_est + k_E (E_tgt − E_n),   k_E ≈ 0.2–0.4   (fraction of error erased/swing)
```

Now the fixed point is E_tgt, not E_tgt − droop. Closed-loop pole ≈ (1 − k_E); settling
≈ −1/ln(1−k_E) swings. Spread corrections over several swings — one big kick rings the
structure.

### 4.3 Timing — why you cut current at the bottom

Near the crossing `θ ≈ θ̇*(t − t*)`, so tangential power `P ≈ −K_m L θ̇*² (t − t*)`.
With charge `Q = ∫ i dt` centered at `t_c`:

```
ΔE_pulse ≈ −K_m L θ̇*² · Q · (t_c − t*)
```

| Charge centroid `t_c` vs crossing `t*` | Result |
|---|---|
| **before** the crossing | ΔE > 0 → **PUMP** |
| **after** the crossing | ΔE < 0 → **BRAKE** |
| symmetric about the bottom | ΔE ≈ 0 → pure heat |

So **to add energy, place the pulse in the approach window and cut at t\*** (or a few
hundred µs early, so freewheel decays the field to zero *at* the bottom). Current
lingering past t* is on the wrong side of the symmetry and *subtracts* the work.
`|ΔE|` grows with offset but coupling `K_m(gap)` collapses (~1/gap²) as the bob leaves,
so the product peaks at a finite offset — a short band ≈ 10–25 % of a half-period,
centered on `θ_opt` (§3.2). Braking:
- **soft-iron:** mirror to the departure window `[t*, t*+T_p]` (attract on departure).
- **PM:** repel on approach `[t*−T_p, t*]` (reverse H-bridge), or regenerate.

Peak current from the calibrated inverse: `Q_needed = |ΔE_cmd| / K_pump(θ̇*, gap)`,
then invert the chosen envelope.

### 4.4 Low vibration — two *different* problems, don't conflate them

There are two distinct vibrations, with two distinct fixes:

**(a) Out-of-plane / conical string sway.** Excited by force components *perpendicular
to the swing plane* — coil-axis misalignment, an off-center bob, field asymmetry — **not**
by pulse timing. The fix is **mechanical**: align the coil axis to the swing plane,
center the bob mass, keep the tangential force in-plane. Instrument a frame accelerometer
on the perpendicular axis to catch it.

**(b) In-plane structural ringing / audible snap.** Excited by sharp force *edges*
(a current step is a near-impulse → broadband). The fix is **band-limiting the force**:
- Smooth envelope — raised-cosine (Hann): `i(t) = i_pk·½(1 − cos(2π(t−t₀)/T_p))`,
  derivative zero at both ends. For the **soft-iron i² bob**, shape the *force*:
  `i(t) = i_pk·√(½(1−cos(…)))` so `F ∝ i²` is the clean raised cosine — but note the
  current then has a *nonzero* endpoint slope `π·i_pk/T_p`, so size `slew_max` ≥ that.
- Slew-limit `|di/dt| ≤ slew_max`, ramp `τ_r ≳ 3 L_c/R`, and keep `slew_max < V_bus/L_c`
  (never command a slope the bridge can't deliver — it degenerates into a step).
- Don't over-drive: small `k_E`, bounded `ΔE_max`, correct over many swings.

**Do not** try to make the pump pulse time-symmetric about the bottom to "cancel lateral
impulse" — pumping is *deliberately* asymmetric (§4.3); a symmetric pulse delivers zero
net energy. Net tangential impulse when pumping is intentional and can't be symmetrized
away; the anti-vibration levers are (a) mechanical alignment and (b) band-limited edges.
Reserve the "symmetric/coast" idea for HOLD-mode dead-band swings where ΔE ≈ 0.

### 4.5 Efficiency — per branch, against fixed *work* (not fixed charge)

Heat per pulse is `∫ i²R dt`. Minimize it for a **fixed delivered work ΔE**:

- **PM branch (`f ∝ i`):** minimize `∫i²dt` s.t. `∫(qθ̇)·i dt = ΔE` ⇒ optimal
  `i(t) ∝ q(θ)·θ̇` — a smooth bump that is **zero at the bottom** (since q(0)=0) and
  peaks at `θ_opt`. So for a PM bob, **maximum efficiency and low vibration align**: the
  most efficient pulse is already smooth and bottom-nulled. No √-shaping needed.
- **Soft-iron branch (`f ∝ i²`):** efficiency depends mainly on **where** you put `i²`
  relative to the peak of `q·θ̇` (concentrate near peak coupling), *not* on peak-vs-flat
  shape. Timing dominates; the √-cosine current used for vibration is not a real
  efficiency sacrifice.

(The naïve "a flat/rectangular pulse minimizes ∫i²dt" only holds if coupling were
constant across the window — but `q(θ) ≈ 0` at the bottom makes that false.)

Other levers: **pulse only in the high-q window** (I²R accrues every ms the coil is on,
but work only where q·θ̇ is large); **synchronous freewheel** at cut (recirculate
½L_c i² through the coil, don't dump it in a snubber); **regenerative braking (PM only)**.

### 4.6 Modes (FSM)

```
IDLE ─start─► STARTUP ─phase-locked─► SWING_UP ─E≥0.95E_tgt─► HOLD
                                          │ E>E_tgt            │ stop
                                          └──► BRAKE/SOFT_STOP ◄┘
   any state ── overcurrent / overtemp / N missed gates ──► FAULT
```

- **STARTUP from rest — read the branch carefully:**
  - **PM bob:** a short repel breaks symmetry; then an open-loop alternating pulse train
    at ω₀ = √(g/L), quarter-period timed, until gate events arrive and phase locks.
  - **Soft-iron bob:** *the naïve approach fails.* At rest the bob hangs at bottom-center
    directly over the coil, where the attractive force is purely radial — **an
    attract-only coil cannot inject the first tangential energy, and "alternating" is
    meaningless for a unipolar attractor.** You need an explicit bootstrap: **(a)** mount
    the coil (or bob) with a small deliberate **lateral offset δ** from bottom-center, so
    an attract pulse produces a tangential component and rocks the bob off equilibrium;
    **(b)** a one-time manual nudge, then phase-lock on the first `Crossing`; or **(c)** a
    small dedicated off-axis kicker coil used for startup only. Document your chosen δ.
- **SWING_UP:** energy shaping, larger `ΔE_max`, still spread over swings.
- **HOLD:** small `k_E`, dead-band `|E − E_tgt| < ε` (~±2 %) → most swings coast → minimum
  heat and vibration at steady state.
- **BRAKE/SOFT_STOP:** ramp `E_tgt` down smoothly (no slam). PM: repel-on-approach or
  regen; soft-iron: attract-on-departure.
- **FAULT:** hard `I_max` clamp; I²t thermal integrator derates `ΔE_max` when hot;
  missed-event watchdog — if N predicted crossings arrive with no edge, fall back to
  open-loop at last-known ω/phase with reduced authority, then safe-coast. Never fire blind.

### 4.7 Per-event update (pseudocode — dimensional bug fixed)

```python
def on_gate_event(t_cross, w_gate):
    theta_dot = calib.dalpha / w_gate                 # 1) rad/s — dalpha is an ANGLE, no /L
    E = 0.5 * m * L**2 * theta_dot**2
    t_star = t_cross + estimator.half_period(E)        # 2) predict next crossing
    err = E_tgt - E
    if abs(err) < eps_E:                               # HOLD dead-band → coast
        return
    dE = clamp(dE_damp_est + k_E * err, -dE_max, +dE_max)   # 3) energy shaping + loss feed-fwd
    Q  = abs(dE) / calib.K_pump(theta_dot, gap_est)    # 4) charge from calibrated pump const
    T_p, offset = calib.window_from_q()                # 5) window/offset from measured q(θ)
    if dE > 0:            t0,t1,pol = t_star-T_p-offset, t_star, ATTRACT   # PUMP, cut at bottom
    elif bob_is_PM:       t0,t1,pol = t_star-T_p-offset, t_star, REPEL     # PM brake (or REGEN)
    else:                 t0,t1,pol = t_star, t_star+T_p, ATTRACT          # soft-iron brake
    i_peak = min(peak_from_charge(Q, T_p, ENV, bob_is_PM), I_max, thermal.derate())
    emit(pulse_cmd(t0, t1, i_peak, ENV, pol, slew_max, seq++))
    watchdog.expect_edge_near(t_star); thermal.accumulate(i_peak, T_p)
```

### 4.8 Controller → power-stage contract

```c
typedef enum { ENV_RAISED_COSINE, ENV_TRAPEZOID, ENV_SQRT_RCOS } envelope_t;
typedef enum { POL_ATTRACT, POL_REPEL, POL_REGEN } polarity_t;   // REPEL/REGEN: PM only
typedef struct {
    uint32_t t_start_us, t_end_us;   // t_end == t* for a pump pulse (hard cut)
    float    i_peak_A;               // pre-clamped ≤ I_max
    envelope_t envelope;             // shape the current loop must track
    polarity_t polarity;
    float    slew_max_A_us;
    uint16_t seq;
} pulse_cmd_t;
```

Guarantees: current is 0 at `t_start`, forced to 0 by `t_end` (freewheel accounted so
the field is gone at the bottom for a pump), `|di/dt| ≤ slew_max`, `i_peak ≤ I_max`.
The current loop's only job: track `i_ref(t)` and report measured `i` and I²t.

**Tuning order:** `E_tgt` (ramp up first time, watch the lateral mode) → `T_p`/offset
(sweep, maximize measured ΔE per ∫i²R) → `k_E` (start 0.2) → `ε_E` dead-band → current-
loop PI + slew. **Benchmarks (mirror dfki):** settling swings to 95 %, steady-state
amplitude RMS error (<1–2 %), energy-per-swing, a vibration proxy (perpendicular-axis
frame-accel RMS, or current jerk ∫(d²i/dt²)²), robustness under M % dropped gate events.

---

## 5. Reference hardware (so Phase 1 has real numbers)

The real-time budgets and slew limits are meaningless without concrete parts. A workable
starting BOM:

| Part | Suggestion | Why it matters |
|---|---|---|
| MCU | ESP32 / STM32F4 (FPU, timer-capture + MCPWM/TIM) | Float loop, hardware capture pins |
| Driver | DRV8871 / VNH7070-class H-bridge (PM) or a low-side N-FET + freewheel diode (soft-iron) | Bidirectional (PM) vs unipolar; sets `V_bus` |
| Bus | `V_bus` 12–24 V (pick to satisfy `slew_max < V_bus/L_c`) | Caps achievable `di/dt` |
| Coil | measure `L_c`, `R` | Set `slew_max`, `τ_r ≳ 3L_c/R`, PWM freq |
| **Current sense** | low-side shunt + INA240 (or INA-series) sized for `I_max`, PWM-synchronous ADC sample | **Required** for the closed current loop and overcurrent fault |
| **Temp sense** | coil-mounted NTC, *or* a first-order thermal estimator `T += (i²R − (T−T_amb)/R_th)·dt/C_th` | **Required** for overtemp/derate; copper R rises ~0.39 %/°C, which shifts the soft-iron `f(i)` map — track `R(T)` |
| Gate | phototransistor/photodiode + comparator with **Schmitt hysteresis**, `t_deb ≪ Δt_block,min` | Clean single edges into the capture pin |

Fill in `V_bus`, `L_c`, `R`, `I_max`, PWM frequency, dead-time, and the timer/pin map,
then compute `slew_max`, `τ_r`, `T_p` from them.

---

## 6. Scaling to the coil ring (with the stall caveat)

The same `Crossing` → phase machinery scales. For **N magnets with N gates between
them**, partition the rotor angle into N sectors of 2π/N; each `Crossing(sensor_id)`
marks a sector-boundary transition; between gates you dead-reckon ψ from rotor speed,
exactly as you dead-reckon θ. **Two gates flanking one magnet** give direction (which
fired first) + speed (`ω ≈ Δβ_gates/Δt`) per pass — the array analog of the offset
second gate, and it removes the alternation-parity fragility entirely.

| Pendulum | Ring |
|---|---|
| single bottom gate | N sector-boundary gates |
| "cut at the bottom" | "commutate at magnet-aligned"; pump/brake offset → **phase advance** `θ_adv = β·ω` to compensate coil L/R lag |
| energy shaping `q* = k(E_des−Ê)·sign(ω)` | sector **torque shaping** via `commTable[sector][dir] → {coil, polarity, advance}` |
| raised-cosine pulse | raised-cosine torque profile blended across adjacent coils (kills cogging/vibration at hand-off) |
| STARTUP → phase-lock | open-loop forced commutation → phase-lock |

```cpp
struct CommEntry { int8_t coil; int8_t polarity; float advance; };
static const CommEntry commTable[N][2];   // [sector][dir]
```

**The stall caveat.** Gates-between-magnets give excellent direction+speed *at running
speed* but **nothing at low speed, high torque, or stall** — no motion, no edges, so
absolute sector is blind (the classic sensorless-BLDC low-speed problem; the
`LOW_AMP`/`SEARCHING` states are its direct precursor). Two mitigations, both leaving the
event pipeline unchanged:
1. **Open-loop startup ramp** — apply a slowly rotating current vector at increasing
   frequency to drag the rotor into synchronism; hand off to closed loop once edges arrive
   at a consistent cadence. Cheap, no extra hardware, but possible cogging and no
   guaranteed direction under load.
2. **One absolute sensor.** An AS5600 is ~$1 but is a **Hall** angle sensor — in a ring
   of switching electromagnets with a magnet rotor, stray/switching fields corrupt it
   exactly in the stall regime where you need it. If you go this route, budget for
   shielding and axial placement away from coil flux, or prefer an **optical/inductive**
   absolute encoder. Otherwise lean on the open-loop ramp and treat absolute encoding as
   an optional, non-trivial add-on. Either way it's just another `IEventSource`.

---

## 7. Software specifics

### 7.1 Core types & queue (correctness note)

```cpp
template <uint16_t N>                       // N a power of two
class EventQueue {                          // lock-free SPSC ring
public:
    bool push(const Crossing& c) noexcept;  // ISR (producer)
    bool pop(Crossing& out) noexcept;       // fast tick (consumer)
private:
    Crossing buf_[N];
    std::atomic<uint16_t> head_{0}, tail_{0};   // NOT plain volatile
};
```

**Plain `volatile` is not sufficient.** It orders volatile accesses relative to each
other but does **not** order the `buf_[tail_] = c` payload write relative to the index
publish — the consumer can pop a half-written `Crossing`. Publish the payload **before**
the index with acquire/release: `store(release)` / `load(acquire)`, or on bare-metal a
`DMB`/compiler barrier between the payload write and the index update (symmetric on the
consumer). This is a correctness requirement, not a style choice.

### 7.2 Estimator — one interface, two implementations

```cpp
class IEstimator {
public:
    virtual State update(EventQueue<64>& q, uint32_t now) = 0;  // drain + dead-reckon to now
    virtual const State& state() const = 0;
};
class PendulumEstimator final : public IEstimator {   // 1- or 2-gate, decaying-sinusoid (Tier 1)
    PendulumEstimator(const PlantModel& plant, uint8_t n_gates);
    void onCrossing(const Crossing&);   // phase reset (θ=0) + speed from pulse_width + parity guard
};
class RingEstimator final : public IEstimator {       // N-gate: sector++/--, ω from spacing/Δt
    RingEstimator(const PlantModel& plant, uint8_t n_sectors);
};
```

### 7.3 Plant + force map

```cpp
class PlantModel {
    void  deriv(const State& x, real_t tau, real_t& dtheta, real_t& domega) const;
    real_t energy(const State& x) const;             // ½mL²ω² + mgL(1−cosθ)
};
enum class MagKind : uint8_t { PermanentBidir, SoftIronAttractOnly };
class IForceMap {
    virtual real_t force(real_t theta, real_t i) const = 0;        // q(θ)·f(i)
    virtual real_t current_for(real_t theta, real_t q) const = 0;  // inverse; soft-iron rejects q<0
    virtual MagKind kind() const = 0;
};
class TableForceMap final : public IForceMap {       // flat const grid in flash, bilinear interp
    TableForceMap(const real_t* grid, uint16_t nT, uint16_t nI,
                  real_t t0, real_t dT, real_t i0, real_t dI, MagKind k);
};
```

### 7.4 Supervisor / current / stage / orchestrator

```cpp
class ISupervisor {
    virtual real_t desiredForce(const State& x, const Targets& t) = 0;  // 0 outside the window
    virtual bool   safe(const State& x) const = 0;
};   // PendulumSupervisor: energy shaping.  RingSupervisor: sector torque shaping + phase advance.

class ICurrentController { virtual real_t update(real_t q_star, real_t theta, uint32_t now)=0;
                           virtual void setSlew(real_t amps_per_s)=0; };
class IPowerStage        { virtual void setDuty(real_t d)=0;                  // [-1,+1] H-bridge
                           virtual void setPhaseDuty(uint8_t ph, real_t d){}  // ring
                           virtual void enable(bool)=0; virtual uint16_t faults() const=0; };

class Actuator {                                     // HW-agnostic; strategy injected
    void controlTick(uint32_t now){                  // fast, 1–10 kHz
        State x = est_.update(q_, now);
        if (!sup_.safe(x)) { stage_.enable(false); return; }
        stage_.setDuty( cc_.update(last_q_star_, x.theta, now) );
    }
    void superTick(){ last_q_star_ = sup_.desiredForce(est_.state(), targets_); } // slow, 50–200 Hz
    EventQueue<64> q_;  /* refs to src_, est_, sup_, cc_, stage_, plant_ */
};
```

### 7.5 Numeric types

Express **every interface signature in terms of `real_t`** from the start (`float` on
FPU targets, a Q16.16 wrapper on M0/AVR). Don't hardcode `float` and claim the fixed-point
port is "just a typedef" — it isn't unless the signatures already use `real_t`. Keep a
float reference build for host validation. ISR code: **integer always**.

### 7.6 Device evaluates, host fits

> The MCU owns anything with a deadline; the host owns anything with a dataset.

- **Device (C++):** capture, queue, estimator, current loop, power stage, safety,
  watchdog. Loads a *flashed* calibration blob — never fits, only evaluates.
- **Host (Python):** `plant_sim` (same ODE in NumPy/SciPy — the primary unit-test oracle;
  feed synthetic `Crossing` streams, assert the estimator reconstructs θ(t)); `calibrate`
  (fit `q(θ)·f(i)`, emit the table); `sysid` (fit L, m_eff, b, Q from ring-down); `tune`
  (optimize `k_E`, `T_p`, slew against the sim); `bench/plot` (energy-per-cycle, lateral
  excitation, regression).
- **Wire formats:** telemetry up = length-prefixed **CBOR** or **COBS-framed packed
  structs** over USB-CDC/UART (reserve a raw-`Crossing` channel so the host sees exactly
  what the ISR saw); JSON only on the slow config channel. Calibration down =
  `calibration.bin` (packed header `{magic, version, MagKind, nθ, ni, θ0, dθ, i0, di,
  crc32}` + grid) for flash/LittleFS, **and** a `constexpr calibration.hpp` for the
  no-filesystem Arduino case — `TableForceMap` consumes either identically.

### 7.7 Repo layout & testing tiers

```
emac/ core/ estimate/ model/ control/ hal/{stm32,esp32,arduino}/ Actuator.hpp
      examples/{pendulum_1gate,pendulum_2gate,ring_3coil}/  tools/python/  tests/{host,hil}/
      data/  docs/  LICENSE(MIT) NOTICE
```

`core/ estimate/ model/ control/` are **hardware-free** → compile natively → unit-testable.
- **Tier 1** native unit tests (synthetic events from `plant_sim`; assert reconstruction
  tolerance, direction never flips spuriously, `current_for(force(θ,i)) ≈ i`, supervisor
  holds E_des) — every commit.
- **Tier 2** closed-loop sim: link the *real* C++ estimator+supervisor+current loop to the
  Python plant (pybind/socket) — nightly.
- **Tier 3** HIL: signal-HIL (a second MCU replays synthetic edges into the real capture
  ISR while you scope PWM) and full-HIL (real rig streaming CBOR) — on demand.

---

## 8. Build roadmap

- **Phase 0 — Skeleton & sim (host only).** `core/ model/`, `Crossing`/`State`,
  `EventQueue`, `PlantModel`, Python `plant_sim`. Tier-1 tests: synthetic events →
  estimator reconstruction within tolerance. Develop everything below against this oracle
  before touching silicon. **Current repo milestone:** the Python simulator is packaged
  as `emac_sim` with a CLI (`emac-phase0`) and pytest coverage; the C++ core skeleton
  remains future work.
- **Phase 1 — MVP drive (open loop, 1 gate).** `IEventSource` (timer capture) +
  `IPowerStage` + a trivial supervisor: fixed raised-cosine pulse in a fixed approach
  window before each crossing, cut at t*. Prove capture, queue, timing budget, and that a
  well-timed pulse pumps. Reference: Vernier PendulumDriver. *Exit:* sustained swing.
- **Phase 2 — Estimator (Tier 1).** `PendulumEstimator`: Δα calibration, pulse-width→speed,
  ½mL²v² energy, symplectic dead-reckoning re-anchored each crossing, ω/ζ from consecutive
  crossings, full `State` incl. `t_next_bottom` + confidence. Add the offset second gate as
  the first hardware upgrade. *Exit:* reconstructed θ(t) tracks ground truth; robust to a
  dropped event.
- **Phase 3 — Calibration & ForceMap.** Run §3.4; fit in Python; emit
  `calibration.bin`/`.hpp`; load into `TableForceMap`. Both branches. *Exit:* `current_for`
  round-trips; predicted ΔE/pulse matches measured within a few %.
- **Phase 4 — Optimal low-vibration control.** Energy-shaping supervisor with damping
  feed-forward, pump/brake window, shaped envelopes, closed current loop (PI + FF), FSM,
  benchmarks. Tune in order. *Exit:* holds amplitude to <1–2 % RMS, no visible lateral
  mode, minimal energy/swing.
- **Phase 5 — Array/ring.** Reuse `Actuator`/queue/current loop/stage verbatim; inject
  `RingEstimator` + `RingSupervisor` + `commTable` + N-phase stage; flanking-gate
  direction+speed; phase advance; cross-coil blending; open-loop startup ramp; optional
  absolute encoder. Reference: SimpleFOC. *Exit:* closed-loop commutation at running speed;
  graceful open-loop start.

---

## 9. What to borrow vs reference (licensing)

Keep `emac/` **MIT**. GPL/vendor material lives *nowhere in the tree* — study for ideas,
reimplement clean, cite in `NOTICE`.

| Project | License | Use |
|---|---|---|
| **SimpleFOC / Arduino-FOC** | MIT | **Borrow** — sensor/driver/current abstractions, layered pattern; template for the ring side. |
| **Vernier PendulumDriver** | permissive | **Borrow** — minimal photogate→coil reference for the 1-gate MVP. |
| **dfki torque_limited_simple_pendulum** | BSD-3 | **Borrow** — energy-shaping math, system-ID, the benchmarking harness for the Python companion. |
| **ODrive firmware** | MIT | **Borrow** — observer/commutation structure, calibration flow, telemetry ideas. |
| **penduino / practable** | GPL | **Reference only** — drive+brake+calibration ideas; clean-room reimplement. |
| **VESC** | GPL | **Reference only** — sparse-sensor commutation ideas. |
| **PulseCapture** | GPL | **Reference only** — the timer-capture *technique*; write your own on the vendor HAL (STM32 TIM / ESP32 MCPWM·PCNT / RP2040 PIO). |
| **Microchip AN957 / NXP app notes** | vendor-restricted | **Reference only** — usually licensed for that vendor's silicon; conceptual only. |

---

### One-paragraph summary

Treat the sensor edge — not an angle — as ground truth. A single `Crossing` from a
hardware-timer capture ISR feeds a lock-free queue; a decaying-sinusoid estimator turns
pulse-width into bottom speed (θ̇ = Δα/Δt_block), into energy (½mL²θ̇²), and dead-reckons
the nonlinear pendulum ODE symplectically between events, re-anchoring at each bottom where
the estimate is most certain. A separable **calibrated** torque map `q(θ)·f(i)` — whose key
feature is that `q(θ)` is **odd and zero at the bottom** — lets an energy-shaping supervisor
(`ΔE = ΔE_damp + k_E(E_tgt−E)`) deliver a shaped, approach-only current pulse **cut at the
bottom** (pump before, brake after), minimizing both I²R and structural excitation. Seven
narrow interfaces keep the MCU owning deadlines and Python owning datasets, and make the
pendulum→ring jump a construction-time swap of `IEstimator`+`ISupervisor`. Ship MIT; borrow
SimpleFOC/Vernier/dfki/ODrive, reference-only the GPL/vendor code.
