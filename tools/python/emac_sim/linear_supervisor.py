"""Forward-only commutation supervisor for the linear stepper (docs/DESIGN_LINEAR.md).

Because q_shape is zero-and-odd at each coil's own center, "pump on approach, cut at
center" -- supervisor.py's pendulum trick -- applies unchanged at every station. PulseCmd
and current_at() are reused verbatim from supervisor.py -- neither has anything pendulum-
specific in it.

For a PM-branch coil, there IS a direction/polarity choice, and both options add forward
energy: attract while approaching (q_shape>0 there, current>0 -- the existing pump), OR
repel while departing (q_shape<0 there, current<0, product still positive -- see
docs/DESIGN_LINEAR.md's repel-pumping section). Every station with a nonzero PM term gets
BOTH: an approach-attract pump (`_run_step`), then, once dead-reckoning says its center has
been reached, a departure-repel kick (`_fire_departure`, scheduled by `on_gate` and fired
by `tick()`). Reluctance-only coils (k_lin=0) can't repel at all (attract-only regardless
of current sign), so they only ever get the approach pump, unchanged from before.

This module assumes the gate-immediately-before-its-coil layout produced by
linear_plant.default_gate_stations() (gate[j] precedes coil[j]): the coil to target after
gate[j] fires is simply coil[j]. A different gate/coil placement scheme would need its own
gate-to-coil mapping; that generalization is not built here (see docs/DESIGN_LINEAR.md
open questions).

STARTUP walks forward through stations (never repeating one consecutively) rather than
alternating -- alternation is meaningless for a one-way, attract-only actuator. A resting
slug can coincide with at most one station's exact zero-force center, so this escapes any
single-coil detent within at most two attempts; exhausting every station with no gate
response means FAULT (no slug / jammed tube).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .linear_estimator import LinearStepperEstimator
from .linear_plant import CoilStation, LinearActuatorParams
from .supervisor import PulseCmd, _q_window_integral, envelope_average_linear

BOOTSTRAP = "BOOTSTRAP"
RUN = "RUN"
DONE = "DONE"
FAULT = "FAULT"

_COAST = PulseCmd(False, "coast", 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class StepperOutput:
    coil_index: int
    cmd: PulseCmd


def _station_k_pump(coil: CoilStation, pm_envelope: str = "rcos") -> tuple[float, float]:
    """(k_quad, k_lin): the reluctance branch delivers dE ~ k_quad*i_peak^2 (J per A^2,
    same K_pump derivation as EnergySupervisor, always using "sqrt_rcos" so i^2 is the
    smooth raised cosine -- its 0.5 factor is fixed, not tied to pm_envelope). The PM
    branch delivers dE ~ k_lin*i_peak (J per A, LINEAR since F ~ i there, not i^2), scaled
    by whichever current envelope the PM branch actually uses (pm_envelope) -- "rcos"
    (default, smooth force, 0.5 average) through "square" (unsmoothed, 1.0 average, more
    thrust per i_peak at the cost of sharper edges); see supervisor.envelope_average_linear."""
    q_win = _q_window_integral(coil.x_c, -coil.x_c, 0.0)
    k_quad = 0.5 * coil.Cmag * q_win                              # J per A^2
    k_lin = envelope_average_linear(pm_envelope) * coil.k_a * q_win   # J per A
    return k_quad, k_lin


def _i_peak_for_energy(dE: float, k_quad: float, k_lin: float, i_max: float) -> float:
    """Invert dE = k_quad*i^2 + k_lin*i for the positive i_peak. Exact (given the fixed
    envelope-shape choice below) for either pure branch -- reduces to sqrt(dE/k_quad) when
    k_lin=0 (pure reluctance, matching EnergySupervisor's original formula exactly) and to
    dE/k_lin when k_quad=0 (pure PM). When BOTH are nonzero (a hybrid slug) this is an
    approximation: the two branches individually want different current envelope shapes
    for a perfectly smooth force (sqrt-raised-cosine vs. plain raised-cosine -- see
    current_at()), so no single current waveform makes both perfectly smooth at once;
    solving the combined quadratic still gives a sensible total-energy-correct i_peak."""
    dE = max(0.0, dE)
    if k_quad <= 1e-15 and k_lin <= 1e-15:
        return 0.0
    if k_quad <= 1e-15:
        i_peak = dE / k_lin
    else:
        disc = k_lin * k_lin + 4.0 * k_quad * dE
        i_peak = (-k_lin + math.sqrt(disc)) / (2.0 * k_quad)
    return min(i_peak, i_max)


class StepperSupervisor:
    def __init__(self, p: LinearActuatorParams, k_v: float = 0.30, T_p_frac: float = 0.30,
                 phase_advance_s: float = 0.002, i_max: float = 6.0,
                 bootstrap_dwell_s: float = 0.05, bootstrap_timeout_s: float = 0.20,
                 pm_envelope: str = "rcos"):
        self.p = p
        self.k_v = k_v
        self.T_p_frac = T_p_frac
        self.phase_advance_s = phase_advance_s
        self.i_max = i_max
        self.bootstrap_dwell_s = bootstrap_dwell_s
        self.bootstrap_timeout_s = bootstrap_timeout_s
        # Current envelope used whenever a station's PM branch is active (k_lin>0):
        # "rcos" (default, smooth force) | "trapezoid" | "square" (both unsmoothed, more
        # average current per i_peak -- a genuine speed-vs-vibration knob). Reluctance-only
        # stations always use "sqrt_rcos" regardless -- see _station_k_pump.
        self.pm_envelope = pm_envelope
        self.K_pump = [_station_k_pump(c, pm_envelope) for c in p.coils]   # list of (k_quad, k_lin)

        self.mode = BOOTSTRAP
        self._bootstrap_coil = 0
        self._bootstrap_t0 = 0.0
        self._attempts = 0
        self.active = StepperOutput(0, _COAST)
        self._final_arrival = None      # predicted time the slug passes the LAST coil's center
        self._final_T_p = self.bootstrap_dwell_s
        self._final_v_hat = 0.0         # speed measured at the last gate, for brake_hold sizing
        self._pending_departure = None  # (coil_index, t_arrival) -- next repel-departure to fire
        self._departure_est = None      # estimator reference, for predict()ing velocity at fire time
        self._last_v_tgt = 0.0

    def start(self, t: float) -> None:
        """Call once before the run starts to arm the first bootstrap pulse."""
        self.mode = BOOTSTRAP
        self._bootstrap_coil = 0
        self._attempts = 0
        self._bootstrap_t0 = t
        self._final_arrival = None
        self._pending_departure = None
        self._arm_bootstrap(t)

    def _arm_bootstrap(self, t: float) -> None:
        _, k_lin = self.K_pump[self._bootstrap_coil]
        envelope = self.pm_envelope if k_lin > 1e-15 else "sqrt_rcos"
        self.active = StepperOutput(
            self._bootstrap_coil,
            PulseCmd(True, "pump", t, t + self.bootstrap_dwell_s, self.bootstrap_dwell_s,
                     self.i_max, 0.0, envelope),
        )

    def tick(self, t: float) -> StepperOutput:
        """Call every fast control tick; advances the bootstrap FSM on timeout (a gate
        event may not arrive for a while -- or ever, if the wrong station was fired), fires
        a station's departure-repel kick once dead-reckoning says its center has been
        reached, and fires end-of-travel once the last coil's center has been reached --
        there is no gate after any of these to trigger them directly."""
        if self.mode == BOOTSTRAP and t - self._bootstrap_t0 > self.bootstrap_timeout_s:
            self._attempts += 1
            if self._attempts >= len(self.p.coils):
                self.mode = FAULT
                self.active = StepperOutput(0, _COAST)
            else:
                self._bootstrap_coil = (self._bootstrap_coil + 1) % len(self.p.coils)
                self._bootstrap_t0 = t
                self._arm_bootstrap(t)

        if self.mode == RUN and self._pending_departure is not None:
            coil_index, t_arrival = self._pending_departure
            if t >= t_arrival:
                self.active = self._fire_departure(coil_index, t_arrival)
                self._pending_departure = None

        if self.mode == RUN and self._final_arrival is not None and t >= self._final_arrival:
            self.active = self._end_of_travel(self._final_arrival)
            self._final_arrival = None

        return self.active

    def on_gate(self, gate_index: int, est: LinearStepperEstimator, t: float,
                v_tgt: float) -> StepperOutput:
        """Call right after a gate crossing is accepted by the estimator. gate[j] precedes
        coil[j] in this layout (see module docstring), so every gate -- including the last
        one -- targets a normal pump-and-cut of its own coil; end-of-travel is decided
        separately, in tick(), once that final approach completes."""
        if self.mode == BOOTSTRAP:
            self.mode = RUN

        if self.mode != RUN:
            return self.active

        self._last_v_tgt = v_tgt
        self.active = self._run_step(gate_index, est, t, v_tgt)

        # Schedule this SAME coil's departure-repel kick (PM branch only -- reluctance
        # can't repel) for once dead-reckoning says its center has been reached, which is
        # after the approach pump above has already cut. Not for the last coil:
        # _end_of_travel below owns what happens once that one's center is reached.
        #
        # Deliberately uses PLAIN, naive dead reckoning here (est.time_to_reach), NOT
        # _run_step's accel-corrected _predict_arrival estimate, even though that estimate
        # is what schedules the approach pump's own cutoff just above. The two timings
        # have OPPOSITE safe-failure directions: the approach pump must not run PAST
        # center (a late estimate is the danger -- _predict_arrival's correction exists
        # precisely to prevent that), while the departure kick must not fire BEFORE
        # center (an early estimate is the danger here). Reusing the same aggressive,
        # early-biased correction for both was the root cause of two separate incidents
        # this session: an unbounded correction firing the departure kick while still
        # approaching, and a kinetic-energy-based cap that "fixed" that by throttling the
        # SAME correction the approach pump also needed, breaking a bootstrap-acceleration
        # design instead. Naive dead reckoning systematically OVER-estimates transit time
        # (the whole reason the approach-pump correction was needed in the first place),
        # which is exactly the safe direction here: the departure kick may fire a little
        # later than the ideal instant (some lost repel-assist), but never before the slug
        # has actually arrived.
        if gate_index < len(self.p.coils) - 1:
            _, k_lin = self.K_pump[gate_index]
            if k_lin > 1e-15:
                coil = self.p.coils[gate_index]
                t_arrival = est.time_to_reach(coil.position_m)
                if t_arrival is not None:
                    self._pending_departure = (gate_index, t_arrival)
                    self._departure_est = est

        if gate_index == len(self.p.coils) - 1:
            last_coil = self.p.coils[-1]
            # Same reasoning as the departure kick above: end-of-travel's repel/brake
            # kick must not fire before the slug truly reaches the last coil, so this
            # uses naive dead reckoning too, not the approach-pump's own early-biased
            # correction.
            self._final_arrival = est.time_to_reach(last_coil.position_m)
            self._final_v_hat = est.v_last
            # Window width matched to how long the slug actually spends inside the departure
            # lobe (one x_c of travel at the measured speed) -- NOT T_p_frac * inter-station
            # time (that timescale is for spacing pump windows between stations, and is
            # usually much shorter than the lobe itself, which would under-deliver the brake).
            self._final_T_p = last_coil.x_c / max(abs(self._final_v_hat), 1e-6)

        return self.active

    def _fire_departure(self, coil_index: int, t_arrival: float) -> StepperOutput:
        """Repel-pump the departure lobe of a coil the slug just passed: q_shape<0 there,
        and a negative (repel) current makes the product positive -- forward force, same
        magnitude as the approach-side attraction would give (docs/DESIGN_LINEAR.md).
        PM-only (reluctance is attract-only regardless of current sign, so k_quad plays no
        part in sizing this -- pass 0.0 for it explicitly, not the station's actual k_quad)."""
        if self._departure_est is None:
            return StepperOutput(coil_index, _COAST)
        coil = self.p.coils[coil_index]
        _, k_lin = self.K_pump[coil_index]
        _, v_hat = self._departure_est.predict(t_arrival)

        E_hat = 0.5 * self.p.mass_kg * v_hat * v_hat
        E_tgt = 0.5 * self.p.mass_kg * self._last_v_tgt * self._last_v_tgt
        dE_cmd = max(0.0, self.k_v * (E_tgt - E_hat))
        if dE_cmd < 1e-9:
            return StepperOutput(coil_index, _COAST)

        T_p = coil.x_c / max(abs(v_hat), 1e-6)   # same lobe-traversal-time sizing as brake_hold
        i_peak = _i_peak_for_energy(dE_cmd, 0.0, k_lin, self.i_max)
        return StepperOutput(coil_index,
                              PulseCmd(True, "pump", t_arrival, t_arrival + T_p, T_p,
                                       i_peak, dE_cmd, self.pm_envelope, "repel"))

    def _predict_arrival(self, est: LinearStepperEstimator, x_target: float,
                          dE_cmd: float, tau_elec: float = None):
        """Predicted time the slug reaches x_target, correcting plain constant-velocity
        dead reckoning (est.time_to_reach) for the accel a pulse of dE_cmd joules is about
        to add before it gets there. Constant-velocity extrapolation from the last gate's
        measured speed systematically UNDER-estimates arrival speed (hence OVER-estimates
        arrival time) whenever the about-to-fire pump meaningfully changes velocity first --
        exactly the regime a high-thrust design lands in. A late arrival estimate pushes the
        pump's cutoff (t1 = t_arrival - phase_advance_s) late too, so the coil is still near
        peak current when the slug actually crosses its center: q_shape's sign has already
        flipped by then, so the same "attract" current now pulls backward, decelerating (or
        even reversing) the slug instead of releasing it -- caught via exactly this failure
        mode at high i_max_a. SUVAT's constant-acceleration average velocity is the standard
        fix: (v0+v1)/2 gives the exact transit time for a genuinely constant accel, unlike
        v0 alone; v1 comes from energy conservation. dE_cmd here must be the energy actually
        DELIVERABLE at the (i_max-clamped) i_peak, not the raw commanded dE -- v_tgt can ask
        for far more than i_max can supply, and using the unreachable raw dE overshoots the
        correction just as badly the other way (predicts arrival too EARLY, firing the
        departure-repel kick while still approaching -- caught in testing right after fixing
        the late-cutoff case).

        Even the deliverable-energy fix above can still overshoot badly for a light slug /
        high-current design: K_pump's energy-per-A(^2) calibration assumes a full lobe-
        spanning pulse, but the resulting T_p (sized FROM this same corrected t_arrival, a
        circular dependency) can end up far too short for the RL current loop to ever
        actually reach, let alone hold, that much current -- so the assumed dE is never
        really delivered, yet the schedule still trusted it. An EARLIER version of this fix
        capped dE at the slug's own current kinetic energy (at most doubling it per lobe
        pass) to guard against exactly this -- but that bound conflates two unrelated
        physical scales: it also throttles the (perfectly legitimate, and the MOST common)
        case of a slug accelerating from near rest, where a strong first pump is SUPPOSED
        to inject many multiples of the slug's current, near-zero kinetic energy. Capping
        by that ratio broke that case, reverting most of the way back to the original
        late-cutoff bug (caught via a design whose i_max_a x pump_envelope sensitivity
        showed an unexplained cliff -- rcos collapsing to ~1.2 m/s at high current when
        square/trapezoid kept climbing -- that traced back to exactly this).

        The right question isn't "how big is dE relative to current KE" -- it's "does the
        coil ELECTRICALLY have time to reach the current this assumes", via tau_elec (the
        coil's own L/R time constant, passed in by the caller). Iterates a small fixed-point
        loop: predict t_arrival from the current energy assumption, size the implied T_p
        from it, then discount the assumed energy by the RC-style charge fraction
        (1 - exp(-T_p/tau_elec)) that fraction of a pulse could actually deliver, and
        re-predict -- converges in a few steps, and leaves a bootstrap-from-rest pump
        (long available T_p relative to tau_elec) essentially undiscounted, while still
        throttling a design whose commanded pulse would need to be electrically
        instantaneous to work as assumed."""
        if not est.have or abs(est.v_last) < 1e-9:
            return None
        v0 = est.v_last
        dE_cmd = max(0.0, dE_cmd)
        dE = dE_cmd
        t_arrival = None
        for _ in range(6):
            if dE > 1e-12:
                v1 = math.sqrt(v0 * v0 + 2.0 * dE / self.p.mass_kg)
                v_eff = 0.5 * (v0 + v1)
            else:
                v_eff = v0
            if abs(v_eff) < 1e-9:
                return None
            t_arrival = est.t_last + (x_target - est.x_last) / v_eff
            if tau_elec is None or tau_elec <= 0.0 or t_arrival <= est.t_last:
                break
            transit = t_arrival - est.t_last
            t_p = self.T_p_frac * transit
            deliverable_frac = max(0.02, min(1.0, 1.0 - math.exp(-t_p / tau_elec)))
            new_dE = dE_cmd * deliverable_frac
            if abs(new_dE - dE) < 1e-9 * max(dE_cmd, 1.0):
                dE = new_dE
                break
            dE = new_dE
        return t_arrival

    def _run_step(self, gate_index: int, est: LinearStepperEstimator, t: float,
                  v_tgt: float) -> StepperOutput:
        coil = self.p.coils[gate_index]
        if not est.have or abs(est.v_last) < 1e-9:
            return self.active     # not enough information yet; keep coasting

        # Energy-shaping law (same shape as EnergySupervisor's, in kinetic-energy terms
        # rather than raw velocity, so it stays dimensionally consistent with K_pump's
        # J-per-A^2 definition): forward-only, so never command a negative (braking) dE here.
        E_hat = 0.5 * self.p.mass_kg * est.v_last * est.v_last
        E_tgt = 0.5 * self.p.mass_kg * v_tgt * v_tgt
        dE_cmd = max(0.0, self.k_v * (E_tgt - E_hat))

        k_quad, k_lin = self.K_pump[gate_index]
        i_peak = _i_peak_for_energy(dE_cmd, k_quad, k_lin, self.i_max)
        dE_deliverable = k_quad * i_peak * i_peak + k_lin * i_peak

        tau_elec = coil.inductance_h / coil.resistance_ohm if coil.resistance_ohm > 0.0 else None
        t_arrival = self._predict_arrival(est, coil.position_m, dE_deliverable, tau_elec)
        if t_arrival is None or t_arrival <= t:
            return self.active     # not enough information yet; keep coasting

        if dE_cmd < 1e-9:
            return StepperOutput(gate_index, PulseCmd(False, "coast", 0.0, 0.0, 0.0, 0.0, 0.0))

        T_p = self.T_p_frac * max(t_arrival - t, 1e-6)
        t1 = max(t, t_arrival - self.phase_advance_s)
        t0 = t1 - T_p
        envelope = self.pm_envelope if k_lin > 1e-15 else "sqrt_rcos"
        return StepperOutput(gate_index,
                              PulseCmd(True, "pump", t0, t1, T_p, i_peak, dE_cmd, envelope))

    def _end_of_travel(self, t_arrival: float) -> StepperOutput:
        last = len(self.p.coils) - 1
        self.mode = DONE
        if self.p.end_of_travel != "brake_hold":
            # "coast": no braking, but the last coil's PM branch (if any) still gets its
            # departure-repel kick -- same free extra thrust every other station got, not
            # just going silent. Falls back to true silence if there's no PM branch here
            # (_fire_departure returns _COAST when _departure_est was never set, i.e. every
            # station up to this one was pure reluctance -- can't repel at all).
            return self._fire_departure(last, t_arrival)
        # Size the brake from the slug's actual kinetic energy at the last gate (same
        # K_pump-based inversion as a pump pulse) rather than always firing at i_max --
        # a single last-station pulse can only remove a finite, calibrated amount of energy.
        E_k = 0.5 * self.p.mass_kg * self._final_v_hat * self._final_v_hat
        k_quad, k_lin = self.K_pump[last]
        i_peak = _i_peak_for_energy(E_k, k_quad, k_lin, self.i_max)
        envelope = self.pm_envelope if k_lin > 1e-15 else "sqrt_rcos"
        return StepperOutput(last, PulseCmd(True, "brake", t_arrival, t_arrival + self._final_T_p,
                                             self._final_T_p, i_peak, -E_k, envelope))
