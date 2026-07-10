"""Closed-loop simulator for the linear one-way stepper (docs/DESIGN_LINEAR.md).

Mirrors sim.py's shape -- true plant stepped at fixed dt -> detect a discrete sensor event
-> feed the sparse estimator -> ask the supervisor to replan -> apply the resulting
current(s) back into the plant -- but the event predicate is an ordered, non-wrapping
gate-position crossing rather than a bottom sign-change, and the estimator/supervisor
contracts differ enough (see linear_supervisor.py's module docstring) that this is a
parallel concrete class, not a shared base with sim.py's Simulator.

Per-coil current is now genuine integrated STATE, not a per-tick idealized computation:
under `p.current_loop == "rl"`, each coil's actual current persists across ticks and is
advanced through its own RL circuit (linear_plant.coil_current_step, an idealized current-
mode PWM controller -- see its docstring for why that replaced a naive bang-bang model),
tracking whatever the supervisor's raised-cosine profile (supervisor.current_at) asks for
with a lag set by that coil's L/R time constant -- including a nonzero decay tail after a
"hard cut", since current can't jump discontinuously. `p.current_loop == "ideal"` (the
default) keeps the pre-inductance behavior: current snaps to the target instantly.

The mechanical step is also CFL-limited (see run()): a fixed dt tuned for the reference
build can silently under-resolve much lighter/faster designs, letting the slug cover more
than a whole coupling half-width (x_c) in one nominal tick and skip clean over a coil's
force lobe. run() subdivides the mechanical integration (holding currents fixed across the
sub-steps -- they evolve on a slower timescale than this failure mode) whenever that would
happen, so the requested dt is a resolution ceiling, not a silent accuracy cliff.

Per-coil winding TEMPERATURE is likewise genuine integrated state, off by default:
`p.thermal_model == True` tracks each coil's own temperature from its i^2*R dissipation
(linear_plant.coil_temperature_step, an exact one-node thermal model -- same exponential-
relaxation shape as the RL circuit above) and feeds the resulting temperature-adjusted
resistance (coil_resistance()) back into "rl" mode's electrical dynamics, so a design that
only looks fast because it never pays a heating penalty stops looking that way. False (the
default) pins every coil at `ambient_temperature_c` forever, exactly reproducing the prior
fixed-resistance behavior.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Sequence

from . import linear_plant
from .linear_estimator import LinearStepperEstimator
from .linear_plant import LinearActuatorParams
from .linear_supervisor import StepperSupervisor
from .numerics import hermite_event_fraction
from .supervisor import current_at


@dataclass
class LinearSimLog:
    t: List[float] = field(default_factory=list)
    x: List[float] = field(default_factory=list)
    v: List[float] = field(default_factory=list)
    active_coil: List[int] = field(default_factory=list)
    active_current: List[float] = field(default_factory=list)
    active_temperature_c: List[float] = field(default_factory=list)
    x_est: List[float] = field(default_factory=list)
    status: List[str] = field(default_factory=list)
    # per-gate records
    gate_t: List[float] = field(default_factory=list)
    gate_index: List[int] = field(default_factory=list)
    gate_v: List[float] = field(default_factory=list)
    supervisor_mode: List[str] = field(default_factory=list)
    # Physical exit-plane event, independent of the final entry-side gate.
    exit_t: float | None = None
    exit_v: float | None = None
    exit_position_m: float | None = None
    # Cumulative reciprocal electromechanical energy ledger (RL mode).
    bus_energy_j: List[float] = field(default_factory=list)
    copper_loss_j: List[float] = field(default_factory=list)
    magnetic_energy_j: List[float] = field(default_factory=list)
    mechanical_em_work_j: List[float] = field(default_factory=list)
    energy_residual_j: List[float] = field(default_factory=list)


class LinearSimulator:
    def __init__(self, p: LinearActuatorParams, est: LinearStepperEstimator,
                 sup: StepperSupervisor, dt: float = 2e-4, sample_every: int = 10,
                 gate_noise_std_s: Sequence[float] | None = None,
                 gate_dropout_probability: Sequence[float] | None = None,
                 random_seed: int = 0):
        self.p = p
        self.est = est
        self.sup = sup
        self.dt = dt
        self.sample_every = sample_every
        self.gate_noise_std_s = list(gate_noise_std_s or [0.0] * len(p.gates))
        self.gate_dropout_probability = list(
            gate_dropout_probability or [0.0] * len(p.gates)
        )
        if (len(self.gate_noise_std_s) != len(p.gates)
                or len(self.gate_dropout_probability) != len(p.gates)):
            raise ValueError("gate noise/dropout arrays must match the gate count")
        self.rng = random.Random(random_seed)

    def run(self, x0: float, v0: float, v_tgt: float | None, t_end: float) -> LinearSimLog:
        p, est, sup = self.p, self.est, self.sup
        log = LinearSimLog()
        n_coils = len(p.coils)
        n_gates = len(p.gates)

        t = 0.0
        x, v = x0, v0
        sup.start(t)
        next_gate = 0
        step_idx = 0
        currents = [0.0] * n_coils    # persistent per-coil electrical state under "rl"
        # Persistent per-coil winding temperature -- stays pinned at ambient (i.e. this
        # array is created but never advanced past its initial value) unless
        # p.thermal_model is True, so the default behavior is bit-for-bit the old
        # fixed-resistance model.
        temperatures = [p.ambient_temperature_c] * n_coils
        min_x_c = min((c.x_c for c in p.coils), default=1.0)
        exit_position = p.resolved_exit_position_m
        cumulative_bus_energy = 0.0
        cumulative_copper_loss = 0.0
        cumulative_mechanical_em_work = 0.0
        cumulative_energy_residual = 0.0

        while t < t_end:
            out = sup.tick(t)
            i_target = current_at(t, out.cmd)

            # Resistance at each coil's CURRENT temperature, computed once per tick and
            # reused for both the electrical step below (if "rl") and the thermal step's
            # own i^2*R dissipation -- keeping the two consistent with what the coil
            # actually saw this tick rather than two independent estimates of it.
            resistances = ([linear_plant.coil_resistance(
                                p.coils[k], temperatures[k], p.ambient_temperature_c)
                            for k in range(n_coils)] if p.thermal_model else None)

            force_currents = [0.0] * n_coils
            mean_square_currents = [0.0] * n_coils
            tick_bus_energy = 0.0
            tick_copper_loss = 0.0
            tick_field_delta = 0.0
            if p.current_loop == "rl":
                for k in range(n_coils):
                    target_k = i_target if k == out.coil_index else 0.0
                    r_override = resistances[k] if resistances is not None else None
                    electrical = linear_plant.coil_electrical_step(
                        currents[k], target_k, p.coils[k], p.bus_voltage_v, self.dt,
                        bipolar=p.driver_bipolar, resistance_ohm_override=r_override,
                        back_emf_v=linear_plant.pm_back_emf_v(x, v, p.coils[k]),
                    )
                    currents[k] = electrical.current_a
                    force_currents[k] = electrical.current_integral_as / self.dt
                    mean_square_currents[k] = electrical.current_sq_integral_a2s / self.dt
                    tick_bus_energy += electrical.bus_energy_j
                    tick_copper_loss += electrical.copper_loss_j
                    tick_field_delta += electrical.field_energy_delta_j
            else:
                currents = [0.0] * n_coils
                if 0 <= out.coil_index < n_coils:
                    i_val = i_target
                    if not p.driver_bipolar and i_val < 0.0:
                        i_val = 0.0     # single-ended driver: no negative rail even under "ideal"
                    currents[out.coil_index] = i_val
                force_currents = list(currents)
                mean_square_currents = [i * i for i in currents]

            if resistances is not None:
                for k in range(n_coils):
                    temperatures[k] = linear_plant.coil_temperature_step_mean_square(
                        temperatures[k], mean_square_currents[k], resistances[k], p.coils[k],
                        p.ambient_temperature_c, self.dt,
                    )

            i_active = currents[out.coil_index] if 0 <= out.coil_index < n_coils else 0.0

            # CFL-like safety: don't let the slug cross more than a small fraction of the
            # finest coupling half-width in one nominal tick, or the discretized force can
            # completely miss (or badly misrepresent) a coil's lobe -- subdivide the
            # mechanical integration accordingly, holding currents fixed across the
            # sub-steps (see module docstring).
            a_start = linear_plant.accel(x, v, force_currents, p)
            displacement_bound = abs(v) * self.dt + 0.5 * abs(a_start) * self.dt * self.dt
            n_sub = max(1, math.ceil(displacement_bound / (0.05 * min_x_c)))
            sub_dt = self.dt / n_sub
            x_n, v_n = x, v
            tick_damping_loss = 0.0
            for _ in range(n_sub):
                v_before = v_n
                x_n, v_n = linear_plant.step(x_n, v_n, force_currents, sub_dt, p)
                tick_damping_loss += (p.damping_n_per_mps
                                      * 0.5 * (v_before * v_before + v_n * v_n) * sub_dt)

            delta_ke = 0.5 * p.mass_kg * (v_n * v_n - v * v)
            pressure_work = p.pressure_bias_n * (x_n - x)
            tick_mechanical_em_work = delta_ke + tick_damping_loss - pressure_work
            cumulative_mechanical_em_work += tick_mechanical_em_work
            if p.current_loop == "rl":
                cumulative_bus_energy += tick_bus_energy
                cumulative_copper_loss += tick_copper_loss
                cumulative_energy_residual += (
                    tick_bus_energy - tick_copper_loss - tick_field_delta
                    - tick_mechanical_em_work
                )

            # A while, not an if: at a coarse enough dt / high enough speed, the slug can
            # cross MORE THAN ONE gate within a single tick. An if here only ever checks
            # the next expected gate once per tick -- any gate after the first crossed
            # in the same tick would be silently skipped forever (x has already moved
            # past it by the next iteration, so `x < x_gate` then reads false), stalling
            # the whole rest of the run's gate sequence. Every crossing this tick is
            # interpolated against the same (x, x_n) bracket.  The interpolation is now a
            # cubic Hermite segment using both endpoint velocities, which is more accurate
            # than the old constant-velocity-within-the-tick approximation.
            while next_gate < n_gates and x < p.gates[next_gate].position_m <= x_n:
                x_gate = p.gates[next_gate].position_m
                frac, v_cross = hermite_event_fraction(x, v, x_n, v_n, self.dt, y_event=x_gate)
                t_cross = t + frac * self.dt
                if abs(v_cross) <= 1e-6:
                    break   # too slow to trust this crossing; retry the same gate next tick
                pulse_width = p.gates[next_gate].w_eff / abs(v_cross)
                noise_std = self.gate_noise_std_s[next_gate]
                dropout = self.gate_dropout_probability[next_gate]
                dropped = self.rng.random() < dropout
                if not dropped:
                    measured_t = t_cross + self.rng.gauss(0.0, noise_std)
                    measured_width = max(1e-9, pulse_width + self.rng.gauss(0.0, noise_std))
                    pulsed = any(abs(i) > 1e-12 for i in currents)
                    if est.on_gate(next_gate, measured_t, measured_width, pulsed=pulsed):
                        sup.on_gate(next_gate, est, measured_t, v_tgt)
                    log.gate_t.append(measured_t)
                    log.gate_index.append(next_gate)
                    log.gate_v.append(v_cross)
                next_gate += 1

            if (log.exit_t is None and x < exit_position <= x_n):
                frac, v_exit = hermite_event_fraction(
                    x, v, x_n, v_n, self.dt, y_event=exit_position
                )
                log.exit_t = t + frac * self.dt
                log.exit_v = v_exit
                log.exit_position_m = exit_position

            x, v = x_n, v_n
            t += self.dt
            est.update_status(t)

            if step_idx % self.sample_every == 0:
                log.t.append(t)
                log.x.append(x)
                log.v.append(v)
                log.active_coil.append(out.coil_index)
                log.active_current.append(i_active)
                t_active = (temperatures[out.coil_index] if 0 <= out.coil_index < n_coils
                            else p.ambient_temperature_c)
                log.active_temperature_c.append(t_active)
                x_e, _ = est.predict(t)
                log.x_est.append(x_e)
                log.status.append(est.status)
                log.supervisor_mode.append(sup.mode)
                log.bus_energy_j.append(cumulative_bus_energy)
                log.copper_loss_j.append(cumulative_copper_loss)
                log.magnetic_energy_j.append(sum(
                    0.5 * coil.inductance_h * current * current
                    for coil, current in zip(p.coils, currents)
                ))
                log.mechanical_em_work_j.append(cumulative_mechanical_em_work)
                log.energy_residual_j.append(
                    cumulative_energy_residual if p.current_loop == "rl" else math.nan
                )
            step_idx += 1

        return log
