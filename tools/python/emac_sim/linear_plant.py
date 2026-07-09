"""Physical plant: a permanent-magnet linear actuator -- a slug with NO iron, just a
magnet, sliding in a tube past N air-core coil stations. Linear analog of plant.py's
pendulum (see docs/DESIGN_LINEAR.md): drop the gravity restoring term, replace angle/
inertia with position/mass, and sum the same odd, zero-at-its-own-center q_shape(.)
coupling lobe across every coil station instead of a single lobe at the bottom.

Each coil's force is the sum of two terms, both reused unchanged from plant.py -- kept
as TWO terms rather than collapsing to one, because a slug's construction is a config
choice, not a fixed assumption:
  - f_current  (reluctance branch): attract-only, quadratic-with-saturation, from a
    ferromagnetic slug body. `CoilStation.Cmag` defaults to 0.0 -- this build's slug has
    NO iron, only the magnet, so this term is off by default. Coil-core saturation was
    never part of this term regardless (the coils are air-core, so their own B-H curve is
    linear; `i_sat` models the reluctance PATH's -- the slug's iron, if any -- saturating).
  - f_current_pm (PM branch): linear and SIGNED, from the slug's magnet interacting with
    the coil's field -- i>0 attracts, i<0 repels (docs/DESIGN.md 3.3). With Cmag=0.0, this
    is the ONLY force term, matching docs/DESIGN.md's "PM bob" branch exactly.
Both use the same q_shape(.) coupling anchor (each coil's own position) as a shared
simplification -- real coupling profiles for the two mechanisms could differ, but there's
no calibration data yet to justify separating them.

CoilStation exposes the same `.Cmag` / `.i_sat` field names PendulumParams does, so
f_current's duck-typed contract is satisfied without touching its signature (it is public
API, used directly by tests).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .plant import f_current, f_current_pm, q_shape, rl_current_step


@dataclass(frozen=True)
class CoilStation:
    position_m: float
    x_c: float = 0.020     # m, coupling half-width -- linear analog of theta_c
    # Reluctance branch (see module docstring): 0.0 assumes an iron-free, pure-PM slug.
    # Set nonzero to model a hybrid (ferromagnetic + embedded magnet) slug instead.
    Cmag: float = 0.0      # N per A^2 -- same field name as PendulumParams (f_current duck-types on it)
    i_sat: float = 6.0     # A, saturation current (of the slug's iron reluctance path, if any)
    # PM-branch gain (N/A, linear/signed) from the slug's magnet -- see module docstring.
    # A placeholder pending real calibration, like every other constant in this model.
    k_a: float = 0.20
    # Electrical properties of the winding itself (air-core: R, L are constant, no
    # saturation-driven inductance drop). See LinearActuatorParams.current_loop for how
    # these get used -- ignored entirely under the "ideal" (instantaneous current) mode.
    resistance_ohm: float = 1.2   # ohm
    inductance_h: float = 0.004   # H


@dataclass(frozen=True)
class GateStation:
    position_m: float
    w_eff: float = 0.004   # m, effective blocked width -- linear analog of dalpha


def default_coil_stations(pitch: float = 0.05, n: int = 5, **kwargs) -> tuple[CoilStation, ...]:
    return tuple(CoilStation(position_m=k * pitch, **kwargs) for k in range(n))


def default_gate_stations(pitch: float = 0.05, n_coils: int = 5, **kwargs) -> tuple[GateStation, ...]:
    """One entry gate before coil 0, then one gate at the midpoint of each adjacent coil
    pair -- n_coils gates for n_coils coils. See docs/DESIGN_LINEAR.md section 2 for why the
    boundary gate goes at entry (bootstrap confirmation) rather than exit."""
    positions = [-0.5 * pitch] + [(k + 0.5) * pitch for k in range(n_coils - 1)]
    return tuple(GateStation(position_m=x, **kwargs) for x in positions)


@dataclass
class LinearActuatorParams:
    mass_kg: float = 0.20
    damping_n_per_mps: float = 0.05     # direct viscous coeff -- no restoring term to derive a Q from
    coils: tuple[CoilStation, ...] = field(default_factory=default_coil_stations)
    gates: tuple[GateStation, ...] = field(default_factory=default_gate_stations)
    end_of_travel: str = "coast"        # "coast" | "brake_hold"
    # Constant forward force from a pressurized reservoir behind the slug (N), independent
    # of any coil current. Default 0.0 reproduces the unpressurized model exactly. See
    # docs/DESIGN_LINEAR.md's "pressurized tube" section: this is what removes the one
    # genuinely fragile case in the coil-only design -- a slug resting EXACTLY at a coil's
    # center has zero force from that coil at any current (q_shape(0, x_c) == 0), so firing
    # it does nothing; a nonzero bias guarantees the net force there is nonzero regardless,
    # so the slug can never be stuck in true static equilibrium. Modeled as constant rather
    # than an ideal-gas P*V=const decay with position -- a reasonable first approximation
    # for a regulated supply or a large reservoir; a decaying-pressure model is a natural
    # refinement if you're modeling a small fixed-volume charge instead.
    pressure_bias_n: float = 0.0
    # Electrical model. "ideal" (default) treats current as instantaneously commandable --
    # the pre-inductance behavior, and still the right choice for fast/simple sims. "rl"
    # integrates each coil's actual current from its own RL circuit (see
    # coil_current_step()), driven by a hysteretic bang-bang voltage tracking whatever the
    # supervisor's raised-cosine profile asks for -- current then lags the ideal profile by
    # roughly the coil's L/R time constant, including a nonzero decay tail after a "hard
    # cut" (there is no negative supply rail in this unipolar reference driver to force it
    # to zero faster). See docs/DESIGN_LINEAR.md's electrical-dynamics section.
    current_loop: str = "ideal"           # "ideal" | "rl"
    bus_voltage_v: float = 12.0
    # False (default): a single half-bridge driver, matching the soft-iron reference
    # build -- can only ever source i>=0. True: an H-bridge, able to source negative
    # current too. Only matters under current_loop="rl" -- "ideal" mode has no hardware
    # model at all, so a signed target is achieved instantly either way. Repel-pumping
    # (StepperSupervisor's departure-side thrust, docs/DESIGN_LINEAR.md) needs this True
    # to actually take effect under "rl"; under "ideal" it works regardless.
    driver_bipolar: bool = False


def net_force(x: float, currents: Sequence[float], p: LinearActuatorParams) -> float:
    """Sum of every energized coil's local, odd, zero-at-its-own-center coupling toward
    that coil's position -- reluctance term (f_current) plus PM term (f_current_pm),
    added together per coil -- plus the constant pressure bias (if any). Reuses
    q_shape/f_current/f_current_pm verbatim -- only the anchor point (each coil's own
    position_m) differs from the pendulum's single anchor at theta=0."""
    total = p.pressure_bias_n
    for coil, i_k in zip(p.coils, currents):
        if i_k == 0.0:
            continue
        q = q_shape(x - coil.position_m, coil.x_c)
        total += q * (f_current(i_k, coil) + f_current_pm(i_k, coil.k_a))
    return total


def accel(x: float, v: float, currents: Sequence[float], p: LinearActuatorParams) -> float:
    return (net_force(x, currents, p) - p.damping_n_per_mps * v) / p.mass_kg


def step(x: float, v: float, currents: Sequence[float], dt: float, p: LinearActuatorParams):
    """One semi-implicit (symplectic) Euler step -- same integrator and rationale as
    plant.step(): explicit Euler would inject energy into what is otherwise a lightly
    damped, momentum-conserving slide."""
    a = accel(x, v, currents, p)
    v_new = v + a * dt
    x_new = x + v_new * dt
    return x_new, v_new


def coil_current_step(i_actual: float, i_target: float, coil: CoilStation,
                       bus_voltage_v: float, dt: float, bipolar: bool = False) -> float:
    """Advance one coil's ACTUAL current toward i_target over one control tick, through
    its own RL circuit rather than snapping to the target instantly. Modeled as an
    idealized current-mode PWM controller: solve for the exact CONSTANT voltage that would
    drive current from i_actual to i_target over exactly this dt (inverting
    rl_current_step's own closed form), then clamp that to what the driver rail can
    actually supply -- [0, bus_voltage_v] for a single half-bridge (bipolar=False) or
    [-bus_voltage_v, bus_voltage_v] for an H-bridge (bipolar=True). When the exact voltage
    is within the rail, this reaches i_target exactly after one step, at ANY dt, with no
    overshoot; when it isn't, applying the rail limit for the whole step is the correctly
    rail-limited (not overshooting) response.

    This replaced a cruder two-state (full-on/full-off) bang-bang model. Bang-bang chatters
    around the target rather than settling, which is a minor cosmetic issue normally -- but
    for a LOW-resistance coil under a HIGH bus voltage (few turns, thick winding: a real,
    reachable combination once turns/dimensions/voltage are all free design knobs), the
    bang-bang steady-state current V/R can be orders of magnitude above the intended
    target. Discovered via a dt-refinement check during development: reported exit speed
    for such a design kept DROPPING as dt shrank instead of converging (no amount of
    sub-stepping fixed it, because the failure was the control law's steady-state target
    being wrong by orders of magnitude, not an integration-accuracy problem). Solving for
    the exact tracking voltage removes the failure mode outright: it can only ever reach
    for the RIGHT target, rail-limited, never an implied one thousands of times larger.
    bipolar=False means a negative i_target still can't actually be reached (v_min=0), same
    as before -- StepperSupervisor's departure-side repel-pumping still needs bipolar=True
    to take effect under "rl" (under "ideal", with no hardware model at all, repel already
    works regardless of this flag).
    """
    r, l = coil.resistance_ohm, coil.inductance_h
    tau = l / r
    decay = math.exp(-dt / tau)
    v_needed = r * (i_target - i_actual * decay) / (1.0 - decay) if decay < 1.0 else 0.0
    v_min = -bus_voltage_v if bipolar else 0.0
    v_applied = max(v_min, min(bus_voltage_v, v_needed))
    return rl_current_step(i_actual, v_applied, r, l, dt)


def kinetic_energy(v: float, p: LinearActuatorParams) -> float:
    return 0.5 * p.mass_kg * v * v
