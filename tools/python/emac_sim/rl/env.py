"""CoilgunEnv -- a Gymnasium environment for learning to drive the many-stage PM coilgun.

The agent commands per-coil CURRENT TARGETS every control step; the existing electrical model
(`linear_plant.coil_current_step`, with motional back-EMF and per-coil thermal) executes them,
and the mechanics/gates/estimator are the same as `linear_sim.LinearSimulator`. The controller
sees only the IR-BEAM-BREAK estimate of the slug (dead-reckoned position + last-gate velocity),
not ground truth -- a genuine POMDP.

Tractability comes from a LOCAL, translation-invariant window: the action is the current target
for the `window` coils straddling the slug (behind / current / ahead), and the observation is
expressed relative to the slug -- so one policy applies at every stage and generalizes to any
stage count. Reward per control step is `dKE - lam*dE_elec` (telescoping to
`KE_final - lam*E_in`), normalized; sweeping `lam` traces the speed/efficiency Pareto frontier.
"""
from __future__ import annotations

import math

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover - gymnasium is an optional dep
    raise ImportError("CoilgunEnv needs gymnasium; install with `pip install -e .[rl]`") from exc

from .. import linear_plant
from ..linear_estimator import LinearStepperEstimator
from ..numerics import hermite_event_fraction
from .geometry import CoilgunSpec, build_params


class CoilgunEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, spec: CoilgunSpec = CoilgunSpec(), force_lut=None, lam: float = 0.0,
                 sensing: str = "beam_break", window: int = 3, control_every: int = 8,
                 dt: float = 2e-4, t_max: float = 1.5, v0_range=(0.5, 2.0),
                 v_ref: float = 20.0, seed: int | None = None):
        super().__init__()
        self.cfg = spec
        self.p = build_params(spec, force_lut=force_lut)
        self.lam = float(lam)
        assert sensing in ("beam_break", "perfect")
        self.sensing = sensing
        self.window = int(window)                       # coils controlled/observed around slug
        self.control_every = int(control_every)         # plant ticks per agent action
        self.dt = float(dt)
        self.t_max = float(t_max)
        self.v0_range = v0_range
        self.i_max = float(spec.i_max_a)
        self.pitch = spec.pitch_m
        self.n_coils = spec.n_coils
        self.mass = self.p.mass_kg
        self.e_ref = 0.5 * self.mass * v_ref * v_ref    # reward normalizer
        self._min_x_c = min((c.x_c for c in self.p.coils), default=1.0)
        self._exit_x = (self.n_coils - 1) * self.pitch + 0.5 * self.cfg.coil_length_m + 0.02

        # action: current target in [-1,1] (scaled to +/-i_max) for each window coil
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.window,), dtype=np.float32)
        # obs: [v_hat, frac_in_pitch, coils_remaining, time_since_gate, confidence, searching,
        #       i_window(window), temp_window(window)] -- all normalized, so finite bounds hold
        obs_dim = 6 + 2 * self.window
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32)
        self._seed = seed

    # --------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)                         # seeds gymnasium's self.np_random
        p = self.p
        self.t = 0.0
        self.v0 = float(self.np_random.uniform(*self.v0_range))
        self.x = -0.5 * self.pitch - 0.001          # just before gate 0 / coil 0
        self.v = self.v0
        self.currents = [0.0] * self.n_coils
        self.temps = [p.ambient_temperature_c] * self.n_coils
        self.est = LinearStepperEstimator([g.position_m for g in p.gates], [g.w_eff for g in p.gates])
        self.next_gate = 0
        self.e_in = 0.0                              # cumulative source energy (J)
        self.ke = 0.5 * self.mass * self.v * self.v
        self.n_steps = 0
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(self.window)
        targets = self._targets_from_action(action)
        ke0 = self.ke
        e0 = self.e_in
        terminated = truncated = False
        reason = ""
        for _ in range(self.control_every):
            self._plant_tick(targets)
            self.t += self.dt
            self.est.update_status(self.t)
            if self.x >= self._exit_x:
                terminated = True; reason = "exit"; break
            if self.x < -2.0 * self.pitch:
                terminated = True; reason = "fell_back"; break       # slid out the breech
            if self.t >= self.t_max:
                truncated = True; reason = "timeout"; break
        self.ke = 0.5 * self.mass * self.v * self.v
        self.n_steps += 1

        # Reward = the objective itself, densely: change in kinetic energy minus lambda*electrical
        # energy this step (telescopes to KE_final - lam*E_in). NO exploration-punishing penalty
        # -- a do-nothing policy just accrues negative dKE as damping/pull-back bleeds speed, so it
        # is strictly worse than accelerating, without a cliff that scares the policy off exploring.
        d_ke = self.ke - ke0
        d_e = self.e_in - e0
        reward = (d_ke - self.lam * d_e) / self.e_ref
        if terminated and reason == "exit":
            reward += 0.5 * (self.v / math.sqrt(2 * self.e_ref / self.mass)) ** 2  # small exit bonus
        info = {"reason": reason, "v": self.v, "x": self.x, "t": self.t,
                "e_in": self.e_in, "ke": self.ke,
                "efficiency": self.ke / self.e_in if self.e_in > 1e-9 else 0.0}
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    # --------------------------------------------------------------- internals
    def _nearest_coil(self, x_ref: float) -> int:
        return int(round(x_ref / self.pitch))

    def _window_indices(self, x_ref: float):
        """The `window` coil indices straddling x_ref (behind..current..ahead), None where off
        the ends of the tube. Slot order is fixed (behind first) for translation invariance."""
        c = self._nearest_coil(x_ref)
        half = self.window // 2
        idx = []
        for s in range(-half, -half + self.window):
            k = c + s
            idx.append(k if 0 <= k < self.n_coils else None)
        return idx

    def _controller_x(self) -> float:
        """The position the CONTROLLER acts on -- the beam-break estimate (dead-reckoned) in
        'beam_break' mode, or the true position in 'perfect' mode."""
        if self.sensing == "perfect":
            return self.x
        x_hat, _ = self.est.predict(self.t)
        return x_hat if self.est.have else self.x

    def _targets_from_action(self, action):
        targets = [0.0] * self.n_coils
        idx = self._window_indices(self._controller_x())
        for slot, k in enumerate(idx):
            if k is not None:
                targets[k] = float(np.clip(action[slot], -1.0, 1.0)) * self.i_max
        return targets

    def _plant_tick(self, targets):
        p, dt = self.p, self.dt
        thermal = p.thermal_model
        resistances = ([linear_plant.coil_resistance(p.coils[k], self.temps[k])
                        for k in range(self.n_coils)] if thermal else None)
        for k in range(self.n_coils):
            e_bemf = linear_plant.coil_force_gradient(
                p.coils[k], self.x - p.coils[k].position_m, self.currents[k]) * self.v
            r_over = resistances[k] if resistances is not None else None
            i_old = self.currents[k]
            i_new, v_app = linear_plant.coil_current_step(
                i_old, targets[k], p.coils[k], p.bus_voltage_v, dt, bipolar=True,
                resistance_ohm_override=r_over, back_emf_v=e_bemf, return_voltage=True)
            i_new = max(-self.i_max, min(self.i_max, i_new))          # driver current limit
            # source energy delivered this tick (signed: <0 when the H-bridge regenerates)
            self.e_in += v_app * 0.5 * (i_old + i_new) * dt
            self.currents[k] = i_new
        if resistances is not None:
            for k in range(self.n_coils):
                self.temps[k] = linear_plant.coil_temperature_step(
                    self.temps[k], self.currents[k], resistances[k], p.coils[k],
                    p.ambient_temperature_c, dt)
        # CFL-subdivided mechanical step (same guard as LinearSimulator.run)
        n_sub = max(1, math.ceil(abs(self.v) * dt / (0.05 * self._min_x_c)))
        sub_dt = dt / n_sub
        x_n, v_n = self.x, self.v
        for _ in range(n_sub):
            x_n, v_n = linear_plant.step(x_n, v_n, self.currents, sub_dt, p)
        # gate crossings (possibly several) -> beam-break estimator
        while self.next_gate < len(p.gates) and self.x < p.gates[self.next_gate].position_m <= x_n:
            x_gate = p.gates[self.next_gate].position_m
            frac, v_cross = hermite_event_fraction(self.x, self.v, x_n, v_n, dt, y_event=x_gate)
            if abs(v_cross) <= 1e-6:
                break
            t_cross = self.t + frac * dt
            pulse_width = p.gates[self.next_gate].w_eff / abs(v_cross)
            pulsed = any(self.currents[k] != 0.0 for k in range(self.n_coils))
            self.est.on_gate(self.next_gate, t_cross, pulse_width, pulsed=pulsed)
            self.next_gate += 1
        self.x, self.v = x_n, v_n

    def _obs(self):
        x_ref = self._controller_x()
        if self.sensing == "perfect":
            v_hat = self.v
        else:
            _, v_hat = self.est.predict(self.t)
            if not self.est.have:
                v_hat = 0.0
        c = self._nearest_coil(x_ref)
        frac = (x_ref - c * self.pitch) / self.pitch          # ~[-0.5, 0.5] within a pitch
        coils_remaining = max(0, self.n_coils - 1 - c) / self.n_coils
        t_since = (self.t - self.est.t_last) if self.est.have else 0.0
        conf = self.est.confidence(self.t)
        searching = 0.0 if self.est.have else 1.0
        idx = self._window_indices(x_ref)
        i_win = [(self.currents[k] / self.i_max) if k is not None else 0.0 for k in idx]
        t_win = [((self.temps[k] - self.p.ambient_temperature_c) / 100.0) if k is not None else 0.0
                 for k in idx]
        obs = [v_hat / 20.0, frac, coils_remaining, min(t_since / 0.01, 5.0), conf, searching]
        obs += i_win + t_win
        return np.clip(np.asarray(obs, dtype=np.float32), -10.0, 10.0)


def make_env(spec: CoilgunSpec = CoilgunSpec(), force_lut=None, lam: float = 0.0, **kw):
    """Convenience factory (also the SB3 env_fn target)."""
    return CoilgunEnv(spec=spec, force_lut=force_lut, lam=lam, **kw)
