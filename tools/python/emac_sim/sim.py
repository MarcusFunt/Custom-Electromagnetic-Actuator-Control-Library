"""Closed-loop simulator.

Integrates the TRUE plant, detects bottom crossings (theta -> 0), synthesizes a
Crossing event (timestamp + beam-block pulse width) exactly as the photogate would,
feeds it to the estimator, asks the supervisor to plan the next pulse, and applies
that pulse's current back into the plant. Nothing but the crossing events crosses the
boundary from plant to controller -- this is the event-first architecture in miniature.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, List
from . import plant
from .plant import PendulumParams
from .estimator import Tier1Estimator
from .numerics import hermite_event_fraction
from .supervisor import EnergySupervisor, PulseCmd, current_at


@dataclass
class SimLog:
    t: List[float] = field(default_factory=list)
    theta: List[float] = field(default_factory=list)
    omega: List[float] = field(default_factory=list)
    i: List[float] = field(default_factory=list)
    E_true: List[float] = field(default_factory=list)
    # reconstruction trace (estimator's dead-reckoned theta at the same sample times)
    theta_est: List[float] = field(default_factory=list)
    # per-crossing records
    cx_t: List[float] = field(default_factory=list)
    cx_A_peak: List[float] = field(default_factory=list)   # physical peak of the half-swing that just ended
    cx_A_energy: List[float] = field(default_factory=list) # true amplitude implied by crossing energy
    cx_A_est: List[float] = field(default_factory=list)    # estimator amplitude from pulse width
    cx_E_true: List[float] = field(default_factory=list)
    cx_E_est: List[float] = field(default_factory=list)
    cx_A_tgt: List[float] = field(default_factory=list)
    cx_kind: List[str] = field(default_factory=list)
    cx_ipeak: List[float] = field(default_factory=list)


class Simulator:
    def __init__(self, p: PendulumParams, est: Tier1Estimator,
                 sup: EnergySupervisor, dt: float = 2e-4, sample_every: int = 10):
        self.p = p
        self.est = est
        self.sup = sup
        self.dt = dt
        self.sample_every = sample_every

    def run(self, theta0: float, omega0: float,
            target_E: Callable[[float], float], t_end: float) -> SimLog:
        p, est, sup = self.p, self.est, self.sup
        log = SimLog()

        t = 0.0
        theta, omega = theta0, omega0
        sched = PulseCmd(False, "coast", 0.0, 0.0, 0.0, 0.0, 0.0)
        peak_since_cx = abs(theta)
        pulsed_since_cx = False       # did any coil current flow this half-swing?
        last_cross_t = -1e9           # for debounce (reject spurious double-crossings)
        min_gap = 0.30 * (math.pi / p.omega0)   # ~0.3 of a nominal half-period
        step_idx = 0

        while t < t_end:
            i = current_at(t, sched)
            if i > 0.0:
                pulsed_since_cx = True
            theta_n, omega_n = plant.step(theta, omega, i, self.dt, p)

            # --- bottom-crossing detection (sign change of theta) ---
            crossed = (theta != 0.0) and (
                (theta > 0.0 and theta_n <= 0.0) or (theta < 0.0 and theta_n >= 0.0)
            )
            if crossed:
                frac, omega_cross = hermite_event_fraction(theta, omega, theta_n, omega_n, self.dt)
                t_cross = t + frac * self.dt
                v = abs(omega_cross)
                accepted = False
                if v > 1e-6 and (t_cross - last_cross_t) >= min_gap:
                    accepted = True
                    last_cross_t = t_cross
                    pw = p.dalpha / v                      # synthetic beam-block width
                    est.on_crossing(t_cross, pw, pulsed=pulsed_since_cx)
                    E_tgt = target_E(t_cross)
                    sched = sup.plan(est, E_tgt)
                    E_cross = plant.energy(0.0, omega_cross, p)

                    # log this crossing (the swing that just ended peaked at peak_since_cx)
                    log.cx_t.append(t_cross)
                    log.cx_A_peak.append(peak_since_cx)
                    log.cx_A_energy.append(plant.amplitude_from_energy(E_cross, p))
                    log.cx_A_est.append(est.amplitude())
                    log.cx_E_true.append(E_cross)
                    log.cx_E_est.append(est.energy())
                    log.cx_A_tgt.append(plant.amplitude_from_energy(E_tgt, p))
                    log.cx_kind.append(sched.kind)
                    log.cx_ipeak.append(sched.i_peak)
                if accepted:
                    peak_since_cx = 0.0
                    pulsed_since_cx = False

            theta, omega = theta_n, omega_n
            t += self.dt
            peak_since_cx = max(peak_since_cx, abs(theta))

            if step_idx % self.sample_every == 0:
                log.t.append(t)
                log.theta.append(theta)
                log.omega.append(omega)
                log.i.append(i)
                log.E_true.append(plant.energy(theta, omega, p))
                th_e, _ = est.predict(t) if est.have else (0.0, 0.0)
                log.theta_est.append(th_e)
            step_idx += 1

        return log
