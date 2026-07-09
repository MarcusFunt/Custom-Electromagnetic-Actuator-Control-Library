"""Position/velocity estimator for the linear one-way stepper (docs/DESIGN_LINEAR.md).

Linear analog of estimator.py's Tier1Estimator, but there is no restoring term to decay
toward, so between gate crossings we dead-reckon via a locally constant-velocity coast
rather than a decaying sinusoid. Gate order must be strictly increasing: a one-way stepper
has no direction ambiguity to resolve, so there is no analog of Tier1Estimator's
alternating-parity trick -- an out-of-order or repeated gate index is an anomaly, not a
signal to interpret.

Confidence/stall model directly implements the docs/DESIGN.md section 6 caveat that
gates-between-magnets give excellent direction+speed AT RUNNING SPEED but nothing at low
speed/stall: if the next expected gate is overdue, status degrades to STALL_SUSPECT rather
than continuing to trust a dead-reckoned guess.
"""

from __future__ import annotations

import math
from typing import Sequence

SEARCHING = "SEARCHING"
TRACKING = "TRACKING"
STALL_SUSPECT = "STALL_SUSPECT"


class LinearStepperEstimator:
    def __init__(self, gate_positions: Sequence[float], gate_widths: Sequence[float],
                 stall_factor: float = 2.5, tau_confidence: float = 0.05):
        self.gate_positions = list(gate_positions)
        self.gate_widths = list(gate_widths)
        self.stall_factor = stall_factor
        self.tau_confidence = tau_confidence
        self.reset()

    def reset(self):
        self.have = False
        self.next_expected = 0     # next gate index we expect to fire
        self.t_last = 0.0          # time of the last accepted gate crossing
        self.x_last = 0.0          # position pinned at that gate (its known location)
        self.v_last = 0.0
        self.dt_last = None        # time since the previous accepted crossing, if any
        self.direction = 1         # always forward -- one-way stepper, not inferred
        self.status = SEARCHING
        self.n = 0

    def on_gate(self, gate_index: int, t: float, pulse_width: float, pulsed: bool = False) -> bool:
        """Fold in a crossing of gate[gate_index]. Returns False (and leaves state
        untouched except flagging STALL_SUSPECT) if gate_index isn't the one we expected --
        a skipped, repeated, or out-of-order index has no alternating-parity fallback here.
        `pulsed` is accepted for symmetry with Tier1Estimator's contract; unlike the
        pendulum's damping estimate, nothing here is currently conditioned on it (see
        docs/DESIGN_LINEAR.md's noted simplification: no damping feed-forward yet)."""
        if gate_index != self.next_expected:
            self.status = STALL_SUSPECT
            return False

        x_gate = self.gate_positions[gate_index]
        v = self.gate_widths[gate_index] / pulse_width

        if self.have:
            dt = t - self.t_last
            if dt > 1e-9:
                self.dt_last = dt

        self.t_last = t
        self.x_last = x_gate
        self.v_last = v
        self.have = True
        self.status = TRACKING
        self.n += 1
        self.next_expected = gate_index + 1     # == len(gate_positions) once the last gate has fired
        return True

    def predict(self, t: float):
        """Dead-reckon (x, v) via constant-velocity coasting from the last gate anchor."""
        if not self.have:
            return self.x_last, self.v_last
        dt = t - self.t_last
        return self.x_last + self.v_last * dt, self.v_last

    def time_to_reach(self, x_target: float):
        """Predicted time at which the dead-reckoned position reaches x_target (constant-
        velocity extrapolation from the last gate anchor), or None if there isn't yet a
        usable velocity estimate. Depends only on the anchor state, not on "now"."""
        if not self.have or abs(self.v_last) < 1e-9:
            return None
        return self.t_last + (x_target - self.x_last) / self.v_last

    def t_next_gate(self):
        """Predicted time of the next expected gate crossing -- linear analog of
        Tier1Estimator.t_next_bottom(). None once the last gate has already fired."""
        if self.next_expected >= len(self.gate_positions):
            return None
        return self.time_to_reach(self.gate_positions[self.next_expected])

    def update_status(self, t: float) -> None:
        """Call periodically (e.g. every control tick) to detect an overdue gate -- the
        'nothing at low speed/stall' case from docs/DESIGN.md section 6."""
        if self.status != TRACKING:
            return
        t_next = self.t_next_gate()
        if t_next is None:
            return
        margin = self.stall_factor * max(self.dt_last or (t_next - self.t_last), 1e-6)
        if t > t_next + margin:
            self.status = STALL_SUSPECT

    def confidence(self, t: float) -> float:
        """0..1, matching the DESIGN.md section 2.7 pattern: staleness * status factor."""
        c_time = math.exp(-(t - self.t_last) / self.tau_confidence) if self.have else 0.0
        c_status = {TRACKING: 1.0, STALL_SUSPECT: 0.3, SEARCHING: 0.1}[self.status]
        return c_time * c_status

    def cleared_last_gate(self) -> bool:
        return self.next_expected >= len(self.gate_positions)
