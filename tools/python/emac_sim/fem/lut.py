"""ForceLUT: a saved (offset, current) -> force grid, and edge-clamped bilinear
interpolation over it -- the artifact sweep.py produces and linear_plant.net_force
consumes at simulation time (see config.py's LinearCoilConfig.force_lut_path).

Interpolation clamps its query point into the grid's own range rather than extrapolating.
A FEM/analytic sweep is only trustworthy inside the region it actually sampled; letting
RegularGridInterpolator extrapolate linearly past the last sampled current or offset could
silently hand the plant an arbitrarily wrong force for an off-grid operating point (e.g. a
runaway current well past what the sweep covered), which is a worse failure mode than
flatly clamping to the nearest edge value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True, eq=False)
class ForceLUT:
    # eq=False: the default dataclass __eq__ would compare numpy arrays with `==`,
    # which returns an elementwise array rather than a bool and breaks equality checks.
    offsets_m: np.ndarray          # shape (n_offsets,), strictly increasing
    currents_a: np.ndarray         # shape (n_currents,), strictly increasing
    force_n: np.ndarray            # shape (n_offsets, n_currents)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets_m, dtype=float)
        currents = np.asarray(self.currents_a, dtype=float)
        force = np.asarray(self.force_n, dtype=float)
        if offsets.ndim != 1 or currents.ndim != 1:
            raise ValueError("offsets_m and currents_a must be 1-D")
        if force.shape != (offsets.size, currents.size):
            raise ValueError(
                f"force_n shape {force.shape} must be (len(offsets_m), len(currents_a)) "
                f"= ({offsets.size}, {currents.size})"
            )
        if offsets.size < 2 or currents.size < 2:
            raise ValueError("offsets_m and currents_a each need at least 2 points to interpolate")
        if np.any(np.diff(offsets) <= 0.0):
            raise ValueError("offsets_m must be strictly increasing")
        if np.any(np.diff(currents) <= 0.0):
            raise ValueError("currents_a must be strictly increasing")
        object.__setattr__(self, "offsets_m", offsets)
        object.__setattr__(self, "currents_a", currents)
        object.__setattr__(self, "force_n", force)
        object.__setattr__(
            self, "_interp",
            RegularGridInterpolator((offsets, currents), force, method="linear",
                                     bounds_error=False, fill_value=None),
        )

    def __call__(self, offset_m: float, current_a: float) -> float:
        clamped_offset = min(max(offset_m, self.offsets_m[0]), self.offsets_m[-1])
        clamped_current = min(max(current_a, self.currents_a[0]), self.currents_a[-1])
        return float(self._interp([[clamped_offset, clamped_current]])[0])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, offsets_m=self.offsets_m, currents_a=self.currents_a, force_n=self.force_n,
            metadata_json=_dict_to_json(self.metadata),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ForceLUT":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = _json_to_dict(data["metadata_json"]) if "metadata_json" in data else {}
            return cls(
                offsets_m=data["offsets_m"], currents_a=data["currents_a"],
                force_n=data["force_n"], metadata=metadata,
            )


def _dict_to_json(d: Mapping[str, Any]) -> str:
    import json
    return json.dumps(dict(d))


def _json_to_dict(arr: np.ndarray) -> dict:
    import json
    return json.loads(str(arr))
