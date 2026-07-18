"""Build the fixed many-stage PM coilgun the RL controller drives.

All stages are IDENTICAL small coils (only position differs), so a single position-independent
force law serves every coil -- either the analytic q_shape lobe from `coil_design.
build_coil_station` (fast, default) or one swept `ForceLUT` (real coupling shape) attached to
every coil. The driver is a per-stage H-bridge at the chosen bus voltage; the per-coil current
target is what the controller (RL policy or a baseline) commands each tick, clamped to +/-i_max.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from .. import coil_design
from ..linear_plant import GateStation, LinearActuatorParams


@dataclass(frozen=True)
class CoilgunSpec:
    """The knobs that define the fixed gun (geometry + driver). i_max_a is a DRIVER/control
    limit (the action range), not a plant field -- it is carried here and handed to the env."""
    n_coils: int = 16
    coil_length_m: float = 0.012        # small stages
    radial_thickness_m: float = 0.006
    turns: int = 450
    magnet_radius_m: float = 0.004      # small, light PM slug (the sweep's fast corner)
    magnet_length_m: float = 0.014
    remanence_t: float = 1.25
    bore_clearance_m: float = 0.0015
    bus_voltage_v: float = 450.0        # H-bridge rail
    i_max_a: float = 100.0              # per-stage peak current (action clamp)
    gate_w_eff: float = 0.002           # IR beam-break effective width
    thermal_model: bool = True

    @property
    def pitch_m(self) -> float:
        return self.coil_length_m       # edge-to-edge stacking

    @property
    def tube_length_m(self) -> float:
        return self.n_coils * self.coil_length_m


def build_params(spec: CoilgunSpec = CoilgunSpec(), force_lut=None) -> LinearActuatorParams:
    """Assemble the LinearActuatorParams: n identical coils (winding R/L/k_a/x_c/thermal from
    coil_design), a gate (IR beam-break) before coil 0 and at each inter-coil midpoint, and the
    H-bridge "rl" driver at spec.bus_voltage_v. Pass `force_lut` (a callable / ForceLUT) to
    override the analytic lobe with a swept table on every coil."""
    pitch = spec.pitch_m
    coils = tuple(
        coil_design.build_coil_station(
            position_m=k * pitch, turns=spec.turns, coil_length_m=spec.coil_length_m,
            radial_thickness_m=spec.radial_thickness_m, magnet_radius_m=spec.magnet_radius_m,
            magnet_length_m=spec.magnet_length_m, remanence_t=spec.remanence_t,
            bore_clearance_m=spec.bore_clearance_m,
        )
        for k in range(spec.n_coils)
    )
    if force_lut is not None:
        coils = tuple(dataclasses.replace(c, force_lut=force_lut) for c in coils)
    # Gate before coil 0 (entry / bootstrap confirm), then one at each adjacent-coil midpoint --
    # n_coils gates for n_coils coils, matching linear_plant.default_gate_stations().
    gate_positions = [-0.5 * pitch] + [(k + 0.5) * pitch for k in range(spec.n_coils - 1)]
    gates = tuple(GateStation(position_m=x, w_eff=spec.gate_w_eff) for x in gate_positions)
    return LinearActuatorParams(
        mass_kg=coil_design.magnet_mass_kg(spec.magnet_radius_m, spec.magnet_length_m),
        coils=coils, gates=gates, current_loop="rl", bus_voltage_v=spec.bus_voltage_v,
        driver_bipolar=True, thermal_model=spec.thermal_model, ambient_temperature_c=20.0,
    )


def build_reference_lut(spec: CoilgunSpec = CoilgunSpec(), n_offsets: int = 41,
                        n_currents: int = 3):
    """A swept `ForceLUT` for one coil/slug of this geometry from the analytic reference backend
    (no FEMM) -- the real (non-Gaussian) coupling shape, and PM force is linear in current so 3
    current points are exact. For a REAL-FEMM table use emac-femgen / study_lib.build_femm_lut
    and pass the result as `force_lut`; the geometry is position-independent so ONE table serves
    every stage."""
    import numpy as np
    from ..fem.geometry import CoilWindingGeometry, SlugGeometry
    from ..fem.reference_backend import AnalyticReferenceBackend
    from ..fem.sweep import sweep_coil

    slug = SlugGeometry(spec.magnet_radius_m, spec.magnet_length_m, spec.remanence_t)
    coil = CoilWindingGeometry(0.0, spec.turns, spec.coil_length_m, spec.radial_thickness_m,
                               bore_clearance_m=spec.bore_clearance_m)
    scale = 1.5 * spec.coil_length_m + 0.5 * spec.magnet_length_m
    offsets = np.linspace(-3.0 * scale, 3.0 * scale, n_offsets)
    currents = np.linspace(-spec.i_max_a, spec.i_max_a, n_currents)
    return sweep_coil(coil, slug, AnalyticReferenceBackend(), offsets_m=offsets, currents_a=currents)
