"""Configuration loading for configurable EMAC virtual hardware simulations.

The simulator intentionally keeps the config format boring and inspectable: TOML in,
small dataclasses out.  The dataclasses are not a full validation framework; they are a
stable boundary between user-editable fictional hardware files and the simulation code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from .plant import PendulumParams


Number = int | float


@dataclass(frozen=True)
class PendulumConfig:
    length_m: float = 0.30
    bob_mass_kg: float = 0.05
    quality_factor: float = 200.0
    initial_angle_rad: float = 0.06
    initial_omega_rad_s: float = 0.0

    def to_params(self, coil: "CoilConfig", gate: "GateConfig") -> PendulumParams:
        return PendulumParams(
            L=self.length_m,
            m=self.bob_mass_kg,
            Q=self.quality_factor,
            theta_c=coil.theta_c_rad,
            Cmag=coil.c_mag_nm_per_a2,
            i_sat=coil.i_sat_a,
            dalpha=gate.angular_width_rad,
        )


@dataclass(frozen=True)
class GateConfig:
    kind: str = "photogate"
    angle_rad: float = 0.0
    angular_width_rad: float = 0.060
    noise_std_s: float = 0.0
    dropout_probability: float = 0.0


@dataclass(frozen=True)
class CoilConfig:
    angle_rad: float = 0.0
    theta_c_rad: float = 0.05
    c_mag_nm_per_a2: float = 0.010
    i_sat_a: float = 8.0
    resistance_ohm: float = 1.2
    inductance_h: float = 0.004
    max_current_a: float = 8.0
    thermal_mass_j_per_k: float = 12.0
    thermal_resistance_k_per_w: float = 8.0


@dataclass(frozen=True)
class DriverConfig:
    bus_voltage_v: float = 12.0
    pwm_frequency_hz: float = 20_000.0
    current_loop: str = "ideal"


@dataclass(frozen=True)
class ControllerConfig:
    kind: str = "energy_supervisor"
    target_amplitude_rad: float = 0.30
    k_energy: float = 0.35
    pulse_width_half_period_fraction: float = 0.30
    hold_deadband_fraction: float = 0.02


@dataclass(frozen=True)
class TargetSegment:
    t_s: float
    amplitude_rad: float


@dataclass(frozen=True)
class SimulationConfig:
    pendulum: PendulumConfig = field(default_factory=PendulumConfig)
    gates: list[GateConfig] = field(default_factory=lambda: [GateConfig()])
    coils: list[CoilConfig] = field(default_factory=lambda: [CoilConfig()])
    driver: DriverConfig = field(default_factory=DriverConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    target_segments: list[TargetSegment] = field(default_factory=list)
    duration_s: float = 22.0
    dt_s: float = 2e-4
    sample_every: int = 10

    @property
    def primary_gate(self) -> GateConfig:
        return self.gates[0]

    @property
    def primary_coil(self) -> CoilConfig:
        return self.coils[0]

    def to_pendulum_params(self) -> PendulumParams:
        return self.pendulum.to_params(self.primary_coil, self.primary_gate)


def default_config() -> SimulationConfig:
    """Return the historical Phase 0 demo configuration."""
    return SimulationConfig(
        target_segments=[
            TargetSegment(0.0, 0.35),
            TargetSegment(8.0, 0.20),
            TargetSegment(15.0, 0.30),
        ]
    )


def load_config(path: str | Path) -> SimulationConfig:
    """Load a TOML simulation config from *path*."""
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return parse_config(raw)


def parse_config(raw: Mapping[str, Any]) -> SimulationConfig:
    pendulum = _pendulum(raw.get("pendulum", {}))
    gates = _gates(raw)
    coils = _coils(raw)
    driver = _driver(raw.get("driver", {}))
    controller = _controller(raw.get("controller", {}))
    sim_raw = _section(raw, "sim")

    return SimulationConfig(
        pendulum=pendulum,
        gates=gates,
        coils=coils,
        driver=driver,
        controller=controller,
        target_segments=_target_segments(raw, controller),
        duration_s=_float(sim_raw, "duration_s", 22.0),
        dt_s=_float(sim_raw, "dt_s", 2e-4),
        sample_every=_int(sim_raw, "sample_every", 10),
    )


def _pendulum(raw: Any) -> PendulumConfig:
    data = _as_mapping(raw, "pendulum")
    return PendulumConfig(
        length_m=_float(data, "length_m", 0.30),
        bob_mass_kg=_float(data, "bob_mass_kg", 0.05),
        quality_factor=_float(data, "quality_factor", 200.0),
        initial_angle_rad=_float(data, "initial_angle_rad", 0.06),
        initial_omega_rad_s=_float(data, "initial_omega_rad_s", 0.0),
    )


def _gates(raw: Mapping[str, Any]) -> list[GateConfig]:
    gate_raw = raw.get("gate", None)
    gates_raw = raw.get("gates", None)
    if gate_raw is not None and gates_raw is not None:
        raise ValueError("Use either [gate]/[[gate]] or [[gates]], not both")
    source = gates_raw if gates_raw is not None else gate_raw
    if source is None:
        return [GateConfig()]
    entries = source if isinstance(source, list) else [source]
    gates = []
    for idx, item in enumerate(entries):
        data = _as_mapping(item, f"gate[{idx}]")
        gates.append(
            GateConfig(
                kind=str(data.get("kind", "photogate")),
                angle_rad=_float(data, "angle_rad", 0.0),
                angular_width_rad=_float(data, "angular_width_rad", 0.060),
                noise_std_s=_float(data, "noise_std_s", 0.0),
                dropout_probability=_float(data, "dropout_probability", 0.0),
            )
        )
    if not gates:
        raise ValueError("At least one gate is required")
    return gates


def _coils(raw: Mapping[str, Any]) -> list[CoilConfig]:
    coil_raw = raw.get("coil", None)
    coils_raw = raw.get("coils", None)
    if coil_raw is not None and coils_raw is not None:
        raise ValueError("Use either [coil]/[[coil]] or [[coils]], not both")
    source = coils_raw if coils_raw is not None else coil_raw
    if source is None:
        return [CoilConfig()]
    entries = source if isinstance(source, list) else [source]
    coils = []
    for idx, item in enumerate(entries):
        data = _as_mapping(item, f"coil[{idx}]")
        coils.append(
            CoilConfig(
                angle_rad=_float(data, "angle_rad", 0.0),
                theta_c_rad=_float(data, "theta_c_rad", 0.05),
                c_mag_nm_per_a2=_float(data, "c_mag_nm_per_a2", 0.010),
                i_sat_a=_float(data, "i_sat_a", 8.0),
                resistance_ohm=_float(data, "resistance_ohm", 1.2),
                inductance_h=_float(data, "inductance_h", 0.004),
                max_current_a=_float(data, "max_current_a", 8.0),
                thermal_mass_j_per_k=_float(data, "thermal_mass_j_per_k", 12.0),
                thermal_resistance_k_per_w=_float(data, "thermal_resistance_k_per_w", 8.0),
            )
        )
    if not coils:
        raise ValueError("At least one coil is required")
    return coils


def _driver(raw: Any) -> DriverConfig:
    data = _as_mapping(raw, "driver")
    return DriverConfig(
        bus_voltage_v=_float(data, "bus_voltage_v", 12.0),
        pwm_frequency_hz=_float(data, "pwm_frequency_hz", 20_000.0),
        current_loop=str(data.get("current_loop", "ideal")),
    )


def _controller(raw: Any) -> ControllerConfig:
    data = _as_mapping(raw, "controller")
    return ControllerConfig(
        kind=str(data.get("kind", "energy_supervisor")),
        target_amplitude_rad=_float(data, "target_amplitude_rad", 0.30),
        k_energy=_float(data, "k_energy", 0.35),
        pulse_width_half_period_fraction=_float(data, "pulse_width_half_period_fraction", 0.30),
        hold_deadband_fraction=_float(data, "hold_deadband_fraction", 0.02),
    )


def _target_segments(raw: Mapping[str, Any], controller: ControllerConfig) -> list[TargetSegment]:
    target_raw = raw.get("target", None)
    if target_raw is None:
        return [TargetSegment(0.0, controller.target_amplitude_rad)]

    if isinstance(target_raw, Mapping):
        if "segments" in target_raw:
            entries = target_raw["segments"]
        else:
            entries = [target_raw]
    else:
        entries = target_raw

    if not isinstance(entries, list):
        raise ValueError("target.segments must be a list")

    segments = []
    for idx, item in enumerate(entries):
        data = _as_mapping(item, f"target.segments[{idx}]")
        segments.append(
            TargetSegment(
                t_s=_float(data, "t_s", 0.0),
                amplitude_rad=_float(data, "amplitude_rad", controller.target_amplitude_rad),
            )
        )
    return sorted(segments, key=lambda s: s.t_s)


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return _as_mapping(raw.get(name, {}), name)


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"Section {name!r} must be a TOML table")


def _float(data: Mapping[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key!r} must be a number")
    return float(value)


def _int(data: Mapping[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer")
    return int(value)
