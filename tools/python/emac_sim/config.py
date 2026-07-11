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
from .linear_plant import CoilStation, GateStation, LinearActuatorParams
from .fem.lut import ForceLUT


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
    # H-bridge (True) vs. single half-bridge (False, default) -- only consumed by the
    # linear stepper's "rl" current loop (see linear_plant.LinearActuatorParams.driver_bipolar).
    bipolar: bool = False


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


@dataclass(frozen=True)
class LinearActuatorConfig:
    mass_kg: float = 0.20
    damping_n_per_mps: float = 0.05
    # Must sit behind gate[0] (default -pitch/2 = -0.025 m) so the bootstrap kick has an
    # entry gate left to cross -- starting AT or AHEAD of gate[0] would stall forever. Also
    # must stay within a few coil x_c of coil[0], or the bootstrap coil can't reach it at all.
    initial_position_m: float = -0.03
    initial_velocity_m_s: float = 0.0
    end_of_travel: str = "coast"          # "coast" | "brake_hold"
    # Constant forward force (N) from a pressurized reservoir behind the slug, independent
    # of coil current -- see linear_plant.LinearActuatorParams.pressure_bias_n and
    # docs/DESIGN_LINEAR.md. Default 0.0 reproduces the unpressurized model exactly.
    pressure_bias_n: float = 0.0
    # One-node-per-coil thermal model -- see linear_plant.LinearActuatorParams.
    # thermal_model. False (default) reproduces the fixed-resistance model exactly.
    thermal_model: bool = False
    ambient_temperature_c: float = 20.0


@dataclass(frozen=True)
class LinearGateConfig:
    position_m: float = 0.0
    effective_width_m: float = 0.004
    noise_std_s: float = 0.0
    dropout_probability: float = 0.0


@dataclass(frozen=True)
class LinearCoilConfig:
    position_m: float = 0.0
    x_c_m: float = 0.020
    # 0.0: this build's slug has no iron, only a magnet -- see linear_plant.CoilStation.Cmag.
    c_mag_n_per_a2: float = 0.0
    i_sat_a: float = 6.0
    resistance_ohm: float = 1.2
    inductance_h: float = 0.004
    max_current_a: float = 6.0
    # PM-branch gain (N/A) from the slug's embedded permanent magnet, air-core coils --
    # see linear_plant.CoilStation.k_a. Matches that dataclass's default (0.0 there would
    # reproduce the pure-reluctance model, but the actual slug now has a weak PM).
    k_a_n_per_a: float = 0.20
    # One-node thermal model -- see linear_plant.CoilStation. Only active when
    # LinearActuatorConfig.thermal_model is True.
    thermal_mass_j_per_k: float = 12.0
    thermal_resistance_k_per_w: float = 8.0
    # Path to a fem.lut.ForceLUT .npz file (see tools/python/emac_sim/fem/cli.py's
    # `emac-femgen`), resolved relative to the current working directory. When set, this
    # coil's force law comes ENTIRELY from the swept table instead of x_c_m/c_mag_n_per_a2/
    # k_a_n_per_a/i_sat_a -- see linear_plant.net_force. None (default) reproduces the
    # exact prior analytic-lobe behavior for every existing config.
    force_lut_path: str | None = None
    # Physical winding geometry (see fem/geometry.py's CoilWindingGeometry / fem/
    # from_config.py's coil_geometry_from_config) -- ONLY consumed by `emac-femgen`,
    # never by to_actuator_params(). Not required for a normal (non-FEM) sim; kept here so
    # the SAME config file that runs the sim can also drive table generation for it,
    # rather than needing a second geometry-only file. Defaults describe a modest
    # 200-turn/20mm coil, matching coil_design.py's own doc examples.
    turns: int = 200
    coil_winding_length_m: float = 0.020
    radial_thickness_m: float = 0.010
    bore_clearance_m: float = 0.0015
    packing_factor: float = 0.8
    winding_temperature_c: float = 20.0


@dataclass(frozen=True)
class SlugConfig:
    """The moving PM slug's geometry -- see fem/geometry.py's SlugGeometry. ONE slug per
    actuator (unlike per-coil config), since every coil in a linear stepper couples to the
    same physical slug as it travels. Only consumed by `emac-femgen`, like the coil
    geometry fields above. Defaults describe a modest N42-ish NdFeB rod."""

    magnet_radius_m: float = 0.008
    magnet_length_m: float = 0.020
    remanence_t: float = 1.2


@dataclass(frozen=True)
class LinearControllerConfig:
    kind: str = "stepper_supervisor"
    target_velocity_m_s: float = 0.5
    k_velocity: float = 0.30
    pulse_width_half_period_fraction: float = 0.30
    phase_advance_s: float = 0.002
    bootstrap_dwell_s: float = 0.05
    bootstrap_timeout_s: float = 0.20
    i_max_a: float = 6.0
    # Current envelope for any station with an active PM branch -- "rcos" (default,
    # smooth force) | "trapezoid" | "square" (unsmoothed, more thrust per i_peak). See
    # linear_supervisor.StepperSupervisor's pm_envelope / supervisor.envelope_average_linear.
    pump_envelope: str = "rcos"


def _default_linear_coils(pitch: float = 0.05, n: int = 5) -> list[LinearCoilConfig]:
    return [LinearCoilConfig(position_m=k * pitch) for k in range(n)]


def _default_linear_gates(pitch: float = 0.05, n_coils: int = 5) -> list[LinearGateConfig]:
    # entry gate before coil 0, then one gate between each adjacent coil pair -- see
    # linear_plant.default_gate_stations() for the same scheme and its rationale.
    positions = [-0.5 * pitch] + [(k + 0.5) * pitch for k in range(n_coils - 1)]
    return [LinearGateConfig(position_m=x) for x in positions]


@dataclass(frozen=True)
class LinearSimulationConfig:
    """Config for the linear one-way stepper actuator (docs/DESIGN_LINEAR.md) -- the
    linear counterpart to SimulationConfig, selected via `[sim] kind = "linear_stepper"`
    (see parse_config). Kept as a fully separate dataclass tree rather than reusing
    PendulumConfig/GateConfig/CoilConfig: the two geometries use different units
    (radians+angle vs meters+position) and sharing fields would need None-guarded
    dual-purpose attributes for no real benefit."""

    actuator: LinearActuatorConfig = field(default_factory=LinearActuatorConfig)
    gates: list[LinearGateConfig] = field(default_factory=_default_linear_gates)
    coils: list[LinearCoilConfig] = field(default_factory=_default_linear_coils)
    driver: DriverConfig = field(default_factory=DriverConfig)
    controller: LinearControllerConfig = field(default_factory=LinearControllerConfig)
    slug: SlugConfig = field(default_factory=SlugConfig)
    duration_s: float = 3.0
    dt_s: float = 2e-4
    sample_every: int = 10

    def to_actuator_params(self) -> LinearActuatorParams:
        coils = tuple(
            CoilStation(position_m=c.position_m, x_c=c.x_c_m, Cmag=c.c_mag_n_per_a2,
                        i_sat=c.i_sat_a, k_a=c.k_a_n_per_a,
                        resistance_ohm=c.resistance_ohm, inductance_h=c.inductance_h,
                        thermal_mass_j_per_k=c.thermal_mass_j_per_k,
                        thermal_resistance_k_per_w=c.thermal_resistance_k_per_w,
                        force_lut=ForceLUT.load(c.force_lut_path) if c.force_lut_path else None)
            for c in self.coils
        )
        gates = tuple(
            GateStation(position_m=g.position_m, w_eff=g.effective_width_m)
            for g in self.gates
        )
        return LinearActuatorParams(
            mass_kg=self.actuator.mass_kg,
            damping_n_per_mps=self.actuator.damping_n_per_mps,
            coils=coils,
            gates=gates,
            end_of_travel=self.actuator.end_of_travel,
            pressure_bias_n=self.actuator.pressure_bias_n,
            current_loop=self.driver.current_loop,
            bus_voltage_v=self.driver.bus_voltage_v,
            driver_bipolar=self.driver.bipolar,
            thermal_model=self.actuator.thermal_model,
            ambient_temperature_c=self.actuator.ambient_temperature_c,
        )


def default_config() -> SimulationConfig:
    """Return the historical Phase 0 demo configuration."""
    return SimulationConfig(
        target_segments=[
            TargetSegment(0.0, 0.35),
            TargetSegment(8.0, 0.20),
            TargetSegment(15.0, 0.30),
        ]
    )


def default_linear_config() -> LinearSimulationConfig:
    """Return the default 5-coil/5-gate linear one-way stepper demo configuration."""
    return LinearSimulationConfig()


def load_config(path: str | Path) -> "SimulationConfig | LinearSimulationConfig":
    """Load a TOML simulation config from *path*. Dispatches on `[sim] kind` -- see
    parse_config()."""
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return parse_config(raw)


def parse_config(raw: Mapping[str, Any]) -> "SimulationConfig | LinearSimulationConfig":
    """Parse a raw TOML mapping into a simulation config. `[sim] kind` selects the
    geometry: "pendulum" (default, so every existing config that omits it is unaffected)
    or "linear_stepper" (docs/DESIGN_LINEAR.md) -- this is the one shared entry point
    ("a method that supports both") that the rest of the CLI dispatches on."""
    kind = str(_section(raw, "sim").get("kind", "pendulum"))
    if kind == "pendulum":
        return _parse_pendulum_config(raw)
    if kind == "linear_stepper":
        return _parse_linear_config(raw)
    raise ValueError(f"unknown [sim] kind: {kind!r}")


def _parse_pendulum_config(raw: Mapping[str, Any]) -> SimulationConfig:
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


def _parse_linear_config(raw: Mapping[str, Any]) -> LinearSimulationConfig:
    actuator = _linear_actuator(raw.get("actuator", {}))
    gates = _linear_gates(raw)
    coils = _linear_coils(raw)
    driver = _driver(raw.get("driver", {}))
    controller = _linear_controller(raw.get("controller", {}))
    slug = _slug(raw.get("slug", {}))
    sim_raw = _section(raw, "sim")

    return LinearSimulationConfig(
        actuator=actuator,
        gates=gates,
        coils=coils,
        driver=driver,
        controller=controller,
        slug=slug,
        duration_s=_float(sim_raw, "duration_s", 3.0),
        dt_s=_float(sim_raw, "dt_s", 2e-4),
        sample_every=_int(sim_raw, "sample_every", 10),
    )


def _linear_actuator(raw: Any) -> LinearActuatorConfig:
    data = _as_mapping(raw, "actuator")
    return LinearActuatorConfig(
        mass_kg=_float(data, "mass_kg", 0.20),
        damping_n_per_mps=_float(data, "damping_n_per_mps", 0.05),
        initial_position_m=_float(data, "initial_position_m", -0.03),
        initial_velocity_m_s=_float(data, "initial_velocity_m_s", 0.0),
        end_of_travel=str(data.get("end_of_travel", "coast")),
        pressure_bias_n=_float(data, "pressure_bias_n", 0.0),
        thermal_model=bool(data.get("thermal_model", False)),
        ambient_temperature_c=_float(data, "ambient_temperature_c", 20.0),
    )


def _linear_gates(raw: Mapping[str, Any]) -> list[LinearGateConfig]:
    source = raw.get("gates", None)
    if source is None:
        return _default_linear_gates()
    entries = source if isinstance(source, list) else [source]
    gates = []
    for idx, item in enumerate(entries):
        data = _as_mapping(item, f"gates[{idx}]")
        gates.append(
            LinearGateConfig(
                position_m=_float(data, "position_m", 0.0),
                effective_width_m=_float(data, "effective_width_m", 0.004),
                noise_std_s=_float(data, "noise_std_s", 0.0),
                dropout_probability=_float(data, "dropout_probability", 0.0),
            )
        )
    if not gates:
        raise ValueError("At least one gate is required")
    return gates


def _linear_coils(raw: Mapping[str, Any]) -> list[LinearCoilConfig]:
    source = raw.get("coils", None)
    if source is None:
        return _default_linear_coils()
    entries = source if isinstance(source, list) else [source]
    coils = []
    for idx, item in enumerate(entries):
        data = _as_mapping(item, f"coils[{idx}]")
        coils.append(
            LinearCoilConfig(
                position_m=_float(data, "position_m", 0.0),
                x_c_m=_float(data, "x_c_m", 0.020),
                c_mag_n_per_a2=_float(data, "c_mag_n_per_a2", 0.0),
                i_sat_a=_float(data, "i_sat_a", 6.0),
                resistance_ohm=_float(data, "resistance_ohm", 1.2),
                inductance_h=_float(data, "inductance_h", 0.004),
                max_current_a=_float(data, "max_current_a", 6.0),
                k_a_n_per_a=_float(data, "k_a_n_per_a", 0.20),
                thermal_mass_j_per_k=_float(data, "thermal_mass_j_per_k", 12.0),
                thermal_resistance_k_per_w=_float(data, "thermal_resistance_k_per_w", 8.0),
                force_lut_path=_optional_str(data, "force_lut_path"),
                turns=_int(data, "turns", 200),
                coil_winding_length_m=_float(data, "coil_winding_length_m", 0.020),
                radial_thickness_m=_float(data, "radial_thickness_m", 0.010),
                bore_clearance_m=_float(data, "bore_clearance_m", 0.0015),
                packing_factor=_float(data, "packing_factor", 0.8),
                winding_temperature_c=_float(data, "winding_temperature_c", 20.0),
            )
        )
    if not coils:
        raise ValueError("At least one coil is required")
    return coils


def _slug(raw: Any) -> SlugConfig:
    data = _as_mapping(raw, "slug")
    return SlugConfig(
        magnet_radius_m=_float(data, "magnet_radius_m", 0.008),
        magnet_length_m=_float(data, "magnet_length_m", 0.020),
        remanence_t=_float(data, "remanence_t", 1.2),
    )


def _linear_controller(raw: Any) -> LinearControllerConfig:
    data = _as_mapping(raw, "controller")
    return LinearControllerConfig(
        kind=str(data.get("kind", "stepper_supervisor")),
        target_velocity_m_s=_float(data, "target_velocity_m_s", 0.5),
        k_velocity=_float(data, "k_velocity", 0.30),
        pulse_width_half_period_fraction=_float(data, "pulse_width_half_period_fraction", 0.30),
        phase_advance_s=_float(data, "phase_advance_s", 0.002),
        bootstrap_dwell_s=_float(data, "bootstrap_dwell_s", 0.05),
        bootstrap_timeout_s=_float(data, "bootstrap_timeout_s", 0.20),
        i_max_a=_float(data, "i_max_a", 6.0),
        pump_envelope=str(data.get("pump_envelope", "rcos")),
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
        bipolar=bool(data.get("bipolar", False)),
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


def _optional_str(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key, None)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key!r} must be a string")
    return value


def _int(data: Mapping[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"{key!r} must be an integer")
    return int(value)
