# EMAC Phase 0 Host Simulator

This repository is currently a Phase 0 host-only simulator for an event-first
electromagnetic actuator control library. It models a soft-iron, attract-only
magnetic pendulum and validates the estimator/supervisor loop before firmware is
written.

The immediate goal is a configurable **virtual actuator lab**: define fictional
pendulum, gate, coil, driver, and controller values in a TOML file, run the sim on
a PC, inspect the plots/visualizer, and use the results to guide later hardware.

The future hardware target remains the one documented in `docs/DESIGN.md`:
ESP32-S3, hardware timer capture for the photogate, and a unipolar coil power
stage for a soft-iron bob.

## Setup

```powershell
python -m pip install -e .[dev]
```

## Run a Configurable Simulation

```powershell
emac-sim --config examples/configs/pendulum_softiron_1gate.toml --outdir build/phase0
```

Override the configured duration for a quick smoke run:

```powershell
emac-sim --config examples/configs/pendulum_softiron_1gate.toml --t-end 3 --no-plots
```

The legacy fixed-demo command still works and uses the built-in Phase 0 defaults:

```powershell
emac-phase0 --no-plots
```

## Run the Visual Simulator

Generate a standalone browser visualizer:

```powershell
emac-visual --outdir build/visual
```

If the console script is not on `PATH`, use the module entrypoint:

```powershell
python -m emac_sim.visual --outdir build/visual
```

The legacy entrypoint still works:

```powershell
python tools/python/run_phase0.py
```

## Verify

```powershell
python -c "import emac_sim"
python -m pytest
```

## Next Milestone

Phase 0B is the configurable virtual-hardware milestone: TOML configs for fictional
pendulums, gates, coils, drivers, and controller settings; visual comparison of runs;
and later sweeps over coil/sensor/controller parameters.

Phase 1 is the first hardware milestone: ESP32-S3 capture input, unipolar coil
power stage, timing-budget proof, and a sustained one-gate swing.
