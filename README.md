# EMAC Phase 0 Host Simulator

This repository is currently a Phase 0 host-only simulator for an event-first
electromagnetic actuator control library. It models a soft-iron, attract-only
magnetic pendulum and validates the estimator/supervisor loop before firmware is
written.

The future hardware target remains the one documented in `docs/DESIGN.md`:
ESP32-S3, hardware timer capture for the photogate, and a unipolar coil power
stage for a soft-iron bob.

## Setup

```powershell
python -m pip install -e .[dev]
```

## Run the Phase 0 Demo

```powershell
emac-phase0 --outdir build/phase0
```

For a smoke run without plots:

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

Phase 1 is the first hardware milestone: ESP32-S3 capture input, unipolar coil
power stage, timing-budget proof, and a sustained one-gate swing.
