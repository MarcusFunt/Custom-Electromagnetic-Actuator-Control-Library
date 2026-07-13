"""Shared library for the long FEMM design-of-experiments study (used by run_study.py).

  REQUIRES FEMM (a CORE requirement, not optional): CorrectedFemmBackend drives REAL FEMM
  axisymmetric solves via `pip install pyfemm` + the FEMM app (http://www.femm.info/). It is
  imported lazily (inside FemmBackend.__init__), so `import study_lib` works without FEMM,
  but actually building a LUT does not. Windows-only. See run_study.py's header.

Bridges the repo's REAL FEMM backend (fem/femm_backend.py) into the linear-stepper
exit-speed simulation, so we can measure how design knobs affect exit speed with the
force physics coming from actual axisymmetric FEMM solves -- something neither the
optimizer (analytic / closed-form fem_reference only) nor emac-femgen (LUT only, no
sim) does on its own.

CorrectedFemmBackend re-applies the three fem/femm_backend.py fixes (sign, air-domain, mesh
-- now also fixed in the shipped backend on this branch) AND adds two study-specific things
the shipped backend doesn't have: a FIXED per-LUT air domain sized to the widest swept offset
(bounded, vs the shipped +|offset| which balloons for large coils) and a per-process temp
filename so parallel workers don't collide.
"""
from __future__ import annotations
import dataclasses
import math
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

# locate the emac_sim package (repo_root/tools/python); this file lives at
# repo_root/studies/femm_trends/, so the repo root is parents[2]. The FEMM run harness
# cannot work without it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "python"))

from emac_sim import coil_design
from emac_sim.fem.femm_backend import (FemmBackend, NDFEB_RELATIVE_PERMEABILITY, MU_0,
                                       _AIR_MARGIN_FACTOR, _SLUG_GROUP, _COIL_GROUP)
from emac_sim.fem.geometry import (CoilWindingGeometry, SlugGeometry, default_sweep_ranges,
                                   coupling_scale_m, _two_region_grid)
from emac_sim.fem.lut import ForceLUT
from emac_sim.fem.sweep import sweep_coil
from emac_sim.linear_estimator import LinearStepperEstimator
from emac_sim.linear_plant import CoilStation, GateStation, LinearActuatorParams
from emac_sim.linear_sim import LinearSimulator
from emac_sim.linear_supervisor import FAULT, StepperSupervisor
from emac_sim import optimize_design as od

V_TGT_FULL_THRUST = od.V_TGT_FULL_THRUST


class CorrectedFemmBackend(FemmBackend):
    """FemmBackend with three bugs fixed (all found empirically -- see the study writeup):

      1. MESH: shipped code passes automesh=1, which makes FEMM ignore its computed mesh
         size (mesh_size_m is a silent no-op). We pass automesh=0 so the intended mesh
         (0.15*min(radial_thickness, magnet_radius)) takes effect -- cuts offset-symmetry
         error on a representative coil from ~3.3% to ~0.6%.
      2. SIGN: shipped backend's Maxwell-stress force is INVERTED relative to the reference
         backend and linear_plant.net_force's convention -- a FEMM LUT drives the slug
         backward, giving 0 exit speed for every design. Verified: negating it makes FEMM
         agree in sign with the reference backend (and produce working designs). We negate.
      3. DOMAIN: shipped air half-extent is _AIR_MARGIN_FACTOR*max(coil_len, magnet_len),
         which does NOT grow with offset -- so far-offset solves (the sweep spans +/-5x the
         coupling scale, wider than this domain) put the slug on/outside the air boundary
         and return phantom forces (~0.9 N where the true value is ~0.07 N), corrupting the
         LUT's clamped tail. We add |offset| to the half-extent so the slug is always
         enclosed with full margin.

    Body copied verbatim from FemmBackend.solve except those three changes."""

    def solve(self, coil: CoilWindingGeometry, slug: SlugGeometry,
              offset_m: float, current_a: float):
        from emac_sim.fem.backend import ForcePoint
        femm = self._femm
        self._ensure_open()
        femm.newdocument(0)
        femm.mi_probdef(0, "meters", "axi", 1e-8, 0, 30)
        mesh = self.mesh_size_m or (0.15 * min(coil.radial_thickness_m, slug.magnet_radius_m))
        femm.mi_getmaterial("Air")
        femm.mi_getmaterial("Copper")
        femm.mi_addmaterial("NdFeB", NDFEB_RELATIVE_PERMEABILITY, NDFEB_RELATIVE_PERMEABILITY,
                            slug.remanence_t / (MU_0 * NDFEB_RELATIVE_PERMEABILITY), 0, 0, 0,
                            0, 1, 0, 0, 0, 0)
        outer_r = _AIR_MARGIN_FACTOR * coil.outer_radius_m(slug)
        # Fix #3: the domain must enclose the slug (at z=-offset) with margin. It is set
        # ONCE per LUT (build_femm_lut -> _half_extent_override) big enough for the widest
        # swept offset, NOT inflated per-offset -- a per-offset "6*part + |offset|" domain
        # ballooned to 0.7 m+ for large coils, which made FEMM's mesher fail outright
        # ("Internal application error") and each surviving solve take minutes.
        half_extent = getattr(self, "_half_extent_override", None) or \
            (_AIR_MARGIN_FACTOR * max(coil.coil_length_m, slug.magnet_length_m) + abs(offset_m))
        femm.mi_drawline(0, -half_extent, outer_r, -half_extent)
        femm.mi_drawline(outer_r, -half_extent, outer_r, half_extent)
        femm.mi_drawline(outer_r, half_extent, 0, half_extent)
        femm.mi_drawline(0, half_extent, 0, -half_extent)
        femm.mi_addboundprop("AirBoundary", 0, 0, 0, 0, 0, 0, 0, 0, 0)
        femm.mi_selectsegment(outer_r, 0)
        femm.mi_setsegmentprop("AirBoundary", 0, 1, 0, 0)
        femm.mi_clearselected()
        femm.mi_addblocklabel(0.5 * outer_r, 0.9 * half_extent)
        femm.mi_selectlabel(0.5 * outer_r, 0.9 * half_extent)
        femm.mi_setblockprop("Air", 1, 0, "<None>", 0, 0, 0)
        femm.mi_clearselected()
        slug_z0 = -offset_m - 0.5 * slug.magnet_length_m
        slug_z1 = -offset_m + 0.5 * slug.magnet_length_m
        femm.mi_drawline(0, slug_z0, slug.magnet_radius_m, slug_z0)
        femm.mi_drawline(slug.magnet_radius_m, slug_z0, slug.magnet_radius_m, slug_z1)
        femm.mi_drawline(slug.magnet_radius_m, slug_z1, 0, slug_z1)
        femm.mi_drawline(0, slug_z1, 0, slug_z0)
        femm.mi_addblocklabel(0.5 * slug.magnet_radius_m, -offset_m)
        femm.mi_selectlabel(0.5 * slug.magnet_radius_m, -offset_m)
        femm.mi_setblockprop("NdFeB", 0, mesh, "<None>", 90, _SLUG_GROUP, 0)   # automesh 1->0
        femm.mi_clearselected()
        circuit = "coil"
        femm.mi_addcircprop(circuit, current_a, 1)
        bore_r = coil.bore_radius_m(slug)
        outer_coil_r = coil.outer_radius_m(slug)
        coil_z0 = -0.5 * coil.coil_length_m
        coil_z1 = 0.5 * coil.coil_length_m
        femm.mi_drawline(bore_r, coil_z0, outer_coil_r, coil_z0)
        femm.mi_drawline(outer_coil_r, coil_z0, outer_coil_r, coil_z1)
        femm.mi_drawline(outer_coil_r, coil_z1, bore_r, coil_z1)
        femm.mi_drawline(bore_r, coil_z1, bore_r, coil_z0)
        femm.mi_addblocklabel(0.5 * (bore_r + outer_coil_r), 0.0)
        femm.mi_selectlabel(0.5 * (bore_r + outer_coil_r), 0.0)
        femm.mi_setblockprop("Copper", 0, mesh, circuit, 0, _COIL_GROUP, coil.turns)  # automesh 1->0
        femm.mi_clearselected()
        femm.mi_saveas(getattr(self, "_tmp_fem", None) or f"_emac_femstudy_{os.getpid()}.fem")
        femm.mi_analyze(1)
        femm.mi_loadsolution()
        femm.mo_groupselectblock(_SLUG_GROUP)
        force_n = femm.mo_blockintegral(19)
        femm.mo_clearblock()
        return ForcePoint(force_n=-float(force_n))   # Fix #2: match reference/plant sign


# ---- LUT resolution + FEMM-appropriate sweep geometry -------------------------------
N_OFFSETS = 35
N_CURRENTS = 7          # force is ~linear in current (air-core coil, mu_r~1.05) -> few needed
CURRENT_MAX_A = 90.0    # LUT current axis span; must exceed every i_max used in the sim sweep
# The repo default sweeps offsets to 5x the coupling scale (edge force <0.1% of peak) -- but
# that puts the slug so far from the coil that the FEMM air domain needed to enclose it
# balloons and the mesher fails. 3x already gets the edge to ~1% of peak (verified against
# the reference backend), which is plenty for a force table whose tail is clamped anyway,
# while keeping the domain FEMM can actually mesh.
FEMM_FAR_SPAN_FACTOR = 3.0
FEMM_FINE_SPAN_FACTOR = 1.5
DOMAIN_MARGIN_FACTOR = 3.0    # air beyond the farthest slug position, in part-sizes


def femm_sweep_grid(turns, coil_length_m, radial_thickness_m, magnet_radius_m,
                    magnet_length_m, remanence_t):
    """(coil, slug, offsets, currents, half_extent) for one geometry -- a FEMM-appropriate
    offset grid (narrower far-span than the repo default) and the fixed air-domain
    half-extent that encloses the farthest swept offset with margin."""
    slug = SlugGeometry(magnet_radius_m, magnet_length_m, remanence_t)
    coil = CoilWindingGeometry(0.0, turns, coil_length_m, radial_thickness_m)
    scale = coupling_scale_m(coil, slug)
    far = FEMM_FAR_SPAN_FACTOR * scale
    offsets = _two_region_grid(FEMM_FINE_SPAN_FACTOR * scale, far, N_OFFSETS)
    currents = tuple(-CURRENT_MAX_A + 2.0 * CURRENT_MAX_A * k / (N_CURRENTS - 1)
                     for k in range(N_CURRENTS))
    half_extent = far + 0.5 * magnet_length_m + DOMAIN_MARGIN_FACTOR * max(coil_length_m, magnet_length_m)
    return coil, slug, offsets, currents, half_extent


def build_femm_lut(turns, coil_length_m, radial_thickness_m, magnet_radius_m,
                   magnet_length_m, remanence_t, backend, on_point=None) -> ForceLUT:
    """One real-FEMM force table for a coil/slug geometry (position-independent: the force
    law depends only on slug-coil offset, so one table serves every coil of this geometry)."""
    coil, slug, offsets, currents, half_extent = femm_sweep_grid(
        turns, coil_length_m, radial_thickness_m, magnet_radius_m, magnet_length_m, remanence_t)
    backend._half_extent_override = half_extent
    try:
        return sweep_coil(coil, slug, backend, offsets_m=offsets, currents_a=currents,
                          on_point=on_point)
    finally:
        backend._half_extent_override = None


def simulate_exit_speed(knobs: od.DesignKnobs, force_law: str, lut: ForceLUT | None,
                        dt: float = 2e-4, t_end: float = 3.0,
                        bootstrap_timeout_s: float = 0.05) -> float:
    """Exit speed (m/s) for a design. force_law in {"analytic","fem_reference"} defers to the
    repo's optimize_design.simulate_design verbatim (so those columns match the repo's own
    tools exactly). force_law=="femm" builds the plant from build_coil_station (real winding
    R/L/k_a/x_c/thermal) but overrides each coil's force law with the supplied real-FEMM LUT."""
    if force_law in ("analytic", "fem_reference"):
        return od.simulate_design(knobs, dt=dt, t_end=t_end,
                                  bootstrap_timeout_s=bootstrap_timeout_s, force_law=force_law)
    if force_law != "femm":
        raise ValueError(force_law)
    if lut is None:
        raise ValueError("femm force_law needs a LUT")
    pitch = knobs.coil_length_m
    base_coils = tuple(
        coil_design.build_coil_station(
            position_m=k * pitch, turns=knobs.turns, coil_length_m=knobs.coil_length_m,
            radial_thickness_m=knobs.radial_thickness_m, magnet_radius_m=knobs.magnet_radius_m,
            magnet_length_m=knobs.magnet_length_m, remanence_t=knobs.remanence_t,
        ) for k in range(knobs.n_coils)
    )
    # Attach the real-FEMM force law to every coil (frozen dataclass -> replace).
    coils = tuple(dataclasses.replace(c, force_lut=lut) for c in base_coils)
    gate_positions = [-0.5 * pitch] + [(k + 0.5) * pitch for k in range(knobs.n_coils - 1)]
    gates = tuple(GateStation(position_m=x, w_eff=0.002) for x in gate_positions)
    p = LinearActuatorParams(
        mass_kg=coil_design.magnet_mass_kg(knobs.magnet_radius_m, knobs.magnet_length_m),
        coils=coils, gates=gates, current_loop="rl", bus_voltage_v=knobs.bus_voltage_v,
        driver_bipolar=knobs.driver_bipolar, thermal_model=True, ambient_temperature_c=20.0,
    )
    x0 = -0.5 * pitch - 0.001
    est = LinearStepperEstimator([g.position_m for g in p.gates], [g.w_eff for g in p.gates])
    sup = StepperSupervisor(p, i_max=knobs.i_max_a, pm_envelope=knobs.pump_envelope,
                            bootstrap_timeout_s=bootstrap_timeout_s)
    sim = LinearSimulator(p, est, sup, dt=dt, sample_every=1_000_000)
    log = sim.run(x0=x0, v0=0.0, v_tgt=V_TGT_FULL_THRUST, t_end=t_end)
    if sup.mode == FAULT or not log.gate_t:
        return 0.0
    return log.gate_v[-1]
