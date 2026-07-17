"""Tests for the reluctance-projectile (soft-iron slug) mode.

Everything here runs with the ANALYTIC reference backend / analytic plant -- no FEMM, CI-safe,
and safe to run while a separate FEMM sweep is in progress. One real-FEMM smoke test is gated
on `pytest.importorskip("femm")`.

The reluctance force must be: attract-only (pulls the iron slug toward the coil center from
both sides), EVEN in current (reversing the coil current doesn't change which way the iron is
pulled), and SATURATING (~i^2 at low current, rolling off past i_sat). The PM path must be
untouched (default slug_type="pm").
"""
from __future__ import annotations

import math

import pytest

from emac_sim import coil_design as cd
from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry
from emac_sim.fem.reference_backend import AnalyticReferenceBackend
from emac_sim.linear_plant import LinearActuatorParams, net_force
from emac_sim.optimize_design import DesignKnobs, build_params, simulate_design

GEOM = dict(turns=500, coil_length_m=0.03, radial_thickness_m=0.01,
            magnet_radius_m=0.004, magnet_length_m=0.02)


# ----------------------------- geometry seam ----------------------------------
def test_slug_geometry_type_validation():
    pm = SlugGeometry(0.004, 0.02, 1.2)
    assert pm.slug_type == "pm" and not pm.is_reluctance
    rel = SlugGeometry(0.004, 0.02, 1.2, slug_type="reluctance")
    assert rel.is_reluctance and rel.steel_material == "1018 Steel"
    with pytest.raises(ValueError):
        SlugGeometry(0.004, 0.02, 1.2, slug_type="bogus")


# ----------------------------- coil_design model ------------------------------
def test_reluctance_force_model_sane():
    w = cd.wind_coil(GEOM["turns"], GEOM["coil_length_m"], GEOM["radial_thickness_m"], 0.0055)
    cmag, i_sat, x_c = cd.reluctance_force_model(
        w.inductance_h, GEOM["coil_length_m"], 0.0055, GEOM["magnet_radius_m"],
        GEOM["magnet_length_m"], GEOM["turns"])
    assert cmag > 0.0 and i_sat > 0.0 and 0.0 < x_c < 0.1


def test_slug_mass_steel_vs_ndfeb():
    r, L = 0.004, 0.02
    pm = cd.slug_mass_kg(r, L, "pm")
    rel = cd.slug_mass_kg(r, L, "reluctance")
    assert pm == pytest.approx(cd.magnet_mass_kg(r, L))          # pm == the NdFeB default
    assert rel > pm                                             # steel is denser than NdFeB


def test_build_coil_station_branches():
    rel = cd.build_coil_station(0.0, remanence_t=1.2, slug_type="reluctance", **GEOM)
    assert rel.Cmag > 0.0 and rel.k_a == 0.0 and rel.i_sat > 0.0
    pm = cd.build_coil_station(0.0, remanence_t=1.2, **GEOM)     # default pm
    assert pm.Cmag == 0.0 and pm.k_a > 0.0                       # PM path unchanged


# ----------------------------- reference-backend force ------------------------
def _rel_backend_pieces():
    coil = CoilWindingGeometry(0.0, GEOM["turns"], GEOM["coil_length_m"], GEOM["radial_thickness_m"])
    slug = SlugGeometry(GEOM["magnet_radius_m"], GEOM["magnet_length_m"], 1.2, slug_type="reluctance")
    ref = AnalyticReferenceBackend()
    w = cd.wind_coil(coil.turns, coil.coil_length_m, coil.radial_thickness_m, coil.bore_radius_m(slug))
    _, i_sat, x_c = cd.reluctance_force_model(w.inductance_h, coil.coil_length_m,
                                              coil.bore_radius_m(slug), slug.magnet_radius_m,
                                              slug.magnet_length_m, coil.turns)
    return ref, coil, slug, i_sat, x_c


def test_reluctance_force_is_even_in_current():
    ref, coil, slug, i_sat, x_c = _rel_backend_pieces()
    for i in (5.0, 25.0, 60.0):
        fp = ref.solve(coil, slug, x_c, i).force_n
        fn = ref.solve(coil, slug, x_c, -i).force_n
        assert fp == pytest.approx(fn)                           # F(-i) == F(i)


def test_reluctance_force_is_attract_only_toward_center():
    ref, coil, slug, i_sat, x_c = _rel_backend_pieces()
    f_ahead = ref.solve(coil, slug, x_c, 30.0).force_n           # slug ahead of coil
    f_behind = ref.solve(coil, slug, -x_c, 30.0).force_n         # slug behind coil
    f_center = ref.solve(coil, slug, 0.0, 30.0).force_n
    assert f_ahead < 0.0                                        # pulled back toward center
    assert f_behind > 0.0                                       # pulled forward toward center
    assert abs(f_center) < 1e-9                                 # no force exactly centered
    assert f_ahead == pytest.approx(-f_behind)                  # symmetric


def test_reluctance_force_saturates():
    ref, coil, slug, i_sat, x_c = _rel_backend_pieces()
    def f(i):
        return abs(ref.solve(coil, slug, x_c, i).force_n)
    lo = 0.15 * i_sat
    ratio_low = f(2 * lo) / f(lo)                                # ~4 if quadratic
    ratio_high = f(2 * i_sat) / f(i_sat)                         # < 4 once saturating
    assert ratio_low > 3.5
    assert ratio_high < 2.6


# ----------------------------- plant / sim ------------------------------------
def test_net_force_reluctance_pulls_toward_center():
    coil = cd.build_coil_station(0.0, remanence_t=1.2, slug_type="reluctance", **GEOM)
    p = LinearActuatorParams(coils=(coil,))
    # positive current only (reluctance is unipolar); force is toward the coil center
    assert net_force(0.01, [20.0], p) < 0.0                      # ahead -> pulled back
    assert net_force(-0.01, [20.0], p) > 0.0                     # behind -> pulled forward
    assert net_force(0.0, [20.0], p) == pytest.approx(0.0)


def test_reluctance_design_launches_analytic_and_fem_reference_agree():
    knobs = DesignKnobs(bus_voltage_v=260.0, driver_bipolar=False, pump_envelope="square",
                        n_coils=5, turns=500, coil_length_m=0.03, radial_thickness_m=0.01,
                        magnet_radius_m=0.004, magnet_length_m=0.02, remanence_t=1.2,
                        i_max_a=40.0, slug_type="reluctance")
    p = build_params(knobs, force_law="analytic")
    assert p.coils[0].Cmag > 0.0 and p.coils[0].k_a == 0.0
    assert p.mass_kg == pytest.approx(cd.slug_mass_kg(0.004, 0.02, "reluctance"))
    v_analytic = simulate_design(knobs, dt=1e-4, t_end=1.0, force_law="analytic")
    v_femref = simulate_design(knobs, dt=1e-4, t_end=1.0, force_law="fem_reference")
    assert v_analytic > 1.0                                     # it actually launches
    # both analytic paths use the same q_shape*reluctance_model force -> identical
    assert v_analytic == pytest.approx(v_femref, rel=1e-6)


def test_pm_path_unchanged_by_default():
    knobs = DesignKnobs(260.0, False, "square", 5, 500, 0.03, 0.01, 0.004, 0.02, 1.2, 40.0)
    assert knobs.slug_type == "pm"
    p = build_params(knobs, force_law="analytic")
    assert p.coils[0].Cmag == 0.0 and p.coils[0].k_a > 0.0
    assert simulate_design(knobs, dt=1e-4, t_end=1.0, force_law="analytic") > 1.0


# ----------------------------- study_lib LUT ----------------------------------
def test_study_lib_reluctance_current_axis_nonnegative():
    study_lib = pytest.importorskip("study_lib")
    _, slug, _, currents, _ = study_lib.femm_sweep_grid(500, 0.03, 0.01, 0.004, 0.02, 1.2,
                                                        slug_type="reluctance")
    assert slug.slug_type == "reluctance"
    assert min(currents) == 0.0 and max(currents) == study_lib.CURRENT_MAX_A
    assert all(c >= 0.0 for c in currents) and len(currents) == study_lib.N_CURRENTS_RELUCTANCE
    # PM axis is still symmetric 3-point
    _, _, _, pm_currents, _ = study_lib.femm_sweep_grid(500, 0.03, 0.01, 0.004, 0.02, 1.2)
    assert min(pm_currents) < 0.0 and len(pm_currents) == study_lib.N_CURRENTS


def test_study_lib_reluctance_lut_and_sim():
    study_lib = pytest.importorskip("study_lib")
    lut = study_lib.build_femm_lut(500, 0.03, 0.01, 0.004, 0.02, 1.2,
                                   AnalyticReferenceBackend(), slug_type="reluctance")
    assert lut.metadata.get("slug_type") == "reluctance"
    knobs = DesignKnobs(260.0, False, "square", 5, 500, 0.03, 0.01, 0.004, 0.02, 1.2, 40.0,
                        slug_type="reluctance")
    assert study_lib.simulate_exit_speed(knobs, "femm", lut, t_end=1.0) > 1.0


# ----------------------------- bo_search --------------------------------------
def test_bo_search_reluctance_reference_eval():
    bo = pytest.importorskip("bo_search")
    geom = {"turns": 500, "coil_length_m": 0.03, "radial_thickness_m": 0.01,
            "magnet_radius_m": 0.004, "magnet_length_m": 0.02, "remanence_t": 1.2}
    be = bo.make_backend("reference")
    res = bo.evaluate_geometry(geom, n_coils=5, objective="speed", backend=be,
                               sim_t_end=0.6, slug_type="reluctance")
    assert res["error"] is None and res["value"] > 0.0
    assert res["slug_type"] == "reluctance"
    assert res["driver"]["driver_bipolar"] is False            # reluctance can't use repel


def test_bo_warmstart_filters_by_slug_type(tmp_path):
    bo = pytest.importorskip("bo_search")
    d = tmp_path / "results"
    d.mkdir()
    geom = {"turns": 450, "coil_length_m": 0.02, "radial_thickness_m": 0.014,
            "magnet_radius_m": 0.008, "magnet_length_m": 0.022, "remanence_t": 1.25}
    # one pm row, one reluctance row, same geometry
    rows = [{"cell_id": 1, **geom, "bus_voltage_v": 260.0, "driver_bipolar": True,
             "pump_envelope": "rcos", "i_max_a": 70.0, "n_coils": 5, "force_law": "femm",
             "exit_speed_mps": 9.0, "slug_type": "pm"},
            {"cell_id": 1, **geom, "bus_voltage_v": 260.0, "driver_bipolar": False,
             "pump_envelope": "square", "i_max_a": 40.0, "n_coils": 5, "force_law": "femm",
             "exit_speed_mps": 4.0, "slug_type": "reluctance"}]
    import json
    (d / "cell_0001.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    pm_recs, _ = bo.load_warmstart([d], "speed", "pm")
    rel_recs, _ = bo.load_warmstart([d], "speed", "reluctance")
    assert len(pm_recs) == 1 and pm_recs[0]["value"] == pytest.approx(9.0)
    assert len(rel_recs) == 1 and rel_recs[0]["value"] == pytest.approx(4.0)


# ----------------------------- GUI auto-detect --------------------------------
def test_gui_analyze_autodetects_reluctance(tmp_path, monkeypatch):
    import numpy as np
    from emac_sim.fem.sweep import sweep_coil
    from emac_sim.gui import server

    coil = CoilWindingGeometry(0.0, 500, 0.03, 0.01)
    slug = SlugGeometry(0.004, 0.02, 1.2, slug_type="reluctance")
    offs = np.linspace(-0.06, 0.06, 25)
    curr = np.linspace(0.0, 90.0, 7)
    lut = sweep_coil(coil, slug, AnalyticReferenceBackend(), offsets_m=offs, currents_a=curr)
    p = tmp_path / "rel.npz"
    lut.save(p)
    # analyze_lut guards paths to the project root; point that guard at tmp_path for the test
    monkeypatch.setattr(server, "_safe_path", lambda x: str(x))
    res = server.analyze_lut(str(p), reluctance=False, compare=True)
    assert res["metadata"]["slug_type"] == "reluctance"
    lin = [c for c in res["qc"]["checks"] if c["name"] == "current_linearity"][0]
    assert lin["applicable"] is False                          # auto-skipped, not failed
    assert "comparison" in res                                 # reluctance analytic overlay built


def test_config_slug_type_roundtrip():
    from emac_sim.config import SlugConfig
    from emac_sim.fem.from_config import slug_geometry_from_config
    assert slug_geometry_from_config(SlugConfig()).slug_type == "pm"
    assert slug_geometry_from_config(SlugConfig(slug_type="reluctance")).slug_type == "reluctance"


# ----------------------------- real-FEMM smoke (gated) ------------------------
def test_femm_reluctance_smoke():
    """One real nonlinear-steel FEMM reluctance solve, only when FEMM is installed. Needs
    EXCLUSIVE FEMM (don't run during a sweep); kept tiny."""
    pytest.importorskip("femm")
    from emac_sim.fem.femm_backend import FemmBackend
    coil = CoilWindingGeometry(0.0, 500, 0.03, 0.01)
    slug = SlugGeometry(0.004, 0.02, 1.2, slug_type="reluctance")
    be = FemmBackend()
    try:
        f_ahead = be.solve(coil, slug, 0.012, 40.0).force_n
        f_ahead_neg = be.solve(coil, slug, 0.012, -40.0).force_n
        f_center = be.solve(coil, slug, 0.0, 40.0).force_n
        f_small = be.solve(coil, slug, 0.012, 10.0).force_n
    finally:
        be.close()
    assert f_ahead < 0.0                                       # attractive toward center
    assert f_ahead == pytest.approx(f_ahead_neg, rel=0.05)     # ~even in current
    assert abs(f_center) < abs(f_ahead)                        # ~zero at center
    assert abs(f_ahead) < 16.0 * abs(f_small)                  # sub-quadratic (saturating): 4x I -> <16x F
