"""Unit tests for the unified GUI server's non-HTTP logic (no browser, no live server).

These cover the parts that are easy to break silently: the command->argv translation the
"Run" button relies on, the numpy-aware JSON coercion (numpy scalars leak in from the LUT /
quality modules and would otherwise 500 the API), the sweep cost estimate, the LUT->JSON
packaging the visualizer consumes, and the path-confinement guard.
"""
import json
import shutil

import numpy as np
import pytest

from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry
from emac_sim.fem.reference_backend import AnalyticReferenceBackend
from emac_sim.fem.sweep import sweep_coil
from emac_sim.gui import server as S


def test_build_argv_maps_flags_bools_and_positionals():
    argv = S.build_argv("sim", {"config": "examples/configs/x.toml", "outdir": "build/o",
                                "t_end": 2.0, "no_plots": True})
    tail = argv[argv.index("-m") + 2:]                 # drop [python, -m, module]
    assert "--config" in tail and "examples/configs/x.toml" in tail
    assert "--t-end" in tail and "2.0" in tail
    assert "--no-plots" in tail                        # bool True -> bare flag
    # femqc takes a POSITIONAL input (flag is None)
    qc = S.build_argv("femqc", {"inputs": "build/gui/fem_lut"})
    assert qc[-1] == "build/gui/fem_lut" and "--inputs" not in qc


def test_build_argv_omits_false_bools_and_empty_values():
    argv = S.build_argv("sim", {"config": "c.toml", "no_plots": False, "t_end": None})
    assert "--no-plots" not in argv
    assert "--t-end" not in argv                        # None value dropped


def test_build_argv_missing_required_raises():
    with pytest.raises(ValueError, match="required"):
        S.build_argv("femgen", {"n_offsets": 21})       # no config


def test_build_argv_unknown_command_raises():
    with pytest.raises(ValueError, match="unknown command"):
        S.build_argv("nope", {})


def test_commands_registry_is_well_formed():
    for name, spec in S.COMMANDS.items():
        # a command is backed by EITHER an importable module or a fixed (server-controlled) script
        assert (spec.get("module") or "").startswith("emac_sim.") or spec.get("script")
        if spec.get("script"):
            assert spec["script"].endswith(".py")
        for arg in spec["args"]:
            assert {"name", "type"} <= set(arg)
            assert arg["type"] in {"config", "text", "number", "int", "bool", "choice"}


def test_build_argv_script_command_launches_the_fixed_script():
    argv = S.build_argv("femm_bo", {"backend": "reference", "n_calls": 12})
    assert "-m" not in argv                              # a script, not a module
    assert argv[1].replace("\\", "/").endswith("studies/femm_trends/bo_search.py")
    assert "--backend" in argv and "reference" in argv and "12" in argv


def test_reproduce_commands_exist_for_every_run_source():
    # every run's Reproduce button must map to a real, launchable command
    for src in S.RUN_SOURCES:
        cmd = src["reproduce"]["cmd"]
        assert cmd in S.COMMANDS, f"{src['id']} -> unknown command {cmd!r}"
        argv = S.build_argv(cmd, src["reproduce"].get("args", {}))  # must not raise
        assert argv[0] and len(argv) >= 2


def test_json_default_coerces_numpy_scalars_and_arrays():
    assert S._json_default(np.bool_(True)) is True
    assert S._json_default(np.int64(7)) == 7 and isinstance(S._json_default(np.int64(7)), int)
    assert S._json_default(np.float64(1.5)) == 1.5
    assert S._json_default(np.array([1, 2])) == [1, 2]
    with pytest.raises(TypeError):
        S._json_default(object())


def test_estimate_sweep_reference_backend_projects_grid():
    est = S.estimate_sweep("examples/configs/linear_stepper_5coil_fem.toml",
                           n_offsets=21, n_currents=7, n_geometries=4,
                           backend="reference", mesh_frac=None)
    assert est["n_solves"] == 21 * 7 * 4
    assert est["backend"] == "reference"
    assert est["total_seconds"] >= 0.0 and est["total_hours"] >= 0.0
    assert "analytic" in est["note"]


def _make_lut_file(tmp_repo_rel: str):
    """Sweep a small reference LUT and save it at a repo-relative path (the API confines
    paths to the repo, so it must live under the project tree)."""
    coil = CoilWindingGeometry(0.0, turns=300, coil_length_m=0.02, radial_thickness_m=0.01)
    slug = SlugGeometry(magnet_radius_m=0.006, magnet_length_m=0.02, remanence_t=1.2)
    lut = sweep_coil(coil, slug, AnalyticReferenceBackend())
    path = S.REPO_ROOT / tmp_repo_rel
    path.parent.mkdir(parents=True, exist_ok=True)
    lut.save(path)
    return path


def test_lut_to_json_is_serializable_and_has_qc():
    rel = "build/gui_test/coil.npz"
    _make_lut_file(rel)
    try:
        d = S.lut_to_json(rel)
        assert len(d["offsets_m"]) == len(d["force_n"])
        assert d["qc"]["ok"] is True and d["qc"]["peak_force_n"] > 0
        assert len(d["qc"]["checks"]) == 8
        # the whole payload must survive json.dumps with the server's coercion (numpy leaks)
        json.dumps(d, default=S._json_default)
    finally:
        shutil.rmtree(S.REPO_ROOT / "build" / "gui_test", ignore_errors=True)


def test_list_luts_finds_saved_tables():
    rel = "build/gui_test/coil.npz"
    _make_lut_file(rel)
    try:
        luts = S.list_luts("build/gui_test")
        assert any(p.endswith("coil.npz") for p in luts)
    finally:
        shutil.rmtree(S.REPO_ROOT / "build" / "gui_test", ignore_errors=True)


def test_safe_path_blocks_directory_traversal():
    with pytest.raises(ValueError, match="escapes"):
        S._safe_path("../../etc/passwd")
    # a normal repo-relative path is fine
    assert S._safe_path("build/x").name == "x"


def test_list_configs_finds_the_example_configs():
    configs = S.list_configs()
    assert any(c.endswith("linear_stepper_5coil.toml") for c in configs)


def test_config_info_reports_coil_count_for_progress_tracking():
    info = S.config_info("examples/configs/linear_stepper_5coil_fem.toml")
    assert info["n_coils"] == 5                          # the GUI turns this into a progress bar
    assert info["turns"] > 0 and info["magnet_radius_mm"] > 0


def test_analyze_lut_reports_stats_and_analytic_overlay():
    rel = "build/gui_test/coil.npz"
    _make_lut_file(rel)                                 # metadata carries the source geometry
    try:
        d = S.analyze_lut(rel, compare=True)
        st = d["stats"]
        # a real coupling: nonzero peak thrust, a positive coupling half-width and lobe width
        assert st["peak_force_n"] > 0 and st["force_per_amp_n_a"] > 0
        assert st["peak_offset_mm"] >= 0 and st["coupling_width_mm"] > 0
        assert 0.0 <= st["far_field_frac"] < 0.2        # a full sweep has decayed at the edge
        # the LUT was built by the analytic backend, so the overlay must match it ~exactly
        assert "analytic_force_n" in d
        assert d["comparison"]["max_rel_error"] < 1e-6
        json.dumps(d, default=S._json_default)
    finally:
        shutil.rmtree(S.REPO_ROOT / "build" / "gui_test", ignore_errors=True)


def test_analyze_lut_without_geometry_metadata_skips_overlay():
    from emac_sim.fem.lut import ForceLUT
    rel = "build/gui_test/bare.npz"
    path = S.REPO_ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = np.linspace(-0.05, 0.05, 9)
    currents = np.array([-3.0, 0.0, 3.0])
    force = np.outer(-np.sin(offsets / 0.02), currents) * 0.3
    ForceLUT(offsets, currents, force, metadata={"backend": "hand"}).save(path)  # no geometry
    try:
        d = S.analyze_lut(rel, compare=True)
        assert "analytic_force_n" not in d and "comparison" not in d
        assert d["stats"]["peak_force_n"] > 0
    finally:
        shutil.rmtree(S.REPO_ROOT / "build" / "gui_test", ignore_errors=True)


def test_qc_directory_batch_triages_every_table():
    _make_lut_file("build/gui_test/a.npz")
    _make_lut_file("build/gui_test/b.npz")
    try:
        rows = S.qc_directory("build/gui_test")
        assert len(rows) == 2
        for r in rows:
            assert r["ok"] is True and r["peak_force_n"] > 0 and r["failed"] == []
        json.dumps(rows, default=S._json_default)
    finally:
        shutil.rmtree(S.REPO_ROOT / "build" / "gui_test", ignore_errors=True)


# --------------------------------------------------------------------------- run-data explorer
def test_normalize_design_search_handles_both_log_shapes():
    rows = [
        {"geom": {"n_coils": 16, "turns": 700}, "speed": 50.0, "efficiency": 0.14},   # rl-hw shape
        {"geom": {"turns": 500}, "driver": {"bus_voltage_v": 260.0}, "speed_mps": 24.0},  # femm-bo shape
        {"geom": {}, "value": 5.0},                                                    # legacy 'value'
        {"geom": {}, "speed": 1.0, "error": "boom"},                                   # dropped (error)
        {"geom": {}, "speed": float("inf")},                                           # dropped (non-finite)
    ]
    recs = S._normalize_design_search(rows)
    assert len(recs) == 3
    assert recs[0]["efficiency"] == 0.14 and recs[0]["params"]["n_coils"] == 16
    assert recs[1]["params"]["bus_voltage_v"] == 260.0 and recs[1]["efficiency"] is None
    assert recs[2]["speed"] == 5.0


def test_pareto_max_keeps_only_nondominated():
    # maximizing both: (1,1) is dominated by (2,2); (2,2),(1,3),(3,1) are all non-dominated
    idx = S._pareto_max([(1, 1), (2, 2), (1, 3), (3, 1)])
    assert set(idx) == {1, 2, 3}


def test_normalize_pareto_drops_nonfinite_and_keeps_fields():
    pts = S._normalize_pareto([{"lam": 0.0, "v": 48.0, "efficiency": 0.15, "exit_rate": 1.0},
                               {"lam": 0.2, "v": None, "efficiency": 0.3}])
    assert len(pts) == 1 and pts[0]["lam"] == 0.0 and pts[0]["exit_rate"] == 1.0


def test_load_factorial_aggregates_best_per_cell(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    # cell 0: two driver configs per force law -> keep the max
    (d / "cell_0000.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"cell_id": 0, "turns": 400, "coil_length_m": 0.01, "radial_thickness_m": 0.005,
         "magnet_radius_m": 0.004, "magnet_length_m": 0.02, "remanence_t": 1.2,
         "force_law": "analytic", "exit_speed_mps": 3.0, "sim_error": None, "bus_voltage_v": 12},
        {"cell_id": 0, "force_law": "analytic", "exit_speed_mps": 9.0, "sim_error": None, "bus_voltage_v": 260},
        {"cell_id": 0, "force_law": "femm", "exit_speed_mps": 8.0, "sim_error": None, "bus_voltage_v": 260},
        {"cell_id": 0, "force_law": "femm", "exit_speed_mps": 2.0, "sim_error": "x"},   # errored -> ignored
    ]))
    cells = S._load_factorial(d)
    assert len(cells) == 1
    c = cells[0]
    assert c["analytic_best"] == 9.0 and c["femm_best"] == 8.0
    assert c["best_config"]["bus_voltage_v"] == 260 and c["geom"]["turns"] == 400


def test_list_runs_and_load_run_are_serializable():
    runs = S.list_runs()                                  # uses whatever real artifacts exist here
    json.dumps(runs, default=S._json_default)
    for r in runs:
        assert {"id", "kind", "title", "reproduce", "summary"} <= set(r)
        d = S.load_run(r["id"])
        assert d["kind"] == r["kind"] and "reproduce_cmdline" in d
        if d["kind"] == "design_search":
            assert len(d["best_so_far"]) == len(d["records"])
        json.dumps(d, default=S._json_default)


def test_load_run_unknown_id_raises():
    with pytest.raises(ValueError, match="unknown run"):
        S.load_run("does_not_exist")


# --------------------------------------------------------------------------- candidate animation
def test_spec_from_params_fills_defaults_and_clamps_coils():
    from emac_sim.rl.geometry import CoilgunSpec
    base = CoilgunSpec()
    spec = S._spec_from_params({"turns": 800, "magnet_radius_m": 0.005})
    assert spec.turns == 800 and spec.magnet_radius_m == 0.005
    assert spec.coil_length_m == base.coil_length_m           # missing knob -> default
    assert S._spec_from_params({}, n_coils=999).n_coils == 40  # clamped
    assert S._spec_from_params({}, n_coils=1).n_coils == 2
    # a categorical/None value must not crash the float conversion -> falls back to default
    assert S._spec_from_params({"turns": None}).turns == base.turns


def test_simulate_candidate_analytic_returns_animation_payload():
    champ = {"n_coils": 12, "turns": 500, "coil_length_m": 0.012, "radial_thickness_m": 0.006,
             "magnet_radius_m": 0.004, "magnet_length_m": 0.014, "remanence_t": 1.25,
             "bus_voltage_v": 450.0, "i_max_a": 100.0}
    d = S.simulate_candidate(champ, force_law="analytic")
    assert d["force_law"] == "analytic"
    L, F, sm = d["layout"], d["frames"], d["summary"]
    assert L["n_coils"] == 12 and len(L["coil_positions_m"]) == 12
    assert L["x_end_m"] > L["x_start_m"]
    assert 2 <= len(F) <= 200                                  # trimmed + downsampled
    f0 = F[0]
    assert {"t", "x", "v", "coil", "i"} <= set(f0)
    assert F[-1]["x"] > F[0]["x"]                              # the slug advances
    assert 0 <= f0["coil"] < 12
    assert sm["v_exit"] > 0 and sm["flight_ms"] > 0 and sm["mass_g"] > 0
    json.dumps(d, default=S._json_default)


def test_render_launch_gif_produces_a_looping_animated_gif():
    import base64
    import io

    from PIL import Image

    sim = S.simulate_candidate({"n_coils": 8, "turns": 450}, force_law="analytic")
    uri = S.render_launch_gif(sim, dark=True)
    assert uri.startswith("data:image/gif;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert raw[:6] in (b"GIF89a", b"GIF87a")
    im = Image.open(io.BytesIO(raw))
    assert im.n_frames > 1                                 # actually animated
    assert im.info.get("loop") == 0                        # loops forever
    assert im.size[0] > 0 and im.size[1] > 0


def test_render_launch_gif_handles_an_empty_trajectory():
    assert S.render_launch_gif({"layout": {}, "frames": [], "summary": {}}) is None


# --------------------------------------------------------------------------- serving over the LAN
def test_lan_flag_binds_all_interfaces_but_default_stays_loopback(monkeypatch):
    """--lan must widen the bind to 0.0.0.0; without it the GUI stays private to this machine
    (it can launch tools, so a network-exposed default would be the wrong safety posture)."""
    seen = {}

    def fake_serve(host, port, open_browser=True):
        seen["host"], seen["port"] = host, port

    monkeypatch.setattr(S, "serve", fake_serve)
    S.main(["--no-browser"])
    assert seen["host"] == "127.0.0.1"                    # safe default
    S.main(["--lan", "--no-browser"])
    assert seen["host"] == "0.0.0.0"                      # reachable over Wi-Fi
    S.main(["--host", "192.168.1.50", "--port", "9100", "--no-browser"])
    assert seen["host"] == "192.168.1.50" and seen["port"] == 9100


def test_lan_ip_returns_a_dotted_address_or_none():
    ip = S.lan_ip()
    assert ip is None or (ip.count(".") == 3 and all(p.isdigit() for p in ip.split(".")))


def test_simulate_candidate_femm_requires_femm():
    if S.femm_available():
        pytest.skip("FEMM installed here; the not-installed guard can't be exercised")
    with pytest.raises(ValueError, match="FEMM is not installed"):
        S.simulate_candidate({"n_coils": 8}, force_law="femm")
