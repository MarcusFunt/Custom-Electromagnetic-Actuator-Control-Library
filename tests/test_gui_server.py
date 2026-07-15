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
        assert spec["module"].startswith("emac_sim.")
        for arg in spec["args"]:
            assert {"name", "type"} <= set(arg)
            assert arg["type"] in {"config", "text", "number", "int", "bool", "choice"}


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
