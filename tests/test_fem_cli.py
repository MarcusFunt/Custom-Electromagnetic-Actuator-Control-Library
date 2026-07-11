import json

import pytest

from emac_sim.config import load_config
from emac_sim.fem.cli import build_arg_parser, generate_luts, main
from emac_sim.fem.lut import ForceLUT

EXAMPLE_LINEAR_CONFIG = "examples/configs/linear_stepper_5coil.toml"


def test_arg_parser_defaults():
    args = build_arg_parser().parse_args(["--config", EXAMPLE_LINEAR_CONFIG])
    assert args.backend == "reference"
    assert args.outdir == "build/fem_lut"
    assert args.coils is None
    assert args.n_offsets == 31
    assert args.n_currents == 11


def test_generate_luts_writes_one_file_per_requested_coil(tmp_path):
    config = load_config(EXAMPLE_LINEAR_CONFIG)
    messages = []
    manifest = generate_luts(
        config, "reference", tmp_path, n_offsets=5, n_currents=3, max_current_a=4.0,
        coil_indices=[0, 2], report=messages.append,
    )

    assert len(manifest["coils"]) == 2
    assert manifest["backend"] == "reference"
    assert messages  # progress was reported

    for entry in manifest["coils"]:
        lut = ForceLUT.load(entry["path"])
        assert lut.force_n.shape == (5, 3)
        assert lut.currents_a[0] == pytest.approx(-4.0)
        assert lut.currents_a[-1] == pytest.approx(4.0)

    assert (tmp_path / "manifest.json").exists()
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest


def test_generate_luts_defaults_to_every_coil(tmp_path):
    config = load_config(EXAMPLE_LINEAR_CONFIG)
    manifest = generate_luts(config, "reference", tmp_path, n_offsets=3, n_currents=3,
                              max_current_a=None)
    assert len(manifest["coils"]) == len(config.coils)


def test_main_rejects_pendulum_config(capsys):
    rc = main(["--config", "examples/configs/pendulum_softiron_1gate.toml"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "linear_stepper" in captured.err


def test_main_end_to_end_writes_manifest(tmp_path):
    outdir = tmp_path / "out"
    rc = main([
        "--config", EXAMPLE_LINEAR_CONFIG, "--outdir", str(outdir), "--coil", "0",
        "--n-offsets", "3", "--n-currents", "3", "--quiet",
    ])
    assert rc == 0
    assert (outdir / "manifest.json").exists()
    assert (outdir / "coil_00.npz").exists()


def test_femm_backend_selection_without_femm_installed_raises_clear_error(tmp_path):
    try:
        import femm  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("femm is installed in this environment -- see test_fem_femm_backend.py instead")

    from emac_sim.fem.femm_backend import FemmNotAvailableError

    with pytest.raises(FemmNotAvailableError):
        main(["--config", EXAMPLE_LINEAR_CONFIG, "--outdir", str(tmp_path), "--backend", "femm",
              "--coil", "0", "--n-offsets", "2", "--n-currents", "2"])
