import numpy as np
import pytest

from emac_sim.fem.lut import ForceLUT


def make_lut(**overrides) -> ForceLUT:
    offsets = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    currents = np.array([-3.0, 0.0, 3.0])
    # force = offset * current -- a simple bilinear-exact surface, so linear
    # interpolation should reproduce it exactly everywhere, not just at grid points.
    force = np.outer(offsets, currents)
    kwargs = dict(offsets_m=offsets, currents_a=currents, force_n=force, metadata={"note": "test"})
    kwargs.update(overrides)
    return ForceLUT(**kwargs)


def test_rejects_mismatched_shape():
    offsets = np.array([0.0, 1.0])
    currents = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError):
        ForceLUT(offsets_m=offsets, currents_a=currents, force_n=np.zeros((2, 2)))


def test_rejects_non_increasing_axes():
    offsets = np.array([0.0, -1.0])
    currents = np.array([0.0, 1.0])
    with pytest.raises(ValueError):
        ForceLUT(offsets_m=offsets, currents_a=currents, force_n=np.zeros((2, 2)))


def test_rejects_too_few_points():
    with pytest.raises(ValueError):
        ForceLUT(offsets_m=np.array([0.0]), currents_a=np.array([0.0, 1.0]), force_n=np.zeros((1, 2)))


def test_call_reproduces_grid_points_exactly():
    lut = make_lut()
    assert lut(0.01, 3.0) == pytest.approx(0.03)
    assert lut(-0.02, -3.0) == pytest.approx(0.06)
    assert lut(0.0, 3.0) == pytest.approx(0.0)


def test_call_interpolates_between_grid_points():
    lut = make_lut()
    # bilinear on offset*current is exact for this separable surface
    assert lut(0.005, 3.0) == pytest.approx(0.015, abs=1e-9)
    assert lut(0.01, 1.5) == pytest.approx(0.015, abs=1e-9)


def test_call_clamps_to_grid_edges_instead_of_extrapolating():
    lut = make_lut()
    edge_value = lut(0.02, 3.0)
    assert lut(5.0, 3.0) == pytest.approx(edge_value)
    assert lut(0.02, 500.0) == pytest.approx(edge_value)
    far_edge_value = lut(-0.02, -3.0)
    assert lut(-99.0, -99.0) == pytest.approx(far_edge_value)


def test_save_load_roundtrip(tmp_path):
    lut = make_lut()
    path = tmp_path / "sub" / "coil.npz"
    lut.save(path)
    assert path.exists()

    loaded = ForceLUT.load(path)
    assert loaded.metadata == {"note": "test"}
    np.testing.assert_allclose(loaded.offsets_m, lut.offsets_m)
    np.testing.assert_allclose(loaded.currents_a, lut.currents_a)
    np.testing.assert_allclose(loaded.force_n, lut.force_n)
    assert loaded(0.005, 3.0) == pytest.approx(lut(0.005, 3.0))
