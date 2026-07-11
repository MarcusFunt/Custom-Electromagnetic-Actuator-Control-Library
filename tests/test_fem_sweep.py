import pytest

from emac_sim.fem.backend import ForcePoint
from emac_sim.fem.geometry import CoilWindingGeometry, SlugGeometry
from emac_sim.fem.reference_backend import AnalyticReferenceBackend
from emac_sim.fem.sweep import sweep_coil


def default_slug() -> SlugGeometry:
    return SlugGeometry(magnet_radius_m=0.008, magnet_length_m=0.020, remanence_t=1.2)


def default_coil() -> CoilWindingGeometry:
    return CoilWindingGeometry(position_m=0.03, turns=200, coil_length_m=0.020, radial_thickness_m=0.010)


class RecordingBackend:
    """A trivial FEMBackend stub -- lets tests assert sweep_coil calls the backend at
    exactly the requested grid points, without depending on AnalyticReferenceBackend's
    actual physics."""

    def __init__(self):
        self.calls = []

    def solve(self, coil, slug, offset_m, current_a) -> ForcePoint:
        self.calls.append((offset_m, current_a))
        return ForcePoint(force_n=offset_m * current_a)


def test_sweep_coil_visits_every_grid_point_once():
    backend = RecordingBackend()
    slug, coil = default_slug(), default_coil()
    offsets = [-0.01, 0.0, 0.01]
    currents = [-2.0, 2.0]

    lut = sweep_coil(coil, slug, backend, offsets_m=offsets, currents_a=currents)

    assert len(backend.calls) == len(offsets) * len(currents)
    assert set(backend.calls) == {(o, c) for o in offsets for c in currents}
    assert lut.force_n.shape == (3, 2)
    assert lut(0.01, 2.0) == pytest.approx(0.02)


def test_sweep_coil_metadata_records_geometry():
    backend = AnalyticReferenceBackend()
    slug, coil = default_slug(), default_coil()
    lut = sweep_coil(coil, slug, backend, offsets_m=[-0.01, 0.0, 0.01], currents_a=[-1.0, 1.0])

    assert lut.metadata["coil_position_m"] == pytest.approx(coil.position_m)
    assert lut.metadata["turns"] == coil.turns
    assert lut.metadata["magnet_radius_m"] == pytest.approx(slug.magnet_radius_m)
    assert lut.metadata["backend"] == "AnalyticReferenceBackend"


def test_sweep_coil_default_ranges_used_when_omitted():
    backend = RecordingBackend()
    slug, coil = default_slug(), default_coil()
    lut = sweep_coil(coil, slug, backend)
    assert lut.force_n.shape[0] > 1
    assert lut.force_n.shape[1] > 1


def test_on_point_callback_reports_progress():
    backend = RecordingBackend()
    slug, coil = default_slug(), default_coil()
    seen = []
    sweep_coil(coil, slug, backend, offsets_m=[0.0, 0.01], currents_a=[1.0, 2.0, 3.0],
               on_point=lambda i, j, n_done, n_total: seen.append((i, j, n_done, n_total)))

    assert len(seen) == 6
    assert seen[-1] == (1, 2, 6, 6)
