"""Real FEM backend: axisymmetric magnetostatic solve via FEMM (femm.info), driven
through its optional `femm` Python bindings (installed alongside the FEMM application on
Windows, or via `pip install pyfemm`).

FEMM itself is not a pip-installable dependency (it's a separate Windows application), so
this module is import-safe without it -- `import femm` is attempted lazily inside
`FemmBackend.__init__`, not at module load time, and raises `FemmNotAvailableError` with a
clear message pointing at http://www.femm.info/ rather than a bare ModuleNotFoundError.
This is the ONLY module in the fem package that requires FEMM; geometry/reference_backend/
sweep/lut/cli all work without it (see reference_backend.py for the fallback used in this
repo's own tests, which run on a machine without FEMM installed).

Model: an axisymmetric ("axi") magnetostatics problem in meters. The slug is a uniformly
axially-magnetized NdFeB cylinder (r in [0, magnet_radius_m], z centered per `offset_m`);
the coil is a copper rectangular winding block (r in [bore_radius_m, outer_radius_m])
carrying `current_a` through a `turns`-turn series circuit. Force on the slug is read via
FEMM's weighted (Maxwell) stress tensor block integral (type 19, the axial component in
axisymmetric problems) over the slug's block group -- the standard FEMM approach for force
on a body fully surrounded by air, used throughout FEMM's own PM-actuator examples.
"""

from __future__ import annotations

import math
from typing import Callable

from .backend import ForcePoint
from .geometry import CoilWindingGeometry, SlugGeometry

_AIR_MARGIN_FACTOR = 6.0     # outer boundary radius/extent, multiples of the largest part
NDFEB_RELATIVE_PERMEABILITY = 1.05
MU_0 = 4.0e-7 * 3.141592653589793

_SLUG_GROUP = 1
_COIL_GROUP = 2

# field_lines() defaults -- see FemmBackend.field_lines below. max_points is deliberately
# modest: the dashboard schematic only ever shows a padded near-field view around the
# coil/magnet (roughly 1.25x their own dimensions), so tracing far into FEMM's much larger
# solved domain (6x margin, see _AIR_MARGIN_FACTOR) just adds mo_getb calls (wall-clock
# cost) and JSON payload for a curve segment that gets clipped by the SVG viewport anyway.
_FIELD_LINE_N_DEFAULT = 14
_FIELD_LINE_MAX_POINTS = 80
_FIELD_LINE_B_FLOOR_T = 1e-6   # stop tracing once |B| decays below this (avoid a runaway
                               # unit-vector normalization once the field is negligible)


def _trace_field_line(
    b_field_fn: Callable[[float, float], tuple[float, float]],
    seed: tuple[float, float],
    step_m: float,
    direction: float,
    in_bounds: Callable[[float, float], bool],
    max_points: int = _FIELD_LINE_MAX_POINTS,
) -> list[tuple[float, float]]:
    """RK4-integrate a single field line from `seed = (r_m, z_m)` in the meridian
    half-plane, stepping along `direction * (Br,Bz)/|B|` (direction is +1.0 or -1.0 -- a
    field line is traced both ways from its seed and the two halves are joined by the
    caller). Takes a plain `b_field_fn: (r,z) -> (br,bz)` callable rather than the femm
    module directly, so this is unit-testable with a synthetic field and needs no FEMM
    installation. Stops when `in_bounds` goes false (left the solved domain, or crossed the
    r=0 symmetry axis), |B| decays below _FIELD_LINE_B_FLOOR_T, or max_points is hit."""

    def unit_b(r: float, z: float) -> tuple[float, float] | None:
        if not in_bounds(r, z):
            return None
        br, bz = b_field_fn(r, z)
        mag = math.hypot(br, bz)
        if mag < _FIELD_LINE_B_FLOOR_T:
            return None
        return (direction * br / mag, direction * bz / mag)

    points = [seed]
    r, z = seed
    for _ in range(max_points - 1):
        k1 = unit_b(r, z)
        if k1 is None:
            break
        k2 = unit_b(r + 0.5 * step_m * k1[0], z + 0.5 * step_m * k1[1])
        if k2 is None:
            break
        k3 = unit_b(r + 0.5 * step_m * k2[0], z + 0.5 * step_m * k2[1])
        if k3 is None:
            break
        k4 = unit_b(r + step_m * k3[0], z + step_m * k3[1])
        if k4 is None:
            break
        dr = (step_m / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        dz = (step_m / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        r, z = r + dr, z + dz
        if not in_bounds(r, z):
            break
        points.append((r, z))
    return points


def _perimeter_seeds(magnet_radius_m: float, z0: float, z1: float,
                      n: int) -> list[tuple[float, float]]:
    """`n` seed points evenly spaced along the magnet's REAL boundary faces -- the bottom
    cap (z=z0, r: 0->R), the outer curved side (r=R, z: z0->z1), and the top cap (z=z1,
    r: R->0) -- deliberately excluding the r=0 edge itself, which is the symmetry axis, not
    a physical surface of the magnet."""
    total = 2.0 * magnet_radius_m + (z1 - z0)
    seeds = []
    for k in range(n):
        s = total * (k + 0.5) / n   # offset by half a step so no seed lands exactly on a corner
        if s < magnet_radius_m:
            seeds.append((s, z0))
        elif s < magnet_radius_m + (z1 - z0):
            seeds.append((magnet_radius_m, z0 + (s - magnet_radius_m)))
        else:
            seeds.append((magnet_radius_m - (s - magnet_radius_m - (z1 - z0)), z1))
    return seeds


class FemmNotAvailableError(RuntimeError):
    """Raised when the optional `femm` Python module (bundled with the FEMM application,
    http://www.femm.info/) can't be imported -- i.e. FEMM isn't installed on this machine."""


class FemmBackend:
    """FEMBackend backed by a real FEMM axisymmetric magnetostatic solve. One `solve()`
    call is one full build-mesh-solve-measure cycle -- expensive (seconds, not
    microseconds) by FEM standards, which is exactly why sweep.py's output is meant to be
    cached as a LUT and interpolated at simulation time, never called per-timestep."""

    def __init__(self, mesh_size_m: float | None = None, keep_open: bool = False) -> None:
        try:
            import femm  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised only with FEMM installed
            raise FemmNotAvailableError(
                "The optional 'femm' Python module could not be imported. FemmBackend "
                "needs the FEMM application installed (http://www.femm.info/) -- its "
                "installer places femm.py on your Python path, or `pip install pyfemm` "
                "if you're using the PyPI wheel. Use fem.reference_backend."
                "AnalyticReferenceBackend for a FEMM-free fallback."
            ) from exc
        self._femm = femm
        self.mesh_size_m = mesh_size_m
        self.keep_open = keep_open
        self._opened = False

    def _ensure_open(self) -> None:  # pragma: no cover - exercised only with FEMM installed
        if not self._opened:
            self._femm.openfemm()
            self._opened = True

    def close(self) -> None:  # pragma: no cover - exercised only with FEMM installed
        if self._opened:
            self._femm.closefemm()
            self._opened = False

    def __enter__(self) -> "FemmBackend":
        self._ensure_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if not self.keep_open:
            self.close()

    def _build_and_solve(self, coil: CoilWindingGeometry, slug: SlugGeometry,
                          offset_m: float, current_a: float) -> tuple[float, float]:  # pragma: no cover
        """Shared setup for solve() and field_lines(): build the mesh, run mi_analyze, and
        load the solution so mo_* postprocessor calls work. Returns (outer_r, half_extent)
        -- the drawn air-boundary box, in meters -- so callers know the solved domain's
        extent (field_lines uses it to stop a trace once it leaves the domain)."""
        femm = self._femm

        femm.newdocument(0)  # 0 = magnetics problem
        femm.mi_probdef(0, "meters", "axi", 1e-8, 0, 30)

        mesh = self.mesh_size_m or (0.15 * min(coil.radial_thickness_m, slug.magnet_radius_m))

        femm.mi_getmaterial("Air")
        femm.mi_getmaterial("Copper")
        femm.mi_addmaterial("NdFeB", NDFEB_RELATIVE_PERMEABILITY, NDFEB_RELATIVE_PERMEABILITY,
                             slug.remanence_t / (MU_0 * NDFEB_RELATIVE_PERMEABILITY), 0, 0, 0,
                             0, 1, 0, 0, 0, 0)

        outer_r = _AIR_MARGIN_FACTOR * coil.outer_radius_m(slug)
        # Must contain the slug's drawn rectangle (centered at z=-offset_m, half-extent
        # 0.5*magnet_length_m) for EVERY offset_m a caller passes in -- not just offset_m=0.
        # Without the abs(offset_m) term, a sweep's far offsets (see geometry.py's
        # FAR_SPAN_FACTOR, deliberately wide so ForceLUT's edge-clamping is physically
        # valid) draw the slug rectangle partially outside this boundary, leaving part of
        # the domain with no block label -- FEMM then fails mi_analyze with "Material
        # properties have not been defined for all regions" instead of a geometry error,
        # so this was easy to miss without actually running a real FEMM solve at those
        # offsets (see tests/test_fem_femm_backend.py).
        half_extent = _AIR_MARGIN_FACTOR * max(
            coil.coil_length_m, slug.magnet_length_m, abs(offset_m) + slug.magnet_length_m,
        )

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

        # Slug: magnet center sits at z = -offset_m relative to the fixed coil (matching
        # reference_backend's convention -- the coil is the fixed frame, the slug moves).
        slug_z0 = -offset_m - 0.5 * slug.magnet_length_m
        slug_z1 = -offset_m + 0.5 * slug.magnet_length_m
        femm.mi_drawline(0, slug_z0, slug.magnet_radius_m, slug_z0)
        femm.mi_drawline(slug.magnet_radius_m, slug_z0, slug.magnet_radius_m, slug_z1)
        femm.mi_drawline(slug.magnet_radius_m, slug_z1, 0, slug_z1)
        femm.mi_drawline(0, slug_z1, 0, slug_z0)
        femm.mi_addblocklabel(0.5 * slug.magnet_radius_m, -offset_m)
        femm.mi_selectlabel(0.5 * slug.magnet_radius_m, -offset_m)
        # magdir=-90 (not +90): FEMM's axisymmetric magnetization-angle convention put +90
        # along -z here, which inverted the ENTIRE force curve relative to the documented
        # sign contract (backend.py: positive offset_m + positive current_a should attract,
        # i.e. force_n < 0) -- confirmed by comparing point-for-point against
        # reference_backend.AnalyticReferenceBackend, which is exactly this backend's
        # negative at every sampled offset before this fix. See
        # tests/test_fem_femm_backend.py::test_femm_backend_force_sign_matches_reference_backend.
        femm.mi_setblockprop("NdFeB", 1, mesh, "<None>", -90, _SLUG_GROUP, 0)
        femm.mi_clearselected()

        # Coil: fixed at z=0 (the coordinate origin is the coil's own center).
        circuit = "coil"
        femm.mi_addcircprop(circuit, current_a, 1)  # 1 = series circuit
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
        femm.mi_setblockprop("Copper", 1, mesh, circuit, 0, _COIL_GROUP, coil.turns)
        femm.mi_clearselected()

        femm.mi_saveas("_emac_femgen_tmp.fem")
        femm.mi_analyze(1)
        femm.mi_loadsolution()

        return outer_r, half_extent

    def solve(self, coil: CoilWindingGeometry, slug: SlugGeometry,
              offset_m: float, current_a: float) -> ForcePoint:  # pragma: no cover
        femm = self._femm
        self._ensure_open()
        self._build_and_solve(coil, slug, offset_m, current_a)

        femm.mo_groupselectblock(_SLUG_GROUP)
        force_n = femm.mo_blockintegral(19)  # axial weighted stress tensor force
        femm.mo_clearblock()

        return ForcePoint(force_n=float(force_n))

    def field_lines(
        self, coil: CoilWindingGeometry, slug: SlugGeometry, offset_m: float, current_a: float,
        n_lines: int = _FIELD_LINE_N_DEFAULT, max_points: int = _FIELD_LINE_MAX_POINTS,
        step_m: float | None = None,
    ) -> list[list[tuple[float, float]]]:  # pragma: no cover
        """Trace `n_lines` real magnetic field lines through the (r,z) meridian half-plane
        at this operating point, via RK4 integration along FEMM's `mo_getb` (see
        _trace_field_line). Seeds are spread evenly around the slug magnet's drawn
        rectangular boundary (the field source closest to what a control-scheme designer
        cares about -- how the PM couples to the coil), each traced in both directions
        until it leaves the solved domain, crosses the r=0 symmetry axis, or the field
        decays away. Returns one polyline per line, each a list of (r_m, z_m) points."""
        femm = self._femm
        self._ensure_open()
        outer_r, half_extent = self._build_and_solve(coil, slug, offset_m, current_a)

        def b_field(r: float, z: float) -> tuple[float, float]:
            br, bz = femm.mo_getb(r, z)
            return float(br), float(bz)

        def in_bounds(r: float, z: float) -> bool:
            return 0.0 <= r <= outer_r and -half_extent <= z <= half_extent

        step = step_m or 0.1 * min(coil.radial_thickness_m, slug.magnet_radius_m)

        slug_z0 = -offset_m - 0.5 * slug.magnet_length_m
        slug_z1 = -offset_m + 0.5 * slug.magnet_length_m
        seeds = _perimeter_seeds(slug.magnet_radius_m, slug_z0, slug_z1, n_lines)

        lines: list[list[tuple[float, float]]] = []
        for seed in seeds:
            forward = _trace_field_line(b_field, seed, step, +1.0, in_bounds, max_points)
            backward = _trace_field_line(b_field, seed, step, -1.0, in_bounds, max_points)
            line = list(reversed(backward[1:])) + forward
            if len(line) >= 2:
                lines.append(line)

        return lines
