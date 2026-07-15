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
carrying `current_a` through a `turns`-turn series circuit.

Force extraction: the axial LORENTZ force on the COIL (FEMM block integral type 12 over the
coil group), which by Newton's third law is equal and opposite to the force on the slug --
and, because the coil is the only current-carrying body, is the total force on the slug's
whole magnetic interaction with it. This deliberately does NOT use the weighted Maxwell
stress tensor over the SLUG block (type 19), which is what a naive reading of FEMM's
PM-actuator examples suggests: empirically (see tests/test_fem_femm_backend.py's
convergence test) the stress tensor integrated over a bare permanent-magnet block does NOT
converge under mesh refinement here -- it is up to ~2x wrong and even flips sign in the far
field -- because the magnet's own huge internal B/H dominates the tensor and the thin
auto-meshed air band around it is too coarse for the weighting function. The coil sits in
linear, non-magnetic copper/air (mu_r = 1) carrying a known current density, so int(J x B)
over it is mesh-robust and exact for this geometry. Validated against the closed-form
AnalyticReferenceBackend (itself checked against plant.f_current_pm): the coil-Lorentz
force agrees to ~1-2% across offsets/geometries and is linear in current to <0.3%, whereas
the old slug-stress-tensor extraction agreed only near-field and diverged with mesh.
"""

from __future__ import annotations

from .backend import ForcePoint
from .geometry import CoilWindingGeometry, SlugGeometry

_AIR_MARGIN_FACTOR = 6.0     # outer boundary radius/extent, multiples of the largest part
NDFEB_RELATIVE_PERMEABILITY = 1.05
MU_0 = 4.0e-7 * 3.141592653589793

_SLUG_GROUP = 1
_COIL_GROUP = 2


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

    def solve(self, coil: CoilWindingGeometry, slug: SlugGeometry,
              offset_m: float, current_a: float) -> ForcePoint:  # pragma: no cover
        femm = self._femm
        self._ensure_open()

        femm.newdocument(0)  # 0 = magnetics problem
        femm.mi_probdef(0, "meters", "axi", 1e-8, 0, 30)

        mesh = self.mesh_size_m or (0.15 * min(coil.radial_thickness_m, slug.magnet_radius_m))

        femm.mi_getmaterial("Air")
        femm.mi_getmaterial("Copper")
        femm.mi_addmaterial("NdFeB", NDFEB_RELATIVE_PERMEABILITY, NDFEB_RELATIVE_PERMEABILITY,
                             slug.remanence_t / (MU_0 * NDFEB_RELATIVE_PERMEABILITY), 0, 0, 0,
                             0, 1, 0, 0, 0, 0)

        outer_r = _AIR_MARGIN_FACTOR * coil.outer_radius_m(slug)
        # The slug is drawn at z=-offset_m, so the air boundary must enclose it with margin
        # at ANY swept offset -- a fixed part-size multiple leaves the slug on/outside the
        # boundary once |offset| approaches it (the default sweep runs out to 5x the coupling
        # scale, well past 6x a small part), which silently returns a spurious near-zero
        # force there. Growing the half-extent with |offset| keeps the slug enclosed.
        half_extent = _AIR_MARGIN_FACTOR * max(coil.coil_length_m, slug.magnet_length_m) + abs(offset_m)

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
        # automesh=0: honor the computed `mesh` size. With automesh=1 (the old value) FEMM
        # ignores meshsize entirely, so mesh_size_m was a silent no-op and force noise
        # (Maxwell-stress symmetry error) sat at ~3.3% instead of the ~0.6% this mesh gives.
        femm.mi_setblockprop("NdFeB", 0, mesh, "<None>", 90, _SLUG_GROUP, 0)
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
        femm.mi_setblockprop("Copper", 0, mesh, circuit, 0, _COIL_GROUP, coil.turns)  # automesh=0 (see above)
        femm.mi_clearselected()

        femm.mi_saveas("_emac_femgen_tmp.fem")
        femm.mi_analyze(1)
        femm.mi_loadsolution()

        # Axial (z) Lorentz force on the coil block -- see this module's docstring for why
        # this, not the weighted stress tensor over the slug. FEMM's raw type-12 value here
        # already carries the SAME sign as fem.reference_backend / linear_plant.net_force's
        # convention (positive offset + positive current => negative, restoring, force
        # pulling the slug back toward the coil), verified point-by-point against
        # AnalyticReferenceBackend -- so no negation is applied. By Newton's third law its
        # magnitude is the force on the slug; because copper/air are linear and mu_r = 1,
        # int(J x B) over the coil is the exact, mesh-robust total interaction force.
        femm.mo_groupselectblock(_COIL_GROUP)
        force_n = femm.mo_blockintegral(12)
        femm.mo_clearblock()

        return ForcePoint(force_n=float(force_n))
