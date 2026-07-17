"""Build fem geometry straight from a LinearSimulationConfig -- the "geometry builder
from TOML" step of the FEM pipeline. Kept as its own module (importing `..config`) rather
than folded into geometry.py, so geometry.py itself stays config-format-agnostic and
config.py -> fem.lut's existing import direction never has to become a cycle."""

from __future__ import annotations

from ..config import LinearCoilConfig, LinearSimulationConfig, SlugConfig
from .geometry import CoilWindingGeometry, SlugGeometry


def slug_geometry_from_config(slug: SlugConfig) -> SlugGeometry:
    return SlugGeometry(
        magnet_radius_m=slug.magnet_radius_m,
        magnet_length_m=slug.magnet_length_m,
        remanence_t=slug.remanence_t,
        slug_type=slug.slug_type,
    )


def coil_geometry_from_config(coil: LinearCoilConfig) -> CoilWindingGeometry:
    return CoilWindingGeometry(
        position_m=coil.position_m,
        turns=coil.turns,
        coil_length_m=coil.coil_winding_length_m,
        radial_thickness_m=coil.radial_thickness_m,
        bore_clearance_m=coil.bore_clearance_m,
        packing_factor=coil.packing_factor,
        temperature_c=coil.winding_temperature_c,
    )


def geometry_from_config(config: LinearSimulationConfig) -> tuple[SlugGeometry, list[CoilWindingGeometry]]:
    """(slug geometry, one coil geometry per `[[coils]]` entry), in config order."""
    slug = slug_geometry_from_config(config.slug)
    coils = [coil_geometry_from_config(c) for c in config.coils]
    return slug, coils
