import pytest

from emac_sim.config import LinearCoilConfig, SlugConfig, parse_config
from emac_sim.fem.from_config import (
    coil_geometry_from_config,
    geometry_from_config,
    slug_geometry_from_config,
)


def test_slug_geometry_from_config_maps_fields():
    slug_cfg = SlugConfig(magnet_radius_m=0.01, magnet_length_m=0.03, remanence_t=1.3)
    geom = slug_geometry_from_config(slug_cfg)
    assert geom.magnet_radius_m == pytest.approx(0.01)
    assert geom.magnet_length_m == pytest.approx(0.03)
    assert geom.remanence_t == pytest.approx(1.3)


def test_coil_geometry_from_config_maps_fields():
    coil_cfg = LinearCoilConfig(
        position_m=0.05, turns=150, coil_winding_length_m=0.018,
        radial_thickness_m=0.009, bore_clearance_m=0.001, packing_factor=0.75,
        winding_temperature_c=40.0,
    )
    geom = coil_geometry_from_config(coil_cfg)
    assert geom.position_m == pytest.approx(0.05)
    assert geom.turns == 150
    assert geom.coil_length_m == pytest.approx(0.018)
    assert geom.radial_thickness_m == pytest.approx(0.009)
    assert geom.bore_clearance_m == pytest.approx(0.001)
    assert geom.packing_factor == pytest.approx(0.75)
    assert geom.temperature_c == pytest.approx(40.0)


def test_geometry_from_config_covers_every_coil_in_order():
    config = parse_config({
        "sim": {"kind": "linear_stepper"},
        "coils": [{"position_m": 0.0}, {"position_m": 0.05}, {"position_m": 0.10}],
    })
    slug_geom, coil_geoms = geometry_from_config(config)
    assert len(coil_geoms) == 3
    assert [c.position_m for c in coil_geoms] == pytest.approx([0.0, 0.05, 0.10])
    assert slug_geom.magnet_radius_m > 0.0


def test_slug_config_defaults_and_toml_override():
    default = parse_config({"sim": {"kind": "linear_stepper"}})
    assert default.slug.magnet_radius_m == pytest.approx(0.008)
    assert default.slug.magnet_length_m == pytest.approx(0.020)
    assert default.slug.remanence_t == pytest.approx(1.2)

    custom = parse_config({
        "sim": {"kind": "linear_stepper"},
        "slug": {"magnet_radius_m": 0.006, "magnet_length_m": 0.025, "remanence_t": 1.4},
    })
    assert custom.slug.magnet_radius_m == pytest.approx(0.006)
    assert custom.slug.magnet_length_m == pytest.approx(0.025)
    assert custom.slug.remanence_t == pytest.approx(1.4)
