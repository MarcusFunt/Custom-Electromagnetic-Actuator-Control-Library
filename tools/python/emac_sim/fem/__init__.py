"""FEM axisymmetric table-generation pipeline for the linear stepper's PM/coil coupling.

`coil_design.py` already estimates each coil's PM-branch gain (`k_a`) and coupling
half-width (`x_c`) analytically, from a closed-form Biot-Savart integral over an ideal
uniformly-magnetized cylinder. That is fast and good enough to make "optimize over turns
and dimensions" meaningful, but it assumes vacuum permeability everywhere, an idealized
magnet, and (for `net_force`) `plant.q_shape`'s Gaussian-lobe SHAPE for how coupling falls
off with slug offset -- not the coil's own actual field profile.

This package replaces that shape assumption with a real axisymmetric field solve, table,
and interpolation, while keeping the analytic path as the default (nothing here is used
unless a coil config opts in via `force_lut_path` / `emac-femgen`):

  geometry.py   -- coil winding + slug dimensions, built from the SAME physical knobs
                   `coil_design.py` already uses (turns, coil_length_m, radial_thickness_m,
                   magnet_radius_m, magnet_length_m, remanence_t) -- see config.py's
                   LinearCoilConfig/SlugConfig for how a TOML config supplies them.
  backend.py    -- FEMBackend protocol: solve(coil, slug, offset_m, current_a) -> ForcePoint.
  reference_backend.py -- an analytic backend (Biot-Savart via coil_design's own field
                   functions, evaluated AT the requested offset instead of only at its peak)
                   that satisfies the same interface. This is NOT a real FEM solver -- it
                   exists so the geometry/sweep/LUT/plant plumbing can be built and tested
                   end to end on a machine without FEMM installed. See its docstring.
  femm_backend.py -- the real backend: builds an axisymmetric magnetostatic problem in
                   FEMM (via the optional `femm` python module) and solves it with the
                   Maxwell stress tensor for force. Raises FemmNotAvailableError with a
                   clear message if FEMM/pyfemm isn't installed.
  sweep.py      -- sweeps a backend over a position x current grid into a ForceLUT.
  lut.py        -- ForceLUT: save/load (.npz) + edge-clamped bilinear interpolation,
                   callable as (offset_m, current_a) -> force_n.
  cli.py        -- `emac-femgen`: config in, per-coil LUT files out.
"""
