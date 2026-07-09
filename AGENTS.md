# Repository Instructions

This repo is currently Phase 0 host-only Python simulation work for the EMAC
design. Keep edits focused on the simulator/package unless the user explicitly
asks for firmware.

- Run `gitnexus analyze .` after large structural changes before relying on
  GitNexus MCP results.
- Before modifying an existing function/class/method, run GitNexus impact
  analysis when the tool is available.
- Do not commit generated artifacts: Python caches, pytest caches, build output,
  and generated Phase 0 plots.
- Verify package work with:
  `python -m pip install -e .[dev]`, `python -c "import emac_sim"`,
  `python -m pytest`, and `emac-phase0 --outdir build/phase0`.
