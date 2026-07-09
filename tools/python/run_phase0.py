"""Backward-compatible wrapper for the Phase 0 demo.

Prefer the installed console script:

    emac-phase0 --outdir build/phase0
"""

from emac_sim.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
