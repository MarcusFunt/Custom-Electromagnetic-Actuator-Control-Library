"""Unified EMAC desktop GUI: one local web app that runs the command-line tools, shows their
live output, launches and cost-estimates FEM sweeps, and visualizes the results -- replacing
the three separate static dashboards (the optimizer dashboard, the FEMM-trends dashboard, and
the standalone `emac-visual` page). Start it with `emac-gui`."""

from .server import main

__all__ = ["main"]
