"""Shared data-loading + helpers for the FEMM study analysis/figure tools.

Keeps analyze_study.py (text trends) and make_figures.py (figures) reading the results the
same way, with pretty labels/levels in one place."""
from __future__ import annotations
import glob, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONT = ["bus_voltage_v", "i_max_a", "turns", "coil_length_m", "radial_thickness_m",
        "magnet_radius_m", "magnet_length_m", "remanence_t"]
BOOL01 = {"driver_bipolar": {False: 0.0, True: 1.0}, "pump_envelope": {"rcos": 0.0, "square": 1.0}}
ALL = CONT + list(BOOL01)

LABEL = {
    "bus_voltage_v": "bus voltage (V)", "i_max_a": "current cap (A)", "turns": "turns",
    "coil_length_m": "coil length (mm)", "radial_thickness_m": "winding thickness (mm)",
    "magnet_radius_m": "magnet radius (mm)", "magnet_length_m": "magnet length (mm)",
    "remanence_t": "remanence (T)", "driver_bipolar": "drive polarity",
    "pump_envelope": "pulse shape",
}
# knobs expressed in mm for readability on axes
MM = {"coil_length_m", "radial_thickness_m", "magnet_radius_m", "magnet_length_m"}

# colorblind-safe (Okabe-Ito / seaborn muted)
BLUE, ORANGE, GREEN, PURPLE, RED, GREY = "#4C72B0", "#DD8452", "#55A868", "#8172B3", "#C44E52", "#888888"


def load(subdir=None):
    """Load every design row from results/. `subdir` is accepted for CLI compatibility but
    the dataset lives next to these scripts (studies/femm_trends/results/)."""
    base = HERE if subdir in (None, ".", "", "study") else HERE / subdir
    rows = []
    for f in glob.glob(str(base / "results" / "cell_*.jsonl")):
        rows += [json.loads(l) for l in Path(f).read_text().splitlines() if l.strip()]
    return rows


def numeric(row, key):
    return BOOL01[key][row[key]] if key in BOOL01 else float(row[key])


def arrays(rows, force_law="femm"):
    r = [x for x in rows if x["force_law"] == force_law and x.get("exit_speed_mps") is not None]
    y = np.array([x["exit_speed_mps"] for x in r], float)
    X = {k: np.array([numeric(x, k) for x in r], float) for k in ALL}
    return r, y, X


def axis_scale(knob, values):
    v = np.array(values, float)
    return v * 1000.0 if knob in MM else v


def level_stats(y, xk):
    """sorted levels, mean, sem for one knob."""
    lv = sorted(set(xk))
    means = np.array([y[np.isclose(xk, v)].mean() for v in lv])
    sems = np.array([y[np.isclose(xk, v)].std() / max(1, np.sqrt((np.isclose(xk, v)).sum())) for v in lv])
    return np.array(lv), means, sems


def std_ols(y, X):
    """standardized main + all 2-way interaction betas; returns dict name->beta and z-dict."""
    import itertools
    z = {k: (X[k] - X[k].mean()) / (X[k].std() or 1.0) for k in ALL}
    cols, names = [np.ones_like(y)], ["intercept"]
    for k in ALL:
        cols.append(z[k]); names.append(k)
    for a, b in itertools.combinations(ALL, 2):
        cols.append(z[a] * z[b]); names.append((a, b))
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
    return dict(zip(names, beta))
