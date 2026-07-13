"""Generate the full FEMM-study figure set into study/figures/. Reads existing results only
(no FEMM). Run: python make_figures.py [subdir]"""
import os, sys, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import study_viz as V

SUB = sys.argv[1] if len(sys.argv) > 1 else None
OUT = os.path.join(str(V.HERE), "figures")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9.5, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})

rows = V.load(SUB)
r, y, X = V.arrays(rows, "femm")
BI = np.isclose(X["driver_bipolar"], 1.0)
SHORT = {"bus_voltage_v":"V", "i_max_a":"Imax", "turns":"N", "coil_length_m":"Lcoil",
         "radial_thickness_m":"Twind", "magnet_radius_m":"Rmag", "magnet_length_m":"Lmag",
         "remanence_t":"Br", "driver_bipolar":"Bipolar", "pump_envelope":"Square"}


def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.relpath(p, str(V.HERE)))


def fig_main_effect_curves():
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    fig.suptitle("Main-effect curves — mean real-FEMM exit speed vs each knob (±SEM)",
                 fontweight="bold", fontsize=12)
    order = ["driver_bipolar","coil_length_m","bus_voltage_v","remanence_t","i_max_a",
             "pump_envelope","turns","magnet_radius_m","magnet_length_m","radial_thickness_m"]
    for ax, k in zip(axes.flat, order):
        lv, m, s = V.level_stats(y, X[k])
        xs = V.axis_scale(k, lv)
        col = V.ORANGE if (m[-1] - m[0]) >= 0 else V.RED
        ax.errorbar(xs, m, yerr=s, marker="o", color=col, lw=2, ms=6, capsize=3)
        if k == "driver_bipolar": ax.set_xticks([0, 1]); ax.set_xticklabels(["uni", "bi"])
        if k == "pump_envelope": ax.set_xticks([0, 1]); ax.set_xticklabels(["rcos", "square"])
        ax.set_title(V.LABEL[k], fontsize=10)
        ax.set_ylabel("exit speed (m/s)")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, "fig1_main_effect_curves.png")


def fig_polarity_moderation():
    knobs = ["bus_voltage_v","i_max_a","remanence_t","turns","coil_length_m",
             "magnet_radius_m","magnet_length_m","radial_thickness_m"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    fig.suptitle("Polarity moderation — every 'more force' knob pays off far more under bipolar drive",
                 fontweight="bold", fontsize=12)
    for ax, k in zip(axes.flat, knobs):
        for mask, col, name in [(~BI, V.BLUE, "unipolar"), (BI, V.ORANGE, "bipolar")]:
            lv, _, _ = V.level_stats(y[mask], X[k][mask])
            m = np.array([y[mask & np.isclose(X[k], v)].mean() for v in lv])
            ax.plot(V.axis_scale(k, lv), m, "o-", color=col, lw=2, ms=5.5, label=name)
        # slope ratio annotation
        su = np.polyfit(X[k][~BI], y[~BI], 1)[0]; sb = np.polyfit(X[k][BI], y[BI], 1)[0]
        ratio = sb/su if abs(su) > 1e-9 else float('nan')
        ax.set_title(f"{V.LABEL[k]}  (slope ×{ratio:.1f})", fontsize=10)
        ax.set_ylabel("exit speed (m/s)")
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, "fig2_polarity_moderation.png")


def fig_interaction_heatmap():
    beta = V.std_ols(y, X)
    n = len(V.ALL); M = np.full((n, n), np.nan)
    for i, a in enumerate(V.ALL):
        M[i, i] = beta[a]
        for j, b in enumerate(V.ALL):
            if j > i:
                v = beta.get((a, b), beta.get((b, a)))
                M[i, j] = M[j, i] = v
    labs = [SHORT[k] for k in V.ALL]
    vmax = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax))
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labs)
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if abs(M[i,j]) > 0.6*vmax else "black")
    ax.set_title("Standardized effects: diagonal = main effect, off-diagonal = 2-way interaction\n"
                 "(Δ m/s per 1 SD; red = raises speed, blue = lowers)", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04, label="standardized β (m/s)")
    fig.tight_layout(); save(fig, "fig3_interaction_heatmap.png")


def _grid(ka, kb, mask):
    la = sorted(set(X[ka][mask])); lb = sorted(set(X[kb][mask]))
    G = np.full((len(lb), len(la)), np.nan)
    for i, vb in enumerate(lb):
        for j, va in enumerate(la):
            sel = mask & np.isclose(X[ka], va) & np.isclose(X[kb], vb)
            if sel.any(): G[i, j] = y[sel].mean()
    return la, lb, G


def fig_design_heatmaps():
    pairs = [("bus_voltage_v","i_max_a"), ("bus_voltage_v","remanence_t"),
             ("coil_length_m","radial_thickness_m"), ("magnet_radius_m","magnet_length_m")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Design maps — mean exit speed over knob pairs (bipolar drive only)",
                 fontweight="bold", fontsize=12)
    for ax, (ka, kb) in zip(axes.flat, pairs):
        la, lb, G = _grid(ka, kb, BI)
        im = ax.imshow(G, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(la))); ax.set_xticklabels([f"{v*1000:.0f}" if ka in V.MM else f"{v:g}" for v in la])
        ax.set_yticks(range(len(lb))); ax.set_yticklabels([f"{v*1000:.0f}" if kb in V.MM else f"{v:g}" for v in lb])
        ax.set_xlabel(V.LABEL[ka]); ax.set_ylabel(V.LABEL[kb]); ax.grid(False)
        for i in range(len(lb)):
            for j in range(len(la)):
                if not np.isnan(G[i, j]):
                    ax.text(j, i, f"{G[i,j]:.1f}", ha="center", va="center", fontsize=8,
                            color="white" if G[i,j] < np.nanmax(G)*0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m/s")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, "fig4_design_heatmaps.png")


def fig_analytic_vs_femm():
    key = lambda x: (x["cell_id"], x["bus_voltage_v"], x["driver_bipolar"], x["pump_envelope"], x["i_max_a"])
    a = {key(x): x["exit_speed_mps"] for x in rows if x["force_law"] == "analytic"}
    f = {key(x): (x["exit_speed_mps"], x["driver_bipolar"]) for x in rows if x["force_law"] == "femm"}
    K = [k for k in a.keys() & f.keys() if a[k] is not None and f[k][0] is not None]
    av = np.array([a[k] for k in K]); fv = np.array([f[k][0] for k in K]); bip = np.array([f[k][1] for k in K])
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Cheap analytic model vs real FEMM", fontweight="bold", fontsize=12)
    for m, col, name in [(~bip, V.BLUE, "unipolar"), (bip, V.ORANGE, "bipolar")]:
        ax[0].scatter(fv[m], av[m], s=9, alpha=0.4, color=col, edgecolors="none", label=name)
    lim = max(av.max(), fv.max())*1.05
    ax[0].plot([0, lim], [0, lim], "k--", lw=1, label="agree")
    ax[0].set(xlabel="real-FEMM exit speed (m/s)", ylabel="analytic exit speed (m/s)", xlim=(0, lim), ylim=(0, lim))
    ax[0].legend(frameon=False); ax[0].set_title("analytic sits above the line → overpredicts")
    # per-factor-level median overprediction
    both = (av > 0.1) & (fv > 0.1); rel = (av[both] - fv[both]) / fv[both] * 100
    labels, vals = [], []
    for k in V.ALL:
        xk = np.array([V.numeric(next(x for x in rows if x["force_law"]=="femm" and key(x)==kk), k) for kk in K])[both]
        for lv in sorted(set(xk)):
            sel = np.isclose(xk, lv)
            if sel.sum() >= 10:
                nm = SHORT[k] + "=" + (("bi" if lv else "uni") if k=="driver_bipolar" else ("sq" if lv else "rc") if k=="pump_envelope" else (f"{lv*1000:.0f}" if k in V.MM else f"{lv:g}"))
                labels.append(nm); vals.append(np.median(rel[sel]))
    orderi = np.argsort(vals)
    labels = [labels[i] for i in orderi]; vals = [vals[i] for i in orderi]
    cols = [V.RED if v > np.median(rel) else V.GREEN for v in vals]
    ax[1].barh(range(len(vals)), vals, color=cols)
    ax[1].axvline(np.median(rel), color="k", lw=1, ls="--", label=f"overall median +{np.median(rel):.0f}%")
    ax[1].set_yticks(range(len(vals))); ax[1].set_yticklabels(labels, fontsize=7)
    ax[1].set(xlabel="median analytic overprediction (%)", title="where the cheap model errs most")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, "fig5_analytic_vs_femm.png")


def fig_feasibility_pareto():
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle("Feasibility, speed distribution, and the speed-vs-mass trade-off", fontweight="bold", fontsize=12)
    # (a) stall rate by factor level
    labels, rates = [], []
    for k in ["driver_bipolar","pump_envelope","bus_voltage_v","i_max_a","coil_length_m","remanence_t"]:
        for lv in sorted(set(X[k])):
            sel = np.isclose(X[k], lv)
            nm = SHORT[k]+"="+(("bi" if lv else "uni") if k=="driver_bipolar" else ("sq" if lv else "rc") if k=="pump_envelope" else (f"{lv*1000:.0f}" if k in V.MM else f"{lv:g}"))
            labels.append(nm); rates.append((y[sel] <= 0.5).mean()*100)
    ax[0].barh(range(len(rates)), rates, color=V.BLUE)
    ax[0].set_yticks(range(len(rates))); ax[0].set_yticklabels(labels, fontsize=7)
    ax[0].set(xlabel="stalled / near-stalled designs (%)", title="Feasibility: fraction < 0.5 m/s")
    ax[0].grid(True, axis="x", alpha=0.25)
    # (b) speed distributions by polarity
    ax[1].hist(y[~BI], bins=30, alpha=0.6, color=V.BLUE, label="unipolar")
    ax[1].hist(y[BI], bins=30, alpha=0.6, color=V.ORANGE, label="bipolar")
    ax[1].set(xlabel="exit speed (m/s)", ylabel="designs", title="Exit-speed distribution by polarity")
    ax[1].legend(frameon=False)
    # (c) speed vs magnet mass (light-and-strong tradeoff), bipolar subset
    mass = np.pi * X["magnet_radius_m"]**2 * X["magnet_length_m"] * 7500 * 1000  # grams
    ax[2].scatter(mass[BI], y[BI], s=10, alpha=0.35, color=V.ORANGE, edgecolors="none")
    # pareto front (max speed at/below each mass)
    o = np.argsort(mass[BI]); mm = mass[BI][o]; yy = y[BI][o]; best = np.maximum.accumulate(yy[::-1])[::-1]
    ax[2].plot(mm, best, color=V.RED, lw=2, label="Pareto front (max speed)")
    ax[2].set(xlabel="magnet mass (g)", ylabel="exit speed (m/s)", title="Speed vs magnet mass — lighter is faster")
    ax[2].legend(frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); save(fig, "fig6_feasibility_pareto.png")


if __name__ == "__main__":
    print(f"loaded {len(r)} FEMM designs from {SUB}")
    fig_main_effect_curves()
    fig_polarity_moderation()
    fig_interaction_heatmap()
    fig_design_heatmaps()
    fig_analytic_vs_femm()
    fig_feasibility_pareto()
    print("done ->", os.path.relpath(OUT, str(V.HERE)))
