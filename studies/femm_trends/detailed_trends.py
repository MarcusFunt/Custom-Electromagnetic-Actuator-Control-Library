"""Deep-dive numerical trends for the FEMM study (companion to analyze_study.py's headline
report and make_figures.py's figures). Reads results only. Run: python detailed_trends.py"""
import sys
import numpy as np
import study_viz as V

SUB = sys.argv[1] if len(sys.argv) > 1 else "study"
rows = V.load(SUB)
r, y, X = V.arrays(rows, "femm")
BI = np.isclose(X["driver_bipolar"], 1.0)


def h(t): print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def fmt_lv(k, v):
    if k == "driver_bipolar": return "bipolar" if v else "unipolar"
    if k == "pump_envelope": return "square" if v else "rcos"
    return f"{v*1000:g}mm" if k in V.MM else f"{v:g}"


def shape(m):
    m = np.asarray(m); d = np.diff(m)
    if np.all(d > 0):
        return "saturating (rising, concave)" if len(d) > 1 and d[-1] < d[0]*0.6 else "rising"
    if np.all(d < 0):
        return "falling"
    imax = int(np.argmax(m))
    if 0 < imax < len(m)-1: return f"peaks @ level {imax+1}"
    return "mixed"


def sec_levels():
    h("1. PER-LEVEL MAIN EFFECTS (mean real-FEMM exit speed at each level, all-designs pooled)")
    for k in V.ALL:
        lv, m, s = V.level_stats(y, X[k])
        cells = "   ".join(f"{fmt_lv(k,v)}:{mm:.2f}" for v, mm in zip(lv, m))
        rng = m.max() - m.min()
        print(f"\n{V.LABEL[k]:<20} [{shape(m)}, span {rng:.2f} m/s]")
        print(f"   {cells}")


def sec_moderation():
    h("2. MODERATION — slope of each knob under unipolar vs bipolar drive (m/s per unit)")
    print(f"{'knob':<20}{'unipolar':>12}{'bipolar':>12}{'ratio':>9}   interpretation")
    out = []
    for k in V.CONT:
        su = np.polyfit(X[k][~BI], y[~BI], 1)[0]; sb = np.polyfit(X[k][BI], y[BI], 1)[0]
        out.append((abs(sb-su), k, su, sb))
    for _, k, su, sb in sorted(out, reverse=True):
        ratio = sb/su if abs(su) > 1e-9 else float('nan')
        note = "sign flip!" if su*sb < 0 else ("bipolar amplifies" if abs(sb) > abs(su) else "")
        print(f"{V.LABEL[k]:<20}{su:>12.4f}{sb:>12.4f}{ratio:>9.2f}   {note}")
    # envelope moderation of voltage
    for env, name in [(0.0,"rcos"),(1.0,"square")]:
        m = np.isclose(X["pump_envelope"], env)
        s = np.polyfit(X["bus_voltage_v"][m], y[m], 1)[0]
        print(f"   voltage slope under {name:<7}: {s*100:+.3f} m/s per 100 V")


def sec_diminishing():
    h("3. DIMINISHING RETURNS — marginal gain per voltage step (bipolar drive)")
    lv, m, _ = V.level_stats(y[BI], X["bus_voltage_v"][BI])
    for i in range(1, len(lv)):
        dv = lv[i]-lv[i-1]; dm = m[i]-m[i-1]
        print(f"   {lv[i-1]:.0f}->{lv[i]:.0f} V (+{dv:.0f} V): +{dm:.2f} m/s  ({dm/dv*100:+.3f} m/s per 100 V)")
    print("   => marginal m/s per volt shrinks with voltage: the returns are concave (saturating).")
    # same for turns (non-monotonic)
    lv, m, _ = V.level_stats(y[BI], X["turns"][BI])
    print(f"\n   turns (bipolar): " + "  ".join(f"{int(v)}t:{mm:.2f}" for v,mm in zip(lv,m)) +
          f"  -> {shape(m)} (best near {int(lv[np.argmax(m)])} turns)")


def sec_feasibility():
    h("4. FEASIBILITY — fraction of designs that stall (< 0.5 m/s), by factor level")
    for k in ["driver_bipolar","pump_envelope","bus_voltage_v","i_max_a","remanence_t","coil_length_m"]:
        parts = []
        for v in sorted(set(X[k])):
            sel = np.isclose(X[k], v)
            parts.append(f"{fmt_lv(k,v)}:{(y[sel]<=0.5).mean()*100:.1f}%")
        print(f"   {V.LABEL[k]:<18} " + "  ".join(parts))


def sec_analytic():
    h("5. WHERE THE CHEAP 'ANALYTIC' MODEL ERRS MOST (median overprediction vs FEMM, by level)")
    key = lambda x:(x["cell_id"],x["bus_voltage_v"],x["driver_bipolar"],x["pump_envelope"],x["i_max_a"])
    a = {key(x):x["exit_speed_mps"] for x in rows if x["force_law"]=="analytic"}
    f = {key(x):x["exit_speed_mps"] for x in rows if x["force_law"]=="femm"}
    K = [k for k in a.keys()&f.keys() if a[k] and f[k] and a[k]>0.1 and f[k]>0.1]
    rel = {k:(a[k]-f[k])/f[k]*100 for k in K}
    fr = [x for x in rows if x["force_law"]=="femm" and key(x) in set(K)]
    idx = {key(x):x for x in fr}
    scored=[]
    for k in V.ALL:
        for v in sorted(set(V.numeric(x,k) for x in fr)):
            vals=[rel[kk] for kk in K if np.isclose(V.numeric(idx[kk],k),v)]
            if len(vals)>=10: scored.append((np.median(vals), f"{V.LABEL[k]}={fmt_lv(k,v)}"))
    scored.sort(reverse=True)
    print("   most overpredicted (cheap model most optimistic here):")
    for md,nm in scored[:6]: print(f"      {nm:<32} +{md:.0f}%")
    print("   least overpredicted:")
    for md,nm in scored[-4:]: print(f"      {nm:<32} +{md:.0f}%")


def sec_best():
    h("6. BEST DESIGNS PER CONSTRAINT (real-FEMM exit speed)")
    def best(mask, label):
        if not mask.any(): return
        i = np.argmax(np.where(mask, y, -1))
        x = r[i]
        print(f"   {label:<26} {y[i]:5.2f} m/s  V={x['bus_voltage_v']:.0f} "
              f"{'bi' if x['driver_bipolar'] else 'uni'} {x['pump_envelope']} i{x['i_max_a']:.0f} "
              f"N{x['turns']} Lc{x['coil_length_m']*1000:.0f} Tw{x['radial_thickness_m']*1000:.0f} "
              f"Rm{x['magnet_radius_m']*1000:.0f} Lm{x['magnet_length_m']*1000:.0f} Br{x['remanence_t']}")
    best(np.ones_like(y, bool), "overall")
    best(~BI, "best UNIPOLAR")
    for v in sorted(set(X["bus_voltage_v"])):
        best(np.isclose(X["bus_voltage_v"], v), f"best at {v:.0f} V")
    mass = np.pi*X["magnet_radius_m"]**2*X["magnet_length_m"]*7500*1000
    best(mass <= np.percentile(mass, 33), "best light magnet (<=33%ile)")


def sec_elasticity():
    h("7. ELASTICITIES — % change in exit speed per +1% of each knob (at the data mean)")
    for k in V.CONT:
        slope = np.polyfit(X[k], y, 1)[0]
        e = slope * X[k].mean() / y.mean()
        print(f"   {V.LABEL[k]:<20} {e:+.2f}%  per +1%")


for fn in [sec_levels, sec_moderation, sec_diminishing, sec_feasibility, sec_analytic, sec_best, sec_elasticity]:
    fn()
print()
