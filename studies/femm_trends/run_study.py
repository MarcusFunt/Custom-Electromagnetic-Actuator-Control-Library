"""Long unattended real-FEMM design-of-experiments study for the EMAC linear stepper.

  ############################################################################
  #  REQUIRES FEMM.  This harness is not optional-FEMM: its entire purpose is #
  #  to drive REAL axisymmetric FEMM solves, so FEMM is a CORE REQUIREMENT.   #
  #    - install the FEMM application  (http://www.femm.info/), AND           #
  #    - `pip install pyfemm`          (the Python bindings it drives).        #
  #  Without both, every solve raises and the run produces nothing -- there   #
  #  is no analytic fallback here (that is what the reference backend / the    #
  #  committed dataset in ../results are for). Windows-only (FEMM is a native  #
  #  Win32 app driven over ActiveX/COM).                                       #
  ############################################################################

Outer factorial over 6 FEM-relevant GEOMETRY knobs -> one real-FEMM force LUT per cell
(the expensive part). Inner factorial over 4 DRIVER/control knobs -> fast sims reusing that
LUT. Every (geometry x driver) design is evaluated under two force laws ("analytic" = the
optimizer's cheap coupling model, "femm" = the bug-corrected real FEMM solve) so we can
measure both the design trends AND where the cheap model misleads.

Parallelism: runs SERIAL by default (STUDY_WORKERS=1). pyfemm's COM automation is not safe
for multiple concurrent FEMM instances under sustained load (they interfere), and FEMM's
solver is already internally multithreaded so one instance uses many cores anyway -- measured
throughput peaked at only ~1.36x with 3 instances, then dropped. GPU/CUDA is not applicable:
FEMM is a CPU-only native solver and the sim is a trivial scipy ODE loop.

MAX_PATH: FEMM's solver writes files next to its temp .fem and they MUST stay under Windows'
260-char limit, or the solver fails and FEMM pops a modal dialog that hangs mi_analyze
forever -- so each worker runs FEMM from a short cwd (C:\\femmwork\\wN), not the deep repo path.

Robustness for a ~9 h unattended run:
  - resumable: a cell whose results/cell_XXXX.jsonl exists is skipped; LUTs are cached.
  - per-solve retry with FEMM restart; sparse solve failures are filled, not fatal.
  - each worker restarts its FEMM every RESTART_EVERY cells to bound COM/memory leaks.
  - shared wall-clock budget counted from a persisted first-start epoch (a restart continues
    toward the same deadline).  Atomic result writes (tmp + os.replace).
  - SEEDED-SHUFFLED cell order: if the budget stops the run mid-factorial, whatever finished
    is a balanced random subset of the design space, not a biased corner.
"""
from __future__ import annotations
import itertools, json, os, random, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
# emac_sim package lives at repo_root/tools/python; this file is at repo_root/studies/femm_trends/.
sys.path.insert(0, str(HERE.parents[1] / "tools" / "python"))
sys.path.insert(0, str(HERE))

# ----------------------------- experiment design -----------------------------
GEOM_FACTORS = {
    "turns":             [80, 200, 450, 900],
    "coil_length_m":     [0.010, 0.020, 0.032],   # capped: bigger coils need FEM domains
    "radial_thickness_m":[0.005, 0.014, 0.028],   # FEMM can't mesh (see study writeup)
    "magnet_radius_m":   [0.004, 0.008, 0.015],
    "magnet_length_m":   [0.010, 0.022, 0.036],   # capped for the same reason
    "remanence_t":       [0.5, 0.9, 1.25],
}                                              # 4*3*3*3*3*3 = 972 geometry cells
DRIVER_FACTORS = {
    "bus_voltage_v":  [12.0, 40.0, 110.0, 260.0],
    "driver_bipolar": [False, True],
    "pump_envelope":  ["rcos", "square"],
    "i_max_a":        [20.0, 70.0],
}                                              # 4*2*2*2 = 32 driver designs / cell
FORCE_LAWS = ["analytic", "femm"]
N_COILS = 5

WALLCLOCK_BUDGET_S = float(os.environ.get("STUDY_BUDGET_S", 9.0 * 3600.0))  # >=8h floor, ~3h buffer
MAX_CELLS = int(os.environ.get("STUDY_MAX_CELLS", 0))     # 0 = no cap (smoke-test hook)
N_WORKERS = int(os.environ.get("STUDY_WORKERS", 1))       # serial: pyfemm COM isn't safe
                                                          # for concurrent instances (see writeup)
RESTART_EVERY = 20
SHUFFLE_SEED = 0

BASE = HERE / os.environ.get("STUDY_SUBDIR", "study")
LUTS = BASE / "luts"; RESULTS = BASE / "results"; STATE = BASE / "state"
START_FILE = BASE / "start_epoch.txt"
DONE_FILE = BASE / "DONE"


def all_cells():
    keys = list(GEOM_FACTORS)
    cells = [dict(zip(keys, vals)) for vals in itertools.product(*GEOM_FACTORS.values())]
    random.Random(SHUFFLE_SEED).shuffle(cells)
    return cells


def _fill_nans(force):
    import numpy as np
    n_bad = int(np.isnan(force).sum())
    if n_bad == 0:
        return force, 0
    cur_idx = np.arange(force.shape[1])
    for i in range(force.shape[0]):
        row = force[i]; good = ~np.isnan(row)
        if good.sum() >= 2:   force[i] = np.interp(cur_idx, cur_idx[good], row[good])
        elif good.sum() == 1: force[i] = row[good][0]
    still = np.where(np.isnan(force).any(axis=1))[0]
    good_rows = np.where(~np.isnan(force).any(axis=1))[0]
    for i in still:
        force[i] = force[good_rows[np.argmin(np.abs(good_rows - i))]] if good_rows.size else 0.0
    return force, n_bad


def build_lut(cell, backend, restart, S, np):
    coil, slug, offsets, currents, half_extent = S.femm_sweep_grid(
        cell["turns"], cell["coil_length_m"], cell["radial_thickness_m"],
        cell["magnet_radius_m"], cell["magnet_length_m"], cell["remanence_t"])
    backend._half_extent_override = half_extent
    offsets = np.asarray(offsets); currents = np.asarray(currents)
    force = np.full((offsets.size, currents.size), np.nan)
    consec = 0
    try:
        for i, off in enumerate(offsets):
            for j, cur in enumerate(currents):
                val = float("nan")
                for attempt in range(2):    # one quick retry; NO per-point restart (restart is
                    try:                     # the slow part -- a restart storm cost 4000 s/cell)
                        val = backend.solve(coil, slug, float(off), float(cur)).force_n
                        break
                    except Exception as e:
                        if attempt:
                            print(f"  solve fail off={off:.4f} I={cur:.1f}: {e!r}", flush=True)
                force[i, j] = val
                if val != val:              # NaN
                    consec += 1
                    if consec >= 20:        # FEMM likely wedged -> restart once, then continue
                        print("  20 consecutive fails -> restart FEMM", flush=True)
                        restart(); backend._half_extent_override = half_extent; consec = 0
                else:
                    consec = 0
    finally:
        backend._half_extent_override = None
    force, n_bad = _fill_nans(force)
    if n_bad > 0.3 * force.size:
        raise RuntimeError(f"too many solve failures: {n_bad}/{force.size}")
    from emac_sim.fem.lut import ForceLUT
    lut = ForceLUT(offsets_m=offsets, currents_a=currents, force_n=force,
                   metadata={"backend": "CorrectedFemmBackend", **cell})
    peak = float(np.abs(force).max()) or 1.0
    zc = int(np.argmin(np.abs(currents))); ic = int(np.argmax(np.abs(currents)))
    zoff = int(np.argmin(np.abs(offsets)))
    qc = {"peak_force_n": round(float(np.abs(force).max()), 4),
          "zero_offset_force_frac": round(abs(force[zoff, ic]) / peak, 4),
          "edge_force_frac": round(float(np.abs(force[[0, -1]][:, ic]).max()) / peak, 4),
          "zero_current_max_frac": round(float(np.abs(force[:, zc]).max()) / peak, 4),
          "n_solve_fail": n_bad}
    return lut, qc


def worker(wid, cells, start_epoch):
    """One isolated worker: own temp dir + FEMM instance, processes cells where idx%%N==wid."""
    # SHORT working dir: FEMM's solver writes .fem/.ans/.poly/.node/.ele next to the temp
    # file, and those paths MUST stay under Windows' 260-char MAX_PATH. The deep scratchpad
    # subtree blows past it -> the solver fails and FEMM pops a blocking modal dialog that
    # hangs mi_analyze forever. Keeping FEMM's cwd at C:\femmwork\wN sidesteps that entirely.
    tmpdir = Path(rf"C:\femmwork\w{wid}"); tmpdir.mkdir(parents=True, exist_ok=True)
    os.chdir(tmpdir)
    import numpy as np
    import study_lib as S
    from emac_sim import optimize_design as od
    from emac_sim.fem.lut import ForceLUT

    def log(m): print(f"[{time.strftime('%H:%M:%S')} w{wid}] {m}", flush=True)

    backend = S.CorrectedFemmBackend(keep_open=True); backend._tmp_fem = f"_w{wid}.fem"
    backend._ensure_open()

    def restart():
        try: backend.close()
        except Exception: pass
        time.sleep(1.0); backend._ensure_open()

    driver_combos = [dict(zip(DRIVER_FACTORS, v)) for v in itertools.product(*DRIVER_FACTORS.values())]
    heartbeat = STATE / f"heartbeat_{wid}.json"; manifest = STATE / f"manifest_{wid}.jsonl"
    my = [(idx, c) for idx, c in enumerate(cells) if idx % N_WORKERS == wid]
    done = built = 0
    for n_seen, (idx, cell) in enumerate(my):
        if time.time() - start_epoch > WALLCLOCK_BUDGET_S:
            log(f"budget reached ({(time.time()-start_epoch)/3600:.2f} h) -> stop"); break
        if MAX_CELLS and done >= MAX_CELLS:
            log("MAX_CELLS reached -> stop"); break
        res_path = RESULTS / f"cell_{idx:04d}.jsonl"
        if res_path.exists():
            done += 1; continue
        if built and built % RESTART_EVERY == 0:
            restart()
        t0 = time.time()
        lut_path = LUTS / f"cell_{idx:04d}.npz"
        try:
            if lut_path.exists():
                lut = ForceLUT.load(lut_path); qc = {"cached": True}
            else:
                lut, qc = build_lut(cell, backend, restart, S, np); lut.save(lut_path)
            built += 1
        except Exception as e:
            log(f"cell {idx} LUT FAILED: {e!r}"); restart(); continue

        rows = []
        for driver in driver_combos:
            knobs = od.DesignKnobs(n_coils=N_COILS, **cell, **driver)
            for fl in FORCE_LAWS:
                try:
                    v = S.simulate_exit_speed(knobs, fl, lut if fl == "femm" else None); err = None
                except Exception as e:
                    v, err = None, repr(e)
                rows.append({"cell_id": idx, **cell, **driver, "n_coils": N_COILS,
                             "force_law": fl, "exit_speed_mps": v, "sim_error": err})
        tmp = res_path.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(r) for r in rows)); os.replace(tmp, res_path)
        with open(manifest, "a") as f:
            f.write(json.dumps({"cell_id": idx, **cell, "build_s": round(time.time()-t0, 1), **qc}) + "\n")
        done += 1
        dt = time.time() - t0; elapsed = time.time() - start_epoch
        best = max((r["exit_speed_mps"] or 0) for r in rows if r["force_law"] == "femm")
        heartbeat.write_text(json.dumps({
            "worker": wid, "cells_done_this_worker": done, "last_cell_id": idx,
            "elapsed_h": round(elapsed/3600, 3), "sec_per_cell": round(dt, 1),
            "last_cell_best_femm_mps": round(best, 3), "qc": qc,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
        if built <= 2 or done % 5 == 0:
            log(f"cell {idx} done (#{done}) {dt:.0f}s best_femm={best:.2f} peakF={qc.get('peak_force_n',0):.0f}N "
                f"zOff={qc.get('zero_offset_force_frac',0)} edge={qc.get('edge_force_frac',0)} elapsed={elapsed/3600:.2f}h")
    try: backend.close()
    except Exception: pass
    log(f"worker done: {done} cells")


def main():
    import multiprocessing as mp
    for d in (BASE, LUTS, RESULTS, STATE):
        d.mkdir(parents=True, exist_ok=True)
    start_epoch = float(START_FILE.read_text().strip()) if START_FILE.exists() else time.time()
    if not START_FILE.exists(): START_FILE.write_text(repr(start_epoch))
    cells = all_cells()
    n_total_rows = len(cells) * len(list(itertools.product(*DRIVER_FACTORS.values()))) * len(FORCE_LAWS)
    print(f"[{time.strftime('%H:%M:%S')}] study start: {len(cells)} cells, {N_WORKERS} workers, "
          f"budget {WALLCLOCK_BUDGET_S/3600:.1f} h, up to {n_total_rows} rows.", flush=True)

    if N_WORKERS == 1:
        worker(0, cells, start_epoch)
    else:
        procs = [mp.Process(target=worker, args=(w, cells, start_epoch)) for w in range(N_WORKERS)]
        for p in procs: p.start()
        for p in procs: p.join()

    done = len(list(RESULTS.glob("cell_*.jsonl")))
    DONE_FILE.write_text(json.dumps({"cells_done": done, "elapsed_h": round((time.time()-start_epoch)/3600, 3),
                                     "finished": time.strftime("%Y-%m-%d %H:%M:%S")}))
    print(f"[{time.strftime('%H:%M:%S')}] STUDY COMPLETE: {done} cells, "
          f"{(time.time()-start_epoch)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
