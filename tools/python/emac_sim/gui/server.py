"""Local web server behind the unified EMAC GUI (see gui/__init__.py).

Design: a dependency-free stdlib ``ThreadingHTTPServer`` that serves one single-page app
(``index.html``) plus a small JSON API. It deliberately avoids Flask/FastAPI and websockets
so the GUI adds NO new runtime dependencies -- everything here is stdlib.

Command execution: each "run" launches one of a WHITELIST of emac CLIs as a subprocess
(``python -m emac_sim.<module> ...``) from the repo root, with a reader thread capturing its
combined stdout/stderr line by line into an in-memory buffer. The frontend polls
``/api/job`` for new lines and the exit status. Only whitelisted commands with structured
arguments can be launched -- the browser never gets to run an arbitrary shell string.

The sweep cost/ETA estimate and the LUT visualization reuse the real library code
(``fem.convergence.estimate_sweep_cost``, ``fem.quality.check_lut``, ``fem.lut.ForceLUT``),
so the numbers the GUI shows are the same ones the tools compute, not a re-implementation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[4]     # tools/python/emac_sim/gui/server.py -> repo
HERE = Path(__file__).resolve().parent
CONFIG_DIR = REPO_ROOT / "examples" / "configs"
BUILD_DIR = REPO_ROOT / "build"


# --------------------------------------------------------------------------- command registry
# One source of truth for BOTH the argv the server builds and the form the frontend renders.
def _cmd(module: str, label: str, help_: str, args: list[dict]) -> dict:
    return {"module": module, "label": label, "help": help_, "args": args}


COMMANDS: dict[str, dict] = {
    "sim": _cmd(
        "emac_sim.cli", "Simulate a config",
        "Run the closed-loop simulator on a TOML config and write plots/data.",
        [
            {"name": "config", "flag": "--config", "type": "config", "required": True},
            {"name": "outdir", "flag": "--outdir", "type": "text", "default": "build/gui/sim"},
            {"name": "t_end", "flag": "--t-end", "type": "number", "default": None,
             "help": "override simulated duration (s)"},
            {"name": "no_plots", "flag": "--no-plots", "type": "bool", "default": False},
        ],
    ),
    "optimize": _cmd(
        "emac_sim.optimize_design", "Optimize a design",
        "Search driver/winding/magnet knobs for maximum slug exit speed.",
        [
            {"name": "maxiter", "flag": "--maxiter", "type": "int", "default": 15},
            {"name": "popsize", "flag": "--popsize", "type": "int", "default": 12},
            {"name": "force_law", "flag": "--force-law", "type": "choice",
             "choices": ["analytic", "fem_reference"], "default": "analytic"},
            {"name": "seed", "flag": "--seed", "type": "int", "default": 0},
        ],
    ),
    "femgen": _cmd(
        "emac_sim.fem.cli", "Generate FEM force tables (sweep)",
        "Sweep a coil/slug over offset x current into force LUTs -- the long sweep.",
        [
            {"name": "config", "flag": "--config", "type": "config", "required": True},
            {"name": "outdir", "flag": "--outdir", "type": "text", "default": "build/gui/fem_lut"},
            {"name": "backend", "flag": "--backend", "type": "choice",
             "choices": ["reference", "femm"], "default": "reference"},
            {"name": "n_offsets", "flag": "--n-offsets", "type": "int", "default": 41},
            {"name": "n_currents", "flag": "--n-currents", "type": "int", "default": 11},
        ],
    ),
    "femqc": _cmd(
        "emac_sim.fem.quality", "Quality-check LUTs",
        "Score force LUTs against physical invariants and flag suspect tables.",
        [
            {"name": "inputs", "flag": None, "type": "text", "default": "build/gui/fem_lut",
             "help": "LUT .npz file(s) or a directory"},
            {"name": "reluctance_slug", "flag": "--reluctance-slug", "type": "bool", "default": False},
        ],
    ),
    "femcheck": _cmd(
        "emac_sim.fem.convergence", "Mesh-convergence + cost (FEMM)",
        "Pre-flight a FEM run: is the mesh converged, and how long will the sweep take?",
        [
            {"name": "config", "flag": "--config", "type": "config", "required": True},
            {"name": "coil", "flag": "--coil", "type": "int", "default": 0},
            {"name": "n_geometries", "flag": "--n-geometries", "type": "int", "default": 1},
        ],
    ),
}


def build_argv(cmd: str, args: dict[str, Any]) -> list[str]:
    """Translate a (command, args) request into a concrete ``python -m ...`` argv, honoring
    the command's declared arg specs. Positional-style args (spec ``flag`` is None) are
    appended as bare tokens; bool flags are added only when true; everything else is
    ``--flag value`` when a non-empty value is given."""
    if cmd not in COMMANDS:
        raise ValueError(f"unknown command {cmd!r}")
    spec = COMMANDS[cmd]
    argv = [sys.executable, "-m", spec["module"]]
    by_name = {a["name"]: a for a in spec["args"]}
    for name, meta in by_name.items():
        val = args.get(name, meta.get("default"))
        if meta["type"] == "bool":
            if val:
                argv.append(meta["flag"])
            continue
        if val is None or val == "":
            if meta.get("required"):
                raise ValueError(f"{cmd}: missing required argument {name!r}")
            continue
        if meta["flag"] is None:                       # positional (e.g. femqc inputs)
            argv.extend(str(val).split())
        else:
            argv.extend([meta["flag"], str(val)])
    return argv


# --------------------------------------------------------------------------- job manager
@dataclass
class Job:
    id: int
    argv: list[str]
    label: str
    lines: list[str] = field(default_factory=list)
    status: str = "running"                # running | done | failed | stopped
    returncode: int | None = None
    started: float = field(default_factory=time.time)
    proc: subprocess.Popen | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[int, Job] = {}
        self._next = 1
        self._lock = threading.Lock()

    def run(self, argv: list[str], label: str) -> int:
        with self._lock:
            jid = self._next
            self._next += 1
        job = Job(id=jid, argv=argv, label=label)
        self._jobs[jid] = job
        env_note = f"$ {' '.join(_pretty(a) for a in argv)}"
        job.lines.append(env_note)
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()
        return jid

    def _pump(self, job: Job) -> None:
        try:
            proc = subprocess.Popen(
                job.argv, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                env=_child_env(),
            )
        except Exception as exc:                       # spawn failure (bad interpreter etc.)
            with job.lock:
                job.lines.append(f"failed to start: {exc!r}")
                job.status = "failed"
                job.returncode = -1
            return
        job.proc = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            with job.lock:
                job.lines.append(line.rstrip("\n"))
        proc.wait()
        with job.lock:
            if job.status != "stopped":
                job.status = "done" if proc.returncode == 0 else "failed"
            job.returncode = proc.returncode

    def stop(self, jid: int) -> bool:
        job = self._jobs.get(jid)
        if job is None or job.proc is None:
            return False
        with job.lock:
            job.status = "stopped"
        job.proc.terminate()
        return True

    def snapshot(self, jid: int, since: int) -> dict[str, Any] | None:
        job = self._jobs.get(jid)
        if job is None:
            return None
        with job.lock:
            new = job.lines[since:]
            return {
                "id": job.id, "status": job.status, "returncode": job.returncode,
                "label": job.label, "lines": new, "next": len(job.lines),
                "elapsed_s": round(time.time() - job.started, 1),
            }


def _pretty(token: str) -> str:
    return f'"{token}"' if " " in token else token


def _child_env() -> dict[str, str]:
    import os
    env = dict(os.environ)
    tools_python = str(REPO_ROOT / "tools" / "python")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = tools_python + (os.pathsep + existing if existing else "")
    env["PYTHONUNBUFFERED"] = "1"                      # so we see output line-by-line live
    return env


# --------------------------------------------------------------------------- library-backed API
def _safe_path(raw: str) -> Path:
    """Resolve a client-supplied path and confine it to the repo (no traversal escapes)."""
    p = (REPO_ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if REPO_ROOT not in p.parents and p != REPO_ROOT:
        raise ValueError("path escapes the project directory")
    return p


def list_configs() -> list[str]:
    if not CONFIG_DIR.exists():
        return []
    return [str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in sorted(CONFIG_DIR.glob("*.toml"))]


def femm_available() -> bool:
    try:
        import femm  # noqa: F401
        return True
    except Exception:
        return False


# FEMM's COM automation is single-threaded and not safe for concurrent instances (see the
# femm-pyfemm operational notes): serialize every in-process FEMM use behind one lock.
_FEMM_LOCK = threading.Lock()


def _femm_thread():
    """Context manager: hold the FEMM lock AND initialize COM on this worker thread, which
    ThreadingHTTPServer does not do (FEMM's ActiveX bindings fail with 'CoInitialize has not
    been called' from a request thread otherwise). CoInitialize is best-effort so a machine
    without pywin32 still errors cleanly on the actual FEMM import rather than here."""
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        with _FEMM_LOCK:
            co = False
            try:
                import pythoncom
                pythoncom.CoInitialize()
                co = True
            except Exception:
                pass
            try:
                yield
            finally:
                if co:
                    try:
                        pythoncom.CoUninitialize()
                    except Exception:
                        pass
    return _cm()


def estimate_sweep(config: str, n_offsets: int, n_currents: int, n_geometries: int,
                   backend: str, mesh_frac: float | None) -> dict[str, Any]:
    """Project a sweep's wall-clock for the chosen grid + backend by timing real solves of
    coil 0 of `config`. Reuses fem.convergence.estimate_sweep_cost verbatim."""
    from ..config import LinearSimulationConfig, load_config
    from ..fem.from_config import geometry_from_config
    from ..fem.convergence import estimate_sweep_cost, femm_backend_factory
    from ..fem.reference_backend import AnalyticReferenceBackend

    cfg = load_config(str(_safe_path(config)))
    if not isinstance(cfg, LinearSimulationConfig):
        raise ValueError("estimate needs a linear_stepper config")
    slug, coils = geometry_from_config(cfg)
    coil = coils[0]

    if backend == "femm":
        base = 0.15 * min(coil.radial_thickness_m, slug.magnet_radius_m)
        mesh = base * (mesh_frac if mesh_frac else 1.0)
        note = f"real FEMM solves, mesh {mesh*1e3:.3f} mm"
        with _femm_thread():
            cost = estimate_sweep_cost(coil, slug, mesh, n_offsets, n_currents,
                                       n_geometries=n_geometries, sample_solves=2,
                                       backend_factory=femm_backend_factory)
    else:
        factory: Callable = lambda _m: AnalyticReferenceBackend()   # noqa: E731
        note = "analytic reference backend (fast; no FEMM)"
        cost = estimate_sweep_cost(coil, slug, 0.0, n_offsets, n_currents,
                                   n_geometries=n_geometries, sample_solves=5,
                                   backend_factory=factory)
    return {
        "seconds_per_solve": round(cost.seconds_per_solve, 4),
        "n_solves": cost.n_solves,
        "total_seconds": round(cost.total_seconds, 1),
        "total_minutes": round(cost.total_seconds / 60.0, 2),
        "total_hours": round(cost.total_hours, 3),
        "backend": backend, "note": note,
    }


def _lut_stats(offsets, currents, force) -> dict[str, Any]:
    """Physically meaningful summary of a coupling table -- the numbers a researcher reads
    off a force curve: peak thrust, the thrust constant k_a, where the coupling peaks (x_c),
    how wide the lobe is (half-max full width), and how far the far-field tail has decayed."""
    import numpy as np
    peak = float(np.max(np.abs(force)))
    j_pk = int(np.argmax(np.abs(currents)))
    col = np.abs(force[:, j_pk])
    i_pk = int(np.argmax(col))
    peak_offset = float(offsets[i_pk])
    i_at_peak = float(currents[j_pk]) or 1.0
    # half-max full width around the peak lobe (in the peak-current column)
    half = 0.5 * col[i_pk]
    lo = i_pk
    while lo > 0 and col[lo] >= half:
        lo -= 1
    hi = i_pk
    while hi < len(col) - 1 and col[hi] >= half:
        hi += 1
    width = float(offsets[hi] - offsets[lo])
    edge = float(max(abs(force[0, j_pk]), abs(force[-1, j_pk]))) / peak if peak else 0.0
    return {
        "peak_force_n": peak,
        "force_per_amp_n_a": peak / abs(i_at_peak) if i_at_peak else 0.0,
        "peak_offset_mm": abs(peak_offset) * 1e3,       # x_c is a positive half-width; the
                                                        # coupling is odd so the peak lands on
                                                        # whichever lobe -- report the distance
        "coupling_width_mm": width * 1e3,
        "far_field_frac": edge,
        "offset_span_mm": float(offsets[-1] - offsets[0]) * 1e3,
        "current_span_a": float(currents[-1] - currents[0]),
    }


def _analytic_overlay(metadata, offsets, currents, force):
    """If a LUT's metadata carries its source geometry, sweep the fast analytic model over the
    SAME grid and report the disagreement -- for a FEMM table this is the accuracy check; for
    an analytic table it confirms consistency. Returns (analytic_grid, comparison) or None."""
    import numpy as np
    keys = ("turns", "coil_length_m", "radial_thickness_m", "magnet_radius_m",
            "magnet_length_m", "remanence_t")
    if not all(k in metadata for k in keys):
        return None
    from ..fem.geometry import CoilWindingGeometry, SlugGeometry
    from ..fem.reference_backend import AnalyticReferenceBackend
    coil = CoilWindingGeometry(0.0, int(metadata["turns"]), float(metadata["coil_length_m"]),
                               float(metadata["radial_thickness_m"]),
                               bore_clearance_m=float(metadata.get("bore_clearance_m", 0.0015)))
    slug = SlugGeometry(float(metadata["magnet_radius_m"]), float(metadata["magnet_length_m"]),
                        float(metadata["remanence_t"]))
    ref = AnalyticReferenceBackend()
    a = np.array([[ref.solve(coil, slug, float(o), float(c)).force_n for c in currents]
                  for o in offsets])
    peak = float(np.max(np.abs(force))) or 1.0
    sig = np.abs(force) >= 0.05 * peak
    diff = np.abs(a - force)
    max_rel = float(np.max(diff[sig])) / peak if np.any(sig) else 0.0
    mean_rel = float(np.mean(diff[sig])) / peak if np.any(sig) else 0.0
    return ([[float(v) for v in row] for row in a],
            {"max_rel_error": max_rel, "mean_rel_error": mean_rel,
             "backend": metadata.get("backend", "?")})


def analyze_lut(path: str, reluctance: bool = False, compare: bool = True) -> dict[str, Any]:
    """Full analysis of one saved ForceLUT: the grid, a quality-control verdict, derived
    physical stats, and (when the source geometry is recoverable from metadata) an
    analytic-model overlay with error metrics."""
    import numpy as np
    from ..fem.lut import ForceLUT
    from ..fem.quality import check_lut

    lut = ForceLUT.load(_safe_path(path))
    report = check_lut(lut, expect_linear_current=not reluctance, label=Path(path).name)
    offsets = np.asarray(lut.offsets_m)
    currents = np.asarray(lut.currents_a)
    force = np.asarray(lut.force_n)
    out: dict[str, Any] = {
        "path": path,
        "offsets_m": [float(x) for x in offsets],
        "currents_a": [float(x) for x in currents],
        "force_n": [[float(v) for v in row] for row in force],
        "metadata": dict(lut.metadata),
        "stats": _lut_stats(offsets, currents, force),
        "qc": {
            "ok": report.ok, "peak_force_n": report.peak_force_n,
            "n_failed": len(report.failures()),
            "checks": [{"name": c.name, "passed": c.passed, "applicable": c.applicable,
                        "detail": c.detail, "value": c.value, "tolerance": c.tolerance}
                       for c in report.checks],
        },
    }
    if compare:
        ov = _analytic_overlay(dict(lut.metadata), offsets, currents, force)
        if ov is not None:
            out["analytic_force_n"], out["comparison"] = ov
    return out


# Back-compat alias (older callers / tests used this name for the lighter payload).
lut_to_json = analyze_lut


def list_luts(directory: str) -> list[str]:
    d = _safe_path(directory)
    if not d.exists():
        return []
    return [str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in sorted(d.rglob("*.npz"))]


def qc_directory(directory: str, reluctance: bool = False) -> list[dict[str, Any]]:
    """Batch quality-control every LUT in a directory -- triage a whole sweep at a glance:
    per table, the verdict, peak force, and which checks (if any) failed."""
    from ..fem.lut import ForceLUT
    from ..fem.quality import check_lut

    rows: list[dict[str, Any]] = []
    for rel in list_luts(directory):
        row: dict[str, Any] = {"path": rel, "name": Path(rel).name}
        try:
            lut = ForceLUT.load(_safe_path(rel))
            rep = check_lut(lut, expect_linear_current=not reluctance, label=Path(rel).name)
            row.update(ok=rep.ok, peak_force_n=rep.peak_force_n,
                       failed=[c.name for c in rep.failures()])
        except Exception as exc:                        # a corrupt/unreadable table is itself a finding
            row.update(ok=False, error=str(exc), failed=["load"])
        rows.append(row)
    return rows


def optimizer_latest() -> dict[str, Any] | None:
    p = BUILD_DIR / "optimize_results" / "latest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# --------------------------------------------------------------------------- HTTP handler
def _json_default(o: Any) -> Any:
    """Coerce numpy scalars/arrays (which leak in from ForceLUT/quality) to native JSON
    types -- json.dumps can't serialize numpy.bool_/int64/float64/ndarray on its own."""
    import numpy as np
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


JOBS = JobManager()


class Handler(BaseHTTPRequestHandler):
    server_version = "emac-gui"

    def log_message(self, *args: Any) -> None:      # keep the console clean
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=_json_default).encode("utf-8"), "application/json")

    def _error(self, exc: Exception, code: int = 400) -> None:
        self._json({"error": str(exc)}, code)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # -- GET ---------------------------------------------------------------
    def do_GET(self) -> None:
        url = urlparse(self.path)
        route = url.path
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if route in ("/", "/index.html"):
                self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/state":
                self._json({
                    "commands": COMMANDS, "configs": list_configs(),
                    "femm_available": femm_available(),
                    "cwd": str(REPO_ROOT),
                    "default_lut_dir": "build/gui/fem_lut",
                })
            elif route == "/api/job":
                snap = JOBS.snapshot(int(q["id"]), int(q.get("since", 0)))
                if snap is None:
                    self._error(ValueError("no such job"), 404)
                else:
                    self._json(snap)
            elif route == "/api/luts":
                self._json({"luts": list_luts(q.get("dir", "build/gui/fem_lut"))})
            elif route == "/api/lut":
                self._json(analyze_lut(q["path"], reluctance=q.get("reluctance") == "1",
                                       compare=q.get("compare", "1") != "0"))
            elif route == "/api/qcdir":
                self._json({"rows": qc_directory(q.get("dir", "build/gui/fem_lut"),
                                                 reluctance=q.get("reluctance") == "1")})
            elif route == "/api/optimizer":
                self._json({"latest": optimizer_latest()})
            else:
                self._error(ValueError("not found"), 404)
        except Exception as exc:                        # noqa: BLE001 - report any API error as JSON
            self._error(exc, 400)

    # -- POST --------------------------------------------------------------
    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            body = self._read_json()
            if route == "/api/run":
                cmd = body["cmd"]
                argv = build_argv(cmd, body.get("args", {}))
                jid = JOBS.run(argv, label=COMMANDS[cmd]["label"])
                self._json({"job_id": jid})
            elif route == "/api/stop":
                self._json({"ok": JOBS.stop(int(body["id"]))})
            elif route == "/api/estimate":
                self._json(estimate_sweep(
                    body["config"], int(body.get("n_offsets", 41)),
                    int(body.get("n_currents", 11)), int(body.get("n_geometries", 1)),
                    body.get("backend", "reference"), body.get("mesh_frac"),
                ))
            else:
                self._error(ValueError("not found"), 404)
        except Exception as exc:                        # noqa: BLE001
            self._error(exc, 400)


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"EMAC GUI serving at {url}")
    print(f"  project root: {REPO_ROOT}")
    print(f"  FEMM available: {femm_available()}")
    print("  press Ctrl+C to stop")
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="emac-gui",
        description="Unified EMAC GUI: run the tools, launch/estimate sweeps, and visualize "
                    "results in one local web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    args = parser.parse_args(argv)
    serve(args.host, args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
