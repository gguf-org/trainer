"""Local HTTP backend for the gguf-trainer GUI (stdlib only).

Serves the static GUI and a JSON API.  Everything is addressed by
filesystem path — the browser and the server are on the same machine, so
nothing is uploaded; the project directory is the unit of state and the
client sends its path with every request.
"""

from __future__ import annotations

import json
import mimetypes
import os
import pathlib
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from . import __version__, corpus, hardware, materials, runner, snapshot
from .packs import get_pack, list_packs
from .vision_data import image_preset_list
from .project import STAGES, STAGE_TITLES, Project, default_projects_root, load_settings, save_settings
from .util import read_json

STATIC_DIR = pathlib.Path(__file__).parent / "static"
MODEL_SUFFIXES = (".gguf", ".safetensors")
TEXT_SUFFIXES = (".txt", ".jsonl", ".csv")


def _finite(o: Any) -> Any:
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


class ApiError(Exception):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


def _browse(path: Optional[str], kind: str) -> Dict[str, Any]:
    base = pathlib.Path(path).expanduser() if path else pathlib.Path.home()
    try:
        base = base.resolve()
    except OSError:
        base = pathlib.Path.home()
    if not base.is_dir():
        base = base.parent
    suffixes = {"model": MODEL_SUFFIXES, "text": TEXT_SUFFIXES}.get(kind)
    entries = []
    try:
        children = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except (PermissionError, OSError):
        children = []
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if is_dir:
            entries.append({"name": child.name, "path": str(child), "is_dir": True,
                            "is_project": (child / "project.json").is_file()})
        elif kind == "dir":
            continue
        elif suffixes is None or child.suffix.lower() in suffixes:
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            entries.append({"name": child.name, "path": str(child), "is_dir": False, "size": size})
    parent = str(base.parent) if base.parent != base else None
    return {"path": str(base), "parent": parent, "entries": entries}


def _project(body: Dict[str, Any], must_exist: bool = True) -> Project:
    path = body.get("path") or ""
    if not path:
        raise ApiError("no project path")
    p = Project(path)
    if must_exist and not p.exists():
        raise ApiError(f"no project at {p.path}", 404)
    return p


def _project_payload(p: Project, online: bool = True) -> Dict[str, Any]:
    pack = get_pack(p.config["pack"])
    token = p.config.get("hf_token", "")
    if online:
        # link copies found in the current directory / other projects before
        # judging what is missing (open, create, Refresh — not the 1.5 s poll)
        adopted = materials.adopt_existing(p, pack)
    else:
        adopted = []
    d = p.summary()
    d["pack"] = pack.describe(p)
    d["materials"] = [materials.material_status(p, pack, m, token, online) for m in pack.materials(p)]
    d["adopted"] = adopted
    d["ready"] = all(m["status"] == "ready" for m in d["materials"] if m["required"])
    d["missing_downloads"] = [m["id"] for m in d["materials"]
                              if m["kind"] == "hf_snapshot" and m["wanted"] and m["status"] in ("missing", "partial")]
    d["downloading"] = [m["id"] for m in d["materials"] if m["status"] == "downloading"]
    extras = {}
    for f in d["output_files"]:
        if f["name"].endswith("_sigvq-f16.gguf"):
            extras["sigvq"] = f["path"]
    student = next((m["path"] for m in d["materials"] if m["id"] == "student"), "<pig_clip.gguf>")
    d["engine_command"] = pack.engine_command(pathlib.Path(student).name if student else "<pig_clip.gguf>",
                                              str(p.adapter_path()), {k: v for k, v in extras.items()})
    d["sample_bytes"] = pack.sample_bytes(p.config)
    d["snapshot_blocker"] = snapshot.can_snapshot(p)      # None = the Snapshot button works now
    d["log_size"] = p.log_file.stat().st_size if p.log_file.exists() else 0
    return d


def _read_log(p: Project, after: int, limit: int = 256 * 1024) -> Dict[str, Any]:
    """Tail pipeline.log incrementally: returns the bytes after `after`
    (capped; a first read of a huge log starts from its last `limit` bytes)."""
    if not p.log_file.exists():
        return {"text": "", "next": 0, "size": 0}
    size = p.log_file.stat().st_size
    if after > size:          # log truncated / project switched
        after = 0
    start = after
    if after == 0 and size > limit:
        start = size - limit
    with open(p.log_file, "rb") as f:
        f.seek(start)
        if start != after:
            f.readline()      # drop the partial first line
        data = f.read(limit)
        nxt = f.tell()
    return {"text": data.decode("utf-8", "replace"), "next": nxt, "size": size}


def _metrics(p: Project, max_points: int = 400) -> Dict[str, Any]:
    path = p.checkpoints_dir / "log.csv"
    if not path.exists():
        return {"rows": []}
    import csv

    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({"step": int(row["step"]), "loss": float(row["loss"]), "cos": float(row["cos"]),
                             "rel_mse": float(row["rel_mse"]), "val_cos": float(row["val_cos"]),
                             "lr": float(row["lr"])})
            except (KeyError, ValueError):
                continue
    if len(rows) > max_points:
        step = len(rows) / max_points
        rows = [rows[int(i * step)] for i in range(max_points)] + [rows[-1]]
    return {"rows": rows}


def _reset(p: Project, what: str) -> None:
    import shutil

    if p.is_running():
        raise ApiError("stop the pipeline before resetting")
    if snapshot.job_running(p):
        raise ApiError("a snapshot export is still running")
    # the snapshot manifest describes the checkpoints of THIS run: it goes
    # with the checkpoints (the step-tagged GGUFs themselves stay until
    # "output" removes them together with the other exports)
    snap_files = [p.snapshots_file, p.snapshot_status_file, p.snapshot_pid_file]
    targets = {
        "corpus": [p.data_dir, *snap_files],
        "shards": [p.shards_dir, *snap_files],
        "shards_val": [p.shards_dir / "val"],
        "train": [p.checkpoints_dir, p.eval_path(), *snap_files],
        # the output folder is shared (it holds the project folder itself):
        # remove only this project's exported files, never the directory
        "output": [pathlib.Path(f["path"]) for f in p.output_files()] + snap_files,
        "state": [],
    }
    if what not in targets:
        raise ApiError(f"unknown reset target {what}")
    for t in targets[what]:
        if t.is_dir():
            shutil.rmtree(t)
        elif t.is_file():
            t.unlink()
    if what in ("state", "train", "shards", "corpus"):
        st = p.read_state()
        st["status"] = "idle"
        st["error"] = None
        st["stage"] = None
        if what == "state":
            st["stages"] = {}
        else:
            for k in list(st.get("stages", {})):
                if what == "corpus" or (what == "shards" and k != "corpus") or (what == "train" and k in ("train", "export", "eval")):
                    st["stages"].pop(k, None)
        p.write_state(st)


class Handler(BaseHTTPRequestHandler):
    server_version = f"gguf-trainer/{__version__}"

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj: Any, status: int = 200):
        # NaN/inf are not JSON: the browser's JSON.parse rejects them, and
        # training metrics carry NaN before the first validation
        body = json.dumps(_finite(obj)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400):
        self._json({"error": message}, status)

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        return json.loads(data.decode("utf-8")) if data else {}

    def _serve_static(self, rel: str):
        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())) or not target.is_file():
            self._error("not found", 404)
            return
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        q = {k: v[0] for k, v in urllib.parse.parse_qs(query).items()}
        try:
            if path in ("/", "/index.html"):
                self._serve_static("index.html")
            elif path == "/api/status":
                settings = load_settings()
                self._json({"version": __version__, "home": str(pathlib.Path.home()),
                            "projects_root": str(default_projects_root()),
                            "last_project": settings.get("last_project", ""),
                            "packs": list_packs(), "corpus_presets": corpus.preset_list(),
                            "image_presets": image_preset_list(),
                            "stages": STAGES, "stage_titles": STAGE_TITLES,
                            "windows": os.name == "nt", "mock_teacher": bool(os.environ.get("GGUF_TRAINER_MOCK_TEACHER"))})
            elif path == "/api/hardware":
                pid = None
                if q.get("path"):
                    p = Project(q["path"])
                    pid = p.pid() if p.is_running() else None
                self._json(hardware.snapshot(q.get("path"), pid))
            elif path == "/api/project":
                p = _project(q)
                self._json(_project_payload(p, online=q.get("online", "1") != "0"))
            elif path == "/api/log":
                p = _project(q)
                self._json(_read_log(p, int(q.get("after", 0))))
            elif path == "/api/metrics":
                self._json(_metrics(_project(q)))
            elif path == "/api/download":
                j = materials.job(_project(q).path and q.get("path"), q.get("id") or "")
                self._json(j.to_dict() if j else {"error": "no download for this material"}, 200 if j else 404)
            elif "/.." not in path and path.count("/") == 1:
                self._serve_static(path.lstrip("/"))
            else:
                self._error("not found", 404)
        except ApiError as e:
            self._error(str(e), e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            traceback.print_exc()
            self._error(str(e), 500)

    def do_POST(self):
        path, _, _q = self.path.partition("?")
        try:
            body = self._read_body()
            if path == "/api/browse":
                self._json(_browse(body.get("path"), body.get("kind") or "any"))
            elif path == "/api/project/create":
                pack = get_pack(body.get("pack") or "llada_image")
                name = (body.get("name") or pack.default_name).strip()
                root = body.get("dir") or str(default_projects_root())
                p = Project(pathlib.Path(root) / name)
                if p.exists():
                    raise ApiError(f"project already exists: {p.path}")
                p.create(name=f"pig_{name}" if not name.startswith("pig_") else name, pack=pack.id)
                s = load_settings()
                s["last_project"] = str(p.path)
                save_settings(s)
                self._json(_project_payload(p))
            elif path == "/api/project/open":
                p = _project(body)
                s = load_settings()
                s["last_project"] = str(p.path)
                save_settings(s)
                self._json(_project_payload(p))
            elif path == "/api/project/save":
                p = _project(body)
                p.save_config(body.get("config") or p.config)
                self._json(_project_payload(p, online=False))
            elif path == "/api/materials/download":
                p = _project(body)
                pack = get_pack(p.config["pack"])
                j = materials.start_download(p, pack, body.get("id") or "", p.config.get("hf_token", ""))
                self._json(j.to_dict())
            elif path == "/api/materials/download_all":
                p = _project(body)
                pack = get_pack(p.config["pack"])
                jobs = materials.start_missing(p, pack, p.config.get("hf_token", ""),
                                               include_optional=body.get("include_optional", True))
                d = _project_payload(p, online=True)
                d["started"] = [j.id for j in jobs]
                self._json(d)
            elif path == "/api/pipeline/start":
                p = _project(body)
                only = body.get("only") or None
                env = {}
                if body.get("mock_teacher"):
                    env["GGUF_TRAINER_MOCK_TEACHER"] = "1"
                force = bool(body.get("force"))
                if force and not (p.checkpoints_dir / "best.pt").is_file():
                    raise ApiError("nothing to export: checkpoints/best.pt does not exist (train first)")
                pid = runner.start(p, only, env, force=force)
                self._json({"pid": pid})
            elif path == "/api/pipeline/stop":
                p = _project(body)
                gone = runner.stop(p, timeout=float(body.get("timeout") or 0))
                self._json({"stopped": gone, "pid": p.pid()})
            elif path == "/api/pipeline/kill":
                p = _project(body)
                self._json({"killed": runner.kill(p)})
            elif path == "/api/snapshot":
                # export the adapter at its current step (works while training runs)
                p = _project(body)
                ck = body.get("checkpoint") or "last"
                if ck not in snapshot.CHECKPOINTS:
                    raise ApiError(f"checkpoint must be one of {snapshot.CHECKPOINTS}")
                self._json(snapshot.start_job(p, ck, do_eval=bool(body.get("eval")),
                                              fresh=body.get("fresh", True) is not False,
                                              device=body.get("device") or None))
            elif path == "/api/project/reset":
                p = _project(body)
                _reset(p, body.get("what") or "")
                self._json(_project_payload(p, online=False))
            else:
                self._error("not found", 404)
        except ApiError as e:
            self._error(str(e), e.status)
        except RuntimeError as e:
            self._error(str(e), 400)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (KeyError, ValueError) as e:
            self._error(f"bad request: {e}", 400)
        except Exception as e:
            traceback.print_exc()
            self._error(str(e), 500)


def serve(host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def auto_resume(log=print) -> Optional[str]:
    """Relaunch the last project if it was running when the machine went
    down (state says running, no live process, auto_resume enabled)."""
    s = load_settings()
    last = s.get("last_project")
    if not last:
        return None
    p = Project(last)
    if not p.exists() or not p.config.get("auto_resume", True):
        return None
    if p.runtime_status() == "interrupted":
        pid = runner.start(p)
        log(f"auto-resume: relaunched {p.path} (pid {pid})")
        return str(p.path)
    return None
