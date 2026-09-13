"""Materials: the files a pack needs, fetched into <project>/materials.

HF snapshots go through huggingface_hub.snapshot_download with
local_dir=<materials>/<subdir> and allow_patterns, which is resumable on its
own (partial blobs live under <local_dir>/.cache/huggingface).  Progress is
measured as bytes on disk against the sizes the Hub reports, so a download
that was cut by a reboot shows its real remaining size, and "Download" again
simply continues it.

Materials that already exist somewhere else — the current directory, another
project's materials folder, a directory named by GGUF_TRAINER_MATERIALS — are
detected and linked into the project instead of being downloaded again
(`adopt_existing`); `start_missing` launches one download per material that
is still absent, so a fresh machine needs a single click (or
`gguf-trainer download --project DIR`).
"""

from __future__ import annotations

import fnmatch
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional

from .packs.base import Material, TrainerPack
from .util import now, pid_alive, read_json

_size_cache: Dict[str, Dict[str, int]] = {}


def repo_file_sizes(repo: str, token: str = "", repo_type: str = "model") -> Dict[str, int]:
    """{path: bytes} for every file in an HF model or dataset repo (cached per process)."""
    key = f"{repo_type}:{repo}"
    if key in _size_cache:
        return _size_cache[key]
    from huggingface_hub import HfApi

    api = HfApi(token=token or None)
    info = api.dataset_info(repo, files_metadata=True) if repo_type == "dataset" \
        else api.model_info(repo, files_metadata=True)
    sizes = {s.rfilename: int(s.size or 0) for s in info.siblings}
    _size_cache[key] = sizes
    return sizes


def _listed(mat: Material) -> bool:
    return f"{mat.repo_type}:{mat.repo}" in _size_cache


def _match(path: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def expected_files(mat: Material, token: str = "") -> Dict[str, int]:
    try:
        sizes = repo_file_sizes(mat.repo, token, mat.repo_type)
    except Exception:
        return {}
    return {p: n for p, n in sizes.items() if _match(p, mat.patterns)}


def _incomplete_bytes(root: pathlib.Path, patterns: List[str]) -> int:
    """Bytes of in-flight blobs for the directories a material's patterns
    name (hf_hub keeps them as <local_dir>/.cache/huggingface/download/
    <dir>/<hash>.<etag>.<rand>.incomplete, not attributable per file)."""
    total = 0
    seen = set()
    for pat in patterns:
        sub = pat.rsplit("/", 1)[0] if "/" in pat else ""
        if sub in seen:
            continue
        seen.add(sub)
        d = root / ".cache" / "huggingface" / "download" / sub
        try:
            for f in d.glob("*.incomplete"):
                total += f.stat().st_size
        except OSError:
            pass
    return total


def material_status(project, pack: TrainerPack, mat: Material, token: str = "", online: bool = True) -> Dict[str, Any]:
    d: Dict[str, Any] = mat.to_dict()
    path = pack.material_path(project, mat)
    d["path"] = str(path) if path else ""
    d["wanted"] = wanted(project, mat)
    try:
        d["linked"] = bool(path) and project.materials_dir not in pathlib.Path(path).resolve().parents \
            and pathlib.Path(path).resolve() != project.materials_dir
    except OSError:
        d["linked"] = False
    if mat.kind == "local_file":
        ok = bool(path) and pathlib.Path(path).is_file()
        d.update({"status": "ready" if ok else "missing",
                  "bytes_done": pathlib.Path(path).stat().st_size if ok else 0, "bytes_total": None})
        return d
    job = DownloadJob(project, pack, mat)
    if job.pid() is None:
        job = None
    # sizes come from the Hub API; once fetched they are cached per process,
    # so polling with online=False still gets real numbers
    files = expected_files(mat, token) if (online or _listed(mat)) else {}
    root = pathlib.Path(path) if path else None
    done = 0
    present = 0
    missing = []
    for rel, size in files.items():
        f = root / rel if root else None
        if f is not None and f.is_file() and (size == 0 or f.stat().st_size == size):
            done += size
            present += 1
        else:
            missing.append(rel)
    total = sum(files.values()) if files else None
    if root and missing:
        done = min(done + _incomplete_bytes(root, mat.patterns), total or 0) if total else done
    if files:
        status = "ready" if not missing else ("partial" if present or done else "missing")
    else:
        # offline and never listed: judge by the files on disk
        have = bool(root) and root.is_dir() and all(
            any(root.glob(pat)) for pat in mat.patterns)
        incomplete = bool(root) and any((root / ".cache" / "huggingface").rglob("*.incomplete")) \
            if root and (root / ".cache" / "huggingface").is_dir() else False
        status = "ready" if have and not incomplete else ("partial" if root and root.is_dir() else "missing")
    if job and job.status() == "running":
        status = "downloading"
    d.update({"status": status, "bytes_done": done, "bytes_total": total, "n_files": len(files),
              "n_present": present, "missing": missing[:8]})
    if job:
        d["job"] = job.to_dict()
    return d


def offline_complete(root: Optional[pathlib.Path], patterns: List[str]) -> bool:
    """True when every pattern matches at least one file under root and no
    hf_hub blob for those directories is still in flight (no Hub access)."""
    if not root or not root.is_dir():
        return False
    if not all(any(f.is_file() for f in root.glob(pat)) for pat in patterns):
        return False
    return _incomplete_bytes(root, patterns) == 0


def _candidate_roots(project) -> List[pathlib.Path]:
    """Where a pre-downloaded HF snapshot may live, most specific first."""
    from .project import default_projects_root

    roots: List[pathlib.Path] = []
    env = os.environ.get("GGUF_TRAINER_MATERIALS")
    if env:
        roots.append(pathlib.Path(env).expanduser())
    cwd = pathlib.Path.cwd()
    roots += [cwd, cwd / "materials", project.path, project.path.parent]
    try:
        for other in sorted(default_projects_root().iterdir()):
            if other != project.path and (other / "materials").is_dir():
                roots.append(other / "materials")
    except OSError:
        pass
    out, seen = [], set()
    for r in roots:
        try:
            key = str(r.resolve())
        except OSError:
            continue
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def find_existing(project, pack: TrainerPack, mat: Material) -> Optional[pathlib.Path]:
    """A complete copy of `mat` outside its project location, or None."""
    if mat.kind == "hf_snapshot":
        sub = mat.subdir or mat.repo.split("/")[-1]
        for root in _candidate_roots(project):
            for cand in (root / sub, root):
                if offline_complete(cand, mat.patterns):
                    return cand
        return None
    if mat.kind == "local_file":
        # a pig_clip GGUF next to the project / in the current directory;
        # prefer f16 (what the recipe trains against), then the biggest file
        hits: List[pathlib.Path] = []
        for root in _candidate_roots(project) + [project.materials_dir]:
            try:
                hits += [f for f in root.glob("pig_clip*") if f.is_file() and f.suffix in (mat.suffixes or [".gguf"])]
            except OSError:
                pass
        if not hits:
            return None
        hits.sort(key=lambda f: (0 if "f16" in f.name.lower() else 1, -f.stat().st_size))
        return hits[0]
    return None


def adopt_existing(project, pack: TrainerPack) -> List[str]:
    """Link materials that are absent from the project but complete elsewhere
    (writes materials.<id>.path into project.json).  Returns the ids linked."""
    adopted: List[str] = []
    overrides = dict(project.config.get("materials") or {})
    for mat in pack.materials(project):
        path = pack.material_path(project, mat)
        if mat.kind == "hf_snapshot":
            if offline_complete(path, mat.patterns):
                continue
            # never move away from a directory a download is writing into
            if DownloadJob(project, pack, mat).running():
                continue
        elif mat.kind == "local_file":
            if path and pathlib.Path(path).is_file():
                continue
        else:
            continue
        found = find_existing(project, pack, mat)
        if found is None:
            continue
        overrides[mat.id] = {"path": str(found)}
        adopted.append(mat.id)
    if adopted:
        cfg = dict(project.config)
        cfg["materials"] = overrides
        project.save_config(cfg)
    return adopted


def wanted(project, mat: Material) -> bool:
    """Whether a run with this project's settings uses the material."""
    if mat.required:
        return True
    if mat.id == "sigvq":
        return bool((project.config.get("export") or {}).get("export_sigvq", True))
    return True


class DownloadJob:
    """A detached `python -m gguf_trainer.materials` process.  Its pid, log
    and final status live next to the material under <project>/materials,
    so it survives the GUI closing and is found again after a restart."""

    def __init__(self, project, pack: TrainerPack, mat: Material):
        self.project_path = str(project.path)
        self.mat = mat
        self.dest = pack.material_path(project, mat)
        self.token = project.config.get("hf_token", "")
        base = project.materials_dir / f".download-{mat.id}"
        self.pid_file = base.with_suffix(".pid")
        self.log_file = base.with_suffix(".log")
        self.status_file = base.with_suffix(".status")

    @property
    def id(self) -> str:
        return self.mat.id

    def pid(self) -> Optional[int]:
        try:
            return int(self.pid_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def running(self) -> bool:
        """The child writes a terminal status as its last act (and start()
        removes that file first), so a done/failed record outranks a pid
        that merely looks alive — pids get recycled."""
        st = read_json(self.status_file, None)
        if isinstance(st, dict) and st.get("status") in ("done", "failed"):
            return False
        return pid_alive(self.pid(), marker="gguf_trainer.materials")

    def start(self) -> int:
        from .runner import spawn_detached

        if self.running():
            return self.pid()
        self.dest.mkdir(parents=True, exist_ok=True)
        try:
            self.status_file.unlink()
        except OSError:
            pass
        cmd = [sys.executable, "-m", "gguf_trainer.materials", "--project", self.project_path, "--id", self.mat.id]
        env = {"HF_TOKEN": self.token} if self.token else {}
        return spawn_detached(cmd, self.log_file, self.pid_file, self.dest, env)

    def status(self) -> str:
        if self.running():
            return "running"
        st = read_json(self.status_file, None)
        if isinstance(st, dict) and st.get("status") in ("done", "failed"):
            return st["status"]
        return "none" if self.pid() is None else "failed"

    def to_dict(self) -> Dict[str, Any]:
        st = read_json(self.status_file, {}) or {}
        started = None
        try:
            started = self.pid_file.stat().st_mtime
        except OSError:
            pass
        done = 0
        if self.dest.exists():
            files = expected_files(self.mat, self.token)
            for rel in files:
                f = self.dest / rel
                if f.is_file():
                    done += f.stat().st_size
            if any(not (self.dest / rel).is_file() for rel in files):
                done += _incomplete_bytes(self.dest, self.mat.patterns)
        total = sum(expected_files(self.mat, self.token).values()) or None
        if total:
            done = min(done, total)
        elapsed = (st.get("finished") or now()) - started if started else 0
        return {"id": self.id, "material": self.mat.id, "status": self.status(), "error": st.get("error"),
                "bytes_done": done, "bytes_total": total, "elapsed": elapsed,
                "rate": (done - st.get("bytes_at_start", 0)) / elapsed if elapsed > 2 else None,
                "pid": self.pid(), "log": str(self.log_file)}


def _job_for(mat_id: str, project_path: str) -> Optional[DownloadJob]:
    from .packs import get_pack
    from .project import Project

    project = Project(project_path)
    pack = get_pack(project.config["pack"])
    mat = pack.material(mat_id, project)
    if mat is None or mat.kind != "hf_snapshot":
        return None
    job = DownloadJob(project, pack, mat)
    if job.pid() is None:
        return None
    return job


def require_hub() -> None:
    """Fail early, with the fix, when the downloader dependency is absent
    (the detached child would otherwise die with a traceback in its log)."""
    import importlib.util

    if importlib.util.find_spec("huggingface_hub") is None:
        raise RuntimeError("downloading needs the huggingface_hub package: run "
                           f"`{sys.executable} -m pip install huggingface_hub` and press Refresh")


def start_download(project, pack: TrainerPack, mat_id: str, token: str = "") -> DownloadJob:
    mat = pack.material(mat_id, project)
    if mat is None or mat.kind != "hf_snapshot":
        raise ValueError(f"'{mat_id}' is not a downloadable material")
    require_hub()
    job = DownloadJob(project, pack, mat)
    job.start()
    return job


def start_missing(project, pack: TrainerPack, token: str = "", include_optional: bool = True) -> List[DownloadJob]:
    """One detached download per HF material that is not complete on disk
    (and not already downloading).  Anything found elsewhere is linked first."""
    adopt_existing(project, pack)
    jobs: List[DownloadJob] = []
    for mat in pack.materials(project):
        if mat.kind != "hf_snapshot":
            continue
        if not mat.required and not (include_optional and wanted(project, mat)):
            continue
        st = material_status(project, pack, mat, token, online=True)
        if st["status"] in ("ready", "downloading"):
            continue
        require_hub()
        j = DownloadJob(project, pack, mat)
        j.start()
        jobs.append(j)
    return jobs


def job(project_path: str, mat_id: str) -> Optional[DownloadJob]:
    return _job_for(mat_id, project_path)


def download_main(argv=None) -> int:
    """Child process: snapshot_download of one material, status to a file."""
    import argparse
    import json
    import time
    import traceback as tb

    from .packs import get_pack
    from .project import Project

    ap = argparse.ArgumentParser(prog="python -m gguf_trainer.materials")
    ap.add_argument("--project", required=True)
    ap.add_argument("--id", required=True)
    args = ap.parse_args(argv)
    project = Project(args.project)
    pack = get_pack(project.config["pack"])
    mat = pack.material(args.id, project)
    if mat is None:
        print(f"no material '{args.id}' in pack {pack.id}", file=sys.stderr)
        return 2
    job = DownloadJob(project, pack, mat)
    token = os.environ.get("HF_TOKEN") or None
    from .util import write_json_atomic

    st = {"status": "running", "started": time.time(), "bytes_at_start": job.to_dict()["bytes_done"]}
    write_json_atomic(job.status_file, st)
    try:
        from huggingface_hub import snapshot_download

        print(f"[{time.strftime('%F %T')}] downloading {mat.repo} {mat.patterns} -> {job.dest}", flush=True)
        snapshot_download(mat.repo, repo_type=mat.repo_type, allow_patterns=mat.patterns, local_dir=str(job.dest),
                          token=token, max_workers=4)
        st["status"] = "done"
        print(f"[{time.strftime('%F %T')}] done", flush=True)
    except Exception as e:
        st["status"] = "failed"
        st["error"] = str(e)
        print(tb.format_exc(), flush=True)
    st["finished"] = time.time()
    write_json_atomic(job.status_file, st)
    return 0 if st["status"] == "done" else 1


def download_missing_cli(project_path: str, ids: Optional[List[str]] = None, include_optional: bool = True,
                         log=print) -> int:
    """Foreground `gguf-trainer download`: link what exists, fetch the rest
    in this process (resumable; Ctrl-C and rerun to continue)."""
    from huggingface_hub import snapshot_download

    from .packs import get_pack
    from .project import Project

    project = Project(project_path)
    if not project.exists():
        log(f"no project at {project.path}")
        return 2
    pack = get_pack(project.config["pack"])
    token = project.config.get("hf_token", "") or os.environ.get("HF_TOKEN") or None
    for mid in adopt_existing(project, pack):
        mat = pack.material(mid, project)
        log(f"[{mid}] found {pack.material_path(project, mat)} — linked, not downloading")
    failed = 0
    for mat in pack.materials(project):
        if ids and mat.id not in ids:
            continue
        st = material_status(project, pack, mat, token or "", online=True)
        if mat.kind != "hf_snapshot":
            log(f"[{mat.id}] {st['status']}: {st['path'] or 'select the file in the GUI (Materials > Browse)'}")
            continue
        if not mat.required and not (include_optional and wanted(project, mat)) and not (ids and mat.id in ids):
            log(f"[{mat.id}] optional, skipped")
            continue
        if st["status"] == "ready":
            log(f"[{mat.id}] present at {st['path']}")
            continue
        if st["status"] == "downloading":
            log(f"[{mat.id}] a detached download is already running (pid {st['job']['pid']})")
            continue
        job = DownloadJob(project, pack, mat)
        job.dest.mkdir(parents=True, exist_ok=True)
        log(f"[{mat.id}] downloading {mat.repo} {mat.patterns} -> {job.dest}"
            + (f" ({st['bytes_done'] / 2**30:.1f} of {st['bytes_total'] / 2**30:.1f} GB present)" if st.get("bytes_total") else ""))
        try:
            snapshot_download(mat.repo, repo_type=mat.repo_type, allow_patterns=mat.patterns,
                              local_dir=str(job.dest), token=token, max_workers=4)
            log(f"[{mat.id}] done")
        except KeyboardInterrupt:
            log(f"[{mat.id}] interrupted — rerun to continue")
            return 130
        except Exception as e:
            log(f"[{mat.id}] FAILED: {e}")
            failed += 1
    return 1 if failed else 0


def datasets_cache_dir() -> str:
    return os.environ.get("HF_HOME") or str(pathlib.Path.home() / ".cache" / "huggingface")


if __name__ == "__main__":
    sys.exit(download_main())
