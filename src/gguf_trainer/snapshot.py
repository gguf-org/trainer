"""Snapshot export: a GGUF of the adapter as it is RIGHT NOW, without
waiting for the run to reach its last step.

    python -m gguf_trainer.snapshot --project DIR [--checkpoint last|best] [--eval]

Runs as its own (detached) process — the pipeline is single-threaded and
may be busy training, and the GUI server never imports torch.  While the
train stage is running and --no-fresh is not given, the job first drops a
SNAPSHOT request file: the trainer validates and saves last.pt at the
current step (train.py), deletes the file and keeps going; only then is
the checkpoint read.  When the pipeline is stopped / interrupted / done the
checkpoint on disk is exported as is.

The GGUF is step-tagged (<name>-step<N>-f16.gguf next to the final export)
so several points of one run can be kept and compared in the engine;
provenance KVs record the step, the planned steps, which checkpoint it
came from and its validation cosine.  <project>/snapshots.json lists them
with those numbers (+ the optional eval), and .snapshot.status/.pid/
snapshot.log carry the job like a material download does.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional

from .project import Project
from .util import now, pid_alive, read_json, write_json_atomic

JOB_MARKER = "gguf_trainer.snapshot"
CHECKPOINTS = ("last", "best")


def job_status(project: Project) -> Dict[str, Any]:
    """The last snapshot job: status none | running | done | failed (+ phase,
    step, path, error...).  A terminal status file outranks the pid (pids
    get recycled, see materials.DownloadJob)."""
    st = read_json(project.snapshot_status_file, None)
    st = dict(st) if isinstance(st, dict) else {}
    pid = None
    try:
        pid = int(project.snapshot_pid_file.read_text().strip())
    except (OSError, ValueError):
        pass
    if st.get("status") in ("done", "failed"):
        status = st["status"]
    elif pid_alive(pid, marker=JOB_MARKER):
        status = "running"
    elif st.get("status") == "running":
        status = "failed"           # died without a terminal record
        st.setdefault("error", "the snapshot process died (see snapshot.log)")
    else:
        status = "none"
    st.update({"status": status, "pid": pid if status == "running" else None,
               "log": str(project.snapshot_log_file)})
    return st


def job_running(project: Project) -> bool:
    return job_status(project)["status"] == "running"


def can_snapshot(project: Project) -> Optional[str]:
    """None when a snapshot is possible now, else the reason it is not."""
    if job_running(project):
        return "a snapshot export is already running"
    has_ck = any((project.checkpoints_dir / f"{c}.pt").is_file() for c in CHECKPOINTS)
    if has_ck:
        return None
    if project.is_running() and project.read_state().get("stage") == "train":
        return None                 # the trainer will write one on request
    return "no checkpoint yet: the train stage has not saved anything to snapshot"


def start_job(project: Project, checkpoint: str = "last", do_eval: bool = False, fresh: bool = True,
              device: Optional[str] = None) -> Dict[str, Any]:
    from .runner import spawn_detached

    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"checkpoint must be one of {CHECKPOINTS}")
    why = can_snapshot(project)
    if why:
        raise RuntimeError(why)
    try:
        project.snapshot_status_file.unlink()
    except OSError:
        pass
    write_json_atomic(project.snapshot_status_file,
                      {"status": "running", "phase": "starting", "checkpoint": checkpoint, "eval": bool(do_eval),
                       "started": now()})
    cmd = [sys.executable, "-m", "gguf_trainer.snapshot", "--project", str(project.path), "--checkpoint", checkpoint]
    if do_eval:
        cmd.append("--eval")
    if not fresh:
        cmd.append("--no-fresh")
    if device:
        cmd += ["--device", device]
    pid = spawn_detached(cmd, project.snapshot_log_file, project.snapshot_pid_file, project.path)
    st = job_status(project)
    st["pid"] = pid
    return st


def _manifest_add(project: Project, entry: Dict[str, Any]) -> None:
    m = read_json(project.snapshots_file, None)
    items = list(m.get("snapshots", [])) if isinstance(m, dict) else []
    items = [e for e in items if e.get("path") != entry["path"]]      # re-snapshot of the same step replaces
    items.append(entry)
    items.sort(key=lambda e: (e.get("step") or 0, e.get("time") or 0))
    write_json_atomic(project.snapshots_file, {"snapshots": items})


def _finite(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def run_snapshot(project: Project, checkpoint: str = "last", do_eval: bool = False, fresh: bool = True,
                 device: Optional[str] = None, log=print, wait_timeout: float = 1800.0) -> Dict[str, Any]:
    """Export (and optionally evaluate) the checkpoint; returns the manifest entry.
    Raises on failure (the caller records it)."""
    from .export import export_adapter
    from .packs import get_pack

    pack = get_pack(project.config["pack"])
    status: Dict[str, Any] = read_json(project.snapshot_status_file, {}) or {}
    status.update({"status": "running", "checkpoint": checkpoint, "eval": bool(do_eval),
                   "started": status.get("started") or now(), "pid": os.getpid()})

    def phase(p: str, **extra):
        status["phase"] = p
        status.update(extra)
        write_json_atomic(project.snapshot_status_file, status)
        log(f"snapshot: {p}")

    training = project.is_running() and project.read_state().get("stage") == "train"
    if fresh and training:
        project.request_snapshot()
        phase("waiting for the trainer to validate + save the current step")
        t0 = time.time()
        while project.snapshot_requested():
            if not project.is_running():
                project.clear_snapshot()
                log("snapshot: the pipeline ended meanwhile; using the checkpoint on disk")
                break
            if time.time() - t0 > wait_timeout:
                project.clear_snapshot()
                raise RuntimeError(f"the trainer did not save within {wait_timeout:.0f}s (is it stuck in validation?)")
            time.sleep(0.5)
    elif training:
        log("snapshot: --no-fresh: exporting the checkpoint on disk while training continues")

    info = project.checkpoint_info(f"{checkpoint}.pt")
    if not info:
        other = "best" if checkpoint == "last" else "last"
        if project.checkpoint_info(f"{other}.pt"):
            raise RuntimeError(f"checkpoints/{checkpoint}.pt does not exist yet (only {other}.pt); "
                               f"pick --checkpoint {other}")
        raise RuntimeError(f"checkpoints/{checkpoint}.pt does not exist yet")
    step = int(info.get("step") or 0)
    out = project.snapshot_path(step)
    phase(f"exporting checkpoints/{checkpoint}.pt (step {step})", step=step, path=str(out))
    export_adapter(project, pack, log, checkpoint=checkpoint, out=out, snapshot=True, copy=False)
    entry: Dict[str, Any] = {
        "id": f"step{step}-{checkpoint}",
        "step": step,
        "planned_steps": int(info.get("steps") or project.config["train"]["steps"]),
        "checkpoint": checkpoint,
        "val_cos": _finite(info.get("val_cos")),
        "best_val_cos": _finite(info.get("best_val_cos")),
        "path": str(out),
        "name": out.name,
        "size": out.stat().st_size,
        "time": now(),
        "while_training": bool(training),
        "eval": None,
    }
    _manifest_add(project, entry)          # listed even if the eval below fails
    if do_eval:
        from .evaluate import evaluate

        # the training process owns the GPU while it runs: evaluate on the CPU
        # then (a 6 GB card has no room for a second copy of the val set)
        dev = device or ("cpu" if project.is_running() else None)
        phase(f"evaluating {out.name} on the val shards ({dev or 'train device'})")
        res = evaluate(project, pack, log, gguf_path=out, checkpoint=checkpoint, device=dev, write=False)
        entry["eval"] = {k: v for k, v in res.items() if isinstance(v, (int, float)) and k != "time"}
        _manifest_add(project, entry)
    status.update({"status": "done", "phase": "done", "finished": now(), "step": step, "path": str(out),
                   "val_cos": entry["val_cos"], "eval": entry["eval"]})
    write_json_atomic(project.snapshot_status_file, status)
    log(f"snapshot: done -> {out}")
    return entry


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gguf_trainer.snapshot",
                                 description="export the adapter at its current training step as a step-tagged GGUF")
    ap.add_argument("--project", required=True)
    ap.add_argument("--checkpoint", choices=CHECKPOINTS, default="last",
                    help="last = the newest step (default), best = the best validation score so far")
    ap.add_argument("--eval", action="store_true", help="also evaluate the GGUF on the val shards (into snapshots.json)")
    ap.add_argument("--no-fresh", action="store_true",
                    help="while training runs: export the checkpoint already on disk instead of asking the trainer to save the current step")
    ap.add_argument("--device", default=None, help="eval device (default: cpu while the pipeline runs, else the train device)")
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for the trainer to save")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    project = Project(args.project)
    if not project.exists():
        print(f"no project at {project.path}", file=sys.stderr)
        return 2

    def log(msg: str):
        print(f"[{time.strftime('%F %T')}] {msg}", flush=True)

    try:
        entry = run_snapshot(project, args.checkpoint, args.eval, not args.no_fresh, args.device, log, args.timeout)
    except Exception as e:
        log("!!! " + "".join(traceback.format_exception(e)).rstrip())
        st = read_json(project.snapshot_status_file, {}) or {}
        st.update({"status": "failed", "error": str(e), "finished": now()})
        write_json_atomic(project.snapshot_status_file, st)
        return 1
    vc = entry.get("val_cos")
    print(f"{entry['path']}  (step {entry['step']}/{entry['planned_steps']}, {entry['checkpoint']}.pt"
          + (f", val cos {vc:.4f}" if vc is not None else "") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
