"""The pipeline process: runs the stages of a project in order, idempotent
and resumable.  Launched detached by runner.py (or directly:
`python -m gguf_trainer.pipeline --project DIR`).  Progress goes to
state.json (atomic) and pipeline.log; a STOP file or SIGTERM makes the
current stage save and return, and the next launch continues.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
import traceback

from .packs import get_pack
from .project import STAGES, Project
from .util import now


class Pipeline:
    # stages a user may re-run on demand (idempotent, cheap, rebuilt from best.pt)
    FORCEABLE = ("export", "eval")

    def __init__(self, project: Project, only=None, force: bool = False):
        self.project = project
        self.pack = get_pack(project.config["pack"])
        self.only = set(only or [])
        self.force = force
        self.state = project.read_state()
        self.state.update({"status": "running", "pid": os.getpid(), "host": socket.gethostname(),
                           "started": now(), "error": None, "stage": None})
        self._stop = False
        self._last_flush = 0.0
        self.project.write_state(self.state)

    # -- plumbing --
    def log(self, msg: str):
        print(f"[{time.strftime('%F %T')}] {msg}", flush=True)

    def should_stop(self) -> bool:
        return self._stop or self.project.stop_requested()

    def on_signal(self, sig, _frame):
        self._stop = True
        self.log(f"signal {sig}: finishing the current step and saving")

    def stage_report(self, stage: str):
        def report(d: dict):
            s = self.state["stages"].setdefault(stage, {})
            s.update(d)
            s["updated"] = now()
            # throttle: metrics arrive per batch during precompute
            if now() - self._last_flush > 1.0 or d.get("status"):
                self.project.write_state(self.state)
                self._last_flush = now()
        return report

    def set_stage(self, stage: str, status: str, **extra):
        s = self.state["stages"].setdefault(stage, {})
        s["status"] = status
        s.update(extra)
        if status == "running":
            s["started"] = now()
            self.state["stage"] = stage
        elif status in ("done", "stopped", "failed"):
            s["finished"] = now()
        self.project.write_state(self.state)

    # -- stages --
    def run(self) -> int:
        signal.signal(signal.SIGINT, self.on_signal)
        signal.signal(signal.SIGTERM, self.on_signal)
        self.project.clear_stop()
        art = self.project.stage_artifacts()
        self.log(f"pipeline start: project {self.project.path}, pack {self.pack.id}, pid {os.getpid()}")
        ctx = None
        try:
            for stage in STAGES:
                if self.only and stage not in self.only:
                    continue
                if art[stage]["done"] and not (self.force and stage in self.FORCEABLE):
                    self.set_stage(stage, "done", skipped=True)
                    self.log(f"=== {stage}: already done, skipping")
                    continue
                if stage == "eval" and not self.project.config["export"].get("run_eval", True):
                    self.set_stage(stage, "skipped")
                    continue
                if self.should_stop():
                    return self.finish("stopped")
                self.log(f"=== {stage}" + (" (forced re-run)" if self.force and art[stage]["done"] else ""))
                self.set_stage(stage, "running")
                report = self.stage_report(stage)
                if stage == "corpus":
                    from .corpus import build_corpus, build_image_corpus

                    if self.pack.needs_images:
                        build_image_corpus(self.project, self.pack, self.project.config["corpus"],
                                           self.project.config.get("hf_token", ""), self.log,
                                           progress=lambda s: report({"detail": s}))
                    else:
                        build_corpus(self.project.config["corpus"], self.project.data_dir,
                                     self.project.config.get("hf_token", ""), self.log,
                                     progress=lambda s: report({"detail": s}))
                elif stage in ("precompute_val", "precompute_train"):
                    from .precompute import PrecomputeContext

                    if ctx is None:
                        report({"detail": "loading student + teacher"})
                        ctx = PrecomputeContext(self.project, self.pack, self.log)
                    ok = ctx.run_split("val" if stage == "precompute_val" else "train", report, self.should_stop)
                    if not ok:
                        self.set_stage(stage, "stopped")
                        return self.finish("stopped")
                elif stage == "train":
                    if ctx is not None:      # free the teacher before training
                        del ctx
                        ctx = None
                        _free_cuda()
                    from .train import train

                    res = train(self.project, self.pack, self.log, report, self.should_stop)
                    if res == "stopped":
                        self.set_stage(stage, "stopped")
                        return self.finish("stopped")
                elif stage == "export":
                    from .export import export_adapter

                    out = export_adapter(self.project, self.pack, self.log)
                    extras = self.pack.extra_exports(self.project, self.log)
                    report({"path": str(out), "extras": [str(p) for p in extras]})
                elif stage == "eval":
                    from .evaluate import evaluate

                    evaluate(self.project, self.pack, self.log)
                self.set_stage(stage, "done")
                art = self.project.stage_artifacts()
            return self.finish("done")
        except Exception as e:
            self.log("!!! " + "".join(traceback.format_exception(e)).rstrip())
            if self.state.get("stage"):
                self.set_stage(self.state["stage"], "failed", error=str(e))
            self.state["error"] = str(e)
            return self.finish("failed")

    def finish(self, status: str) -> int:
        self.state["status"] = status
        self.state["finished"] = now()
        self.state["pid"] = None
        self.project.write_state(self.state)
        self.project.clear_stop()
        self.log(f"pipeline {status}")
        return 0 if status in ("done", "stopped") else 1


def _free_cuda():
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gguf_trainer.pipeline")
    ap.add_argument("--project", required=True)
    ap.add_argument("--only", nargs="*", default=None, help=f"run only these stages: {STAGES}")
    ap.add_argument("--force", action="store_true", help="re-run export/eval even if their files are up to date")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    project = Project(args.project)
    if not project.exists():
        print(f"no project at {project.path}", file=sys.stderr)
        return 2
    return Pipeline(project, args.only, force=args.force).run()


if __name__ == "__main__":
    sys.exit(main())
