#!/usr/bin/env python3
"""gguf-trainer command line entry point.

    gguf-trainer                       launch the trainer GUI in the browser
    gguf-trainer run --project DIR     run / resume a project's pipeline in this terminal
    gguf-trainer start --project DIR   launch it detached (survives the terminal)
    gguf-trainer stop --project DIR    ask a running pipeline to save and exit
    gguf-trainer status --project DIR  print the project's stage status
    gguf-trainer start --project DIR --only export eval --force
                                       regenerate the GGUF (+ eval) from checkpoints/best.pt
    gguf-trainer snapshot --project DIR [--checkpoint last|best] [--eval]
                                       export the adapter at its CURRENT step as <name>-step<N>-f16.gguf,
                                       while training runs (it saves the step first) or after a stop
    gguf-trainer download --project DIR [--id ID ...]
                                       fetch the materials the project still lacks (foreground, resumable)
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser

from . import __version__


def _cmd_serve(args) -> int:
    from .server import auto_resume, serve

    if args.auto_resume:
        try:
            auto_resume()
        except Exception as e:  # never block the GUI on this
            print(f"auto-resume failed: {e}")
    httpd = serve(host=args.host, port=args.port)
    host, port = httpd.server_address[0], httpd.server_address[1]
    url = f"http://{host}:{port}/"
    print(f"gguf-trainer {__version__} — serving on {url}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down (a running pipeline keeps going; use `gguf-trainer stop` to end it)")
    finally:
        httpd.server_close()
    return 0


def _apply_steps(project, steps) -> None:
    """--steps N: write train.steps into project.json before launching (a
    pipeline already running picks it up too, see `set`)."""
    if steps is None:
        return
    before = project.config["train"]["steps"]
    after = project.set_train_steps(steps)
    print(f"train.steps: {before} -> {after}" if before != after else f"train.steps: {after} (unchanged)")


def _cmd_run(args) -> int:
    from .pipeline import main
    from .project import Project

    p = Project(args.project)
    if not p.exists():
        print(f"no project at {p.path}")
        return 2
    _apply_steps(p, args.steps)
    return main(["--project", args.project] + (["--only", *args.only] if args.only else [])
                + (["--force"] if args.force else []))


def _cmd_start(args) -> int:
    from .project import Project
    from .runner import start

    p = Project(args.project)
    if not p.exists():
        print(f"no project at {p.path}")
        return 2
    _apply_steps(p, args.steps)
    pid = start(p, args.only or None, force=args.force)
    print(f"pipeline launched (pid {pid}), logging to {p.log_file}")
    return 0


def _cmd_stop(args) -> int:
    from .project import Project
    from .runner import stop

    p = Project(args.project)
    gone = stop(p, timeout=args.timeout)
    print("stopped" if gone else f"stop requested (pid {p.pid()} still finishing its step)")
    return 0


def _cmd_snapshot(args) -> int:
    from .project import Project
    from .snapshot import main, start_job

    p = Project(args.project)
    if not p.exists():
        print(f"no project at {p.path}")
        return 2
    if args.detach:
        st = start_job(p, args.checkpoint, do_eval=args.eval, fresh=not args.no_fresh, device=args.device)
        print(f"snapshot job launched (pid {st.get('pid')}), logging to {p.snapshot_log_file}")
        return 0
    return main(["--project", str(p.path), "--checkpoint", args.checkpoint]
                + (["--eval"] if args.eval else []) + (["--no-fresh"] if args.no_fresh else [])
                + (["--device", args.device] if args.device else []) + ["--timeout", str(args.timeout)])


def _cmd_set(args) -> int:
    """Change settings of an existing project (currently: the planned steps).
    Works while the pipeline runs: the train stage re-reads project.json at
    its next log interval and finishes / continues at the new count."""
    from .project import Project

    p = Project(args.project)
    if not p.exists():
        print(f"no project at {p.path}")
        return 2
    if args.steps is None:
        print(f"train.steps: {p.config['train']['steps']}")
        return 0
    _apply_steps(p, args.steps)
    art = p.stage_artifacts()["train"]
    if p.is_running():
        print("pipeline running: the trainer picks the new count up within a log interval")
    elif art["step"] is not None and art["step"] >= art["steps"]:
        print(f"checkpoint at step {art['step']} already covers it: the training stage counts as done")
    elif art["step"] is not None:
        print(f"checkpoint at step {art['step']}: `gguf-trainer start --project {p.path}` continues to {art['steps']}")
    return 0


def _cmd_status(args) -> int:
    from .project import STAGES, Project

    p = Project(args.project)
    if not p.exists():
        print(f"no project at {p.path}")
        return 2
    st = p.read_state()
    art = p.stage_artifacts()
    print(f"{p.path}: {p.runtime_status()} (pid {p.pid()})")
    for s in STAGES:
        info = st.get("stages", {}).get(s, {})
        print(f"  {s:18s} {'done' if art[s]['done'] else info.get('status', '-'):10s} "
              f"{json.dumps({k: v for k, v in info.items() if k in ('step', 'steps', 'done_shards', 'total_shards', 'val_cos', 'best_val_cos', 'error')})}")
    tr = art["train"]
    if tr["has_last"] or tr["has_best"]:
        print(f"  checkpoints: last step {tr['step']}"
              + (f" (val cos {tr['last_val_cos']:.4f})" if tr.get("last_val_cos") is not None else "")
              + (f", best step {tr['best_step']} (val cos {tr['best_val_cos']:.4f})" if tr.get("best_step") is not None else ""))
    snaps = p.snapshots()
    if snaps:
        print("  snapshots:")
        for e in snaps:
            ev = e.get("eval") or {}
            key = next((k for k in ("cos_centered", "cos_slice", "cos") if k in ev), None)
            vc = e.get("val_cos")
            line = f"    step {e['step']:>7}/{e['planned_steps']}  {e['checkpoint']:4s}  "
            line += f"val cos {vc:.4f}" if vc is not None else "val cos   -   "
            if key:
                line += f"  eval {key} {ev[key]:.4f}"
            print(line + f"  {e['path']}")
    job = p.snapshot_job_status()
    if job["status"] == "running":
        print(f"  snapshot job: running — {job.get('phase')}")
    elif job["status"] == "failed":
        print(f"  snapshot job: failed — {job.get('error')}")
    return 0


def _cmd_download(args) -> int:
    from .materials import download_missing_cli

    return download_missing_cli(args.project, args.id or None, include_optional=not args.required_only)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gguf-trainer",
                                     description="Train pig_clip adapters that replace a diffusion model's text encoder in ggk.")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="launch the trainer GUI (default)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8655, help="port to listen on (0 = auto; default 8655)")
    p_serve.add_argument("--no-browser", action="store_true")
    p_serve.add_argument("--auto-resume", action="store_true",
                         help="relaunch the last project if it was interrupted (e.g. by a reboot)")
    p_serve.set_defaults(func=_cmd_serve)

    for name, fn, help_ in (("run", _cmd_run, "run/resume a project's pipeline in the foreground"),
                            ("start", _cmd_start, "launch a project's pipeline detached")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--project", required=True)
        sp.add_argument("--only", nargs="*", default=None, help="restrict to these stages")
        sp.add_argument("--force", action="store_true",
                        help="re-run export/eval even if the GGUF / eval.json are up to date (e.g. after deleting the GGUF)")
        sp.add_argument("--steps", default=None,
                        help="set the planned training steps (train.steps, default 20000; '12k' works) before launching")
        sp.set_defaults(func=fn)
    p_set = sub.add_parser("set", help="change a project's planned training steps (also while it runs)")
    p_set.add_argument("--project", required=True)
    p_set.add_argument("--steps", default=None, help="new train.steps (omit to print the current value)")
    p_set.set_defaults(func=_cmd_set)
    p_stop = sub.add_parser("stop", help="stop a running pipeline (it saves first)")
    p_stop.add_argument("--project", required=True)
    p_stop.add_argument("--timeout", type=float, default=30.0)
    p_stop.set_defaults(func=_cmd_stop)
    p_snap = sub.add_parser("snapshot", help="export the adapter at its current step as a step-tagged GGUF "
                                             "(while training runs, or from the saved checkpoint after a stop)")
    p_snap.add_argument("--project", required=True)
    p_snap.add_argument("--checkpoint", choices=("last", "best"), default="last",
                        help="last = newest step (default), best = best validation score so far")
    p_snap.add_argument("--eval", action="store_true", help="also evaluate it on the val shards (result in snapshots.json)")
    p_snap.add_argument("--no-fresh", action="store_true",
                        help="while training runs: take the checkpoint on disk instead of asking the trainer to save the current step")
    p_snap.add_argument("--device", default=None, help="eval device (default: cpu while the pipeline runs)")
    p_snap.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for the trainer to save")
    p_snap.add_argument("--detach", action="store_true", help="run in the background (like the GUI does)")
    p_snap.set_defaults(func=_cmd_snapshot)
    p_status = sub.add_parser("status", help="print a project's stage status")
    p_status.add_argument("--project", required=True)
    p_status.set_defaults(func=_cmd_status)
    p_dl = sub.add_parser("download", help="download the materials a project still lacks (links copies found nearby)")
    p_dl.add_argument("--project", required=True)
    p_dl.add_argument("--id", nargs="*", default=None, help="restrict to these material ids")
    p_dl.add_argument("--required-only", action="store_true", help="skip optional materials (e.g. the SigVQ vision encoder)")
    p_dl.set_defaults(func=_cmd_download)

    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list or argv_list[0] not in sub.choices and argv_list[0] not in ("-h", "--help", "--version"):
        argv_list.insert(0, "serve")
    args = parser.parse_args(argv_list)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
