#!/usr/bin/env python3
"""gguf-trainer command line entry point.

    gguf-trainer                       launch the trainer GUI in the browser
    gguf-trainer run --project DIR     run / resume a project's pipeline in this terminal
    gguf-trainer start --project DIR   launch it detached (survives the terminal)
    gguf-trainer stop --project DIR    ask a running pipeline to save and exit
    gguf-trainer status --project DIR  print the project's stage status
    gguf-trainer start --project DIR --only export eval --force
                                       regenerate the GGUF (+ eval) from checkpoints/best.pt
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


def _cmd_run(args) -> int:
    from .pipeline import main

    return main(["--project", args.project] + (["--only", *args.only] if args.only else [])
                + (["--force"] if args.force else []))


def _cmd_start(args) -> int:
    from .project import Project
    from .runner import start

    p = Project(args.project)
    if not p.exists():
        print(f"no project at {p.path}")
        return 2
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
        sp.set_defaults(func=fn)
    p_stop = sub.add_parser("stop", help="stop a running pipeline (it saves first)")
    p_stop.add_argument("--project", required=True)
    p_stop.add_argument("--timeout", type=float, default=30.0)
    p_stop.set_defaults(func=_cmd_stop)
    p_status = sub.add_parser("status", help="print a project's stage status")
    p_status.add_argument("--project", required=True)
    p_status.set_defaults(func=_cmd_status)
    p_dl = sub.add_parser("download", help="download the materials a project still lacks (links copies found nearby)")
    p_dl.add_argument("--project", required=True)
    p_dl.add_argument("--id", nargs="*", default=None, help="restrict to these material ids")
    p_dl.add_argument("--required-only", action="store_true", help="skip optional materials (e.g. the SigVQ vision encoder)")
    p_dl.set_defaults(func=_cmd_download)

    argv_list = list(sys.argv[1:] if argv is None else argv)
    if not argv_list or argv_list[0] not in ("serve", "run", "start", "stop", "status", "download", "-h", "--help", "--version"):
        argv_list.insert(0, "serve")
    args = parser.parse_args(argv_list)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
