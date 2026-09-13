"""Launch / stop the detached pipeline process for a project.

The process is started with its own session (setsid on POSIX, a detached
process group on Windows) and its output appended to pipeline.log, so it
survives the GUI server and the terminal closing; only a reboot or a Stop
ends it — and after a reboot the project resumes from its artifacts.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
from typing import List, Optional

from .project import Project
from .util import no_console_kwargs, pid_alive


def spawn_detached(cmd: List[str], log_path, pid_path, cwd, env_extra: Optional[dict] = None) -> int:
    """Start `cmd` in its own session with output appended to log_path and
    its pid recorded in pid_path.  The child imports gguf_trainer from
    wherever THIS copy lives (a source checkout on a relative PYTHONPATH
    would break under a different cwd)."""
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    pkg_root = str(pathlib.Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(env_extra or {})
    pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    log.write(f"\n===== launch {time.strftime('%F %T')}: {' '.join(cmd)}\n".encode())
    log.flush()
    # own session on POSIX; on Windows a hidden console of its own (never
    # DETACHED_PROCESS: that makes the interpreter re-spawned by a venv
    # launcher pop up a visible console window — see util.no_console_kwargs)
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         cwd=str(cwd), env=env, close_fds=True, **no_console_kwargs(detach=True))
    log.close()
    pathlib.Path(pid_path).write_text(str(p.pid))
    return p.pid


def start(project: Project, only: Optional[List[str]] = None, env_extra: Optional[dict] = None,
          force: bool = False) -> int:
    if project.is_running():
        raise RuntimeError(f"pipeline already running (pid {project.pid()})")
    project.clear_stop()
    project.path.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "gguf_trainer.pipeline", "--project", str(project.path)]
    if only:
        cmd += ["--only", *only]
    if force:
        cmd.append("--force")
    return spawn_detached(cmd, project.log_file, project.pid_file, project.path, env_extra)


def stop(project: Project, timeout: float = 0.0) -> bool:
    """Ask the pipeline to save and exit.  Returns True when it is gone."""
    pid = project.pid()
    if not pid_alive(pid):
        return True
    project.request_stop()
    if sys.platform != "win32":
        try:
            os.kill(pid, 15)  # SIGTERM: interrupts a long teacher batch sooner
        except OSError:
            pass
    t0 = time.time()
    while timeout > 0 and time.time() - t0 < timeout:
        if not pid_alive(pid):
            return True
        time.sleep(0.5)
    return not pid_alive(pid)


def kill(project: Project) -> bool:
    pid = project.pid()
    if not pid_alive(pid):
        return True
    try:
        import psutil

        psutil.Process(pid).kill()
    except Exception:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, **no_console_kwargs())
        else:
            try:
                os.kill(pid, 9)
            except OSError:
                pass
    time.sleep(0.5)
    if not pid_alive(pid):
        st = project.read_state()
        if st.get("status") == "running":
            st["status"] = "stopped"
            st["pid"] = None
            project.write_state(st)
        return True
    return False
