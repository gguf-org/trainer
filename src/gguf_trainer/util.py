"""Small shared helpers: atomic JSON, time formatting, process liveness."""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from typing import Any, Optional


def replace_atomic(tmp: os.PathLike | str, path: os.PathLike | str, timeout: float = 15.0) -> None:
    """os.replace that survives Windows sharing violations.

    On Windows a rename over a file fails with PermissionError (WinError 5 /
    32) for as long as *any* process has it open — and the GUI server opens
    state.json / project.json every poll, an antivirus or indexer opens fresh
    checkpoints and GGUFs.  Those windows are milliseconds, so retry with a
    short backoff instead of failing the stage (the export stage died on its
    very first state write this way).  POSIX renames never contend."""
    tmp, path = os.fspath(tmp), os.fspath(path)
    if sys.platform != "win32":
        os.replace(tmp, path)
        return
    delay = 0.05
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


def write_json_atomic(path: os.PathLike | str, obj: Any) -> None:
    """Write JSON so a reboot mid-write never leaves a torn file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    replace_atomic(tmp, path)


def read_json(path: os.PathLike | str, default: Any = None) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def now() -> float:
    return time.time()


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def fmt_bytes(n: Optional[float]) -> str:
    if n is None:
        return "—"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TB"


def pid_alive(pid: Optional[int], marker: Optional[str] = None) -> bool:
    """Best-effort liveness check, cross-platform.

    `marker` is a substring the process's command line must contain (e.g.
    "gguf_trainer.materials").  Pids are recycled — on Windows within
    minutes — so a pid file left by a finished job can name an unrelated
    process (a finished SigVQ download showed as "downloading" forever
    because its pid had become svchost.exe).  With psutil the command line
    is checked; a pid that is alive but not ours counts as dead."""
    if not pid or pid <= 0:
        return False
    try:
        import psutil  # type: ignore

        try:
            p = psutil.Process(pid)
            if not (p.is_running() and p.status() != psutil.STATUS_ZOMBIE):
                return False
            if marker:
                try:
                    cmd = " ".join(p.cmdline())
                except (psutil.AccessDenied, psutil.ZombieProcess):
                    return True  # cannot inspect: assume it is ours
                if cmd and marker not in cmd:
                    return False
            return True
        except psutil.NoSuchProcess:
            return False
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def dir_size(path: os.PathLike | str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def free_disk_bytes(path: os.PathLike | str) -> Optional[int]:
    p = pathlib.Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        import shutil

        return shutil.disk_usage(str(p)).free
    except OSError:
        return None


def no_console_kwargs(detach: bool = False) -> dict:
    """subprocess kwargs that keep Windows from opening a console window.

    A child started with DETACHED_PROCESS has *no* console, and Windows then
    ignores CREATE_NO_WINDOW — so when a venv `python.exe` redirector or the
    `py` launcher re-spawns the real interpreter, that grandchild gets a
    fresh, visible console window that lives as long as the download or the
    pipeline.  CREATE_NO_WINDOW alone gives the child its own *hidden*
    console, which every grandchild (interpreter, nvidia-smi, taskkill)
    inherits; CREATE_NEW_PROCESS_GROUP keeps it out of the server's Ctrl-C
    group.  Elsewhere the caller uses start_new_session instead."""
    import subprocess
    import sys

    if sys.platform != "win32":
        return {"start_new_session": True} if detach else {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    if detach:
        flags |= subprocess.CREATE_NEW_PROCESS_GROUP
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return {"creationflags": flags, "startupinfo": si}
