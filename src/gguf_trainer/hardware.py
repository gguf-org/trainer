"""Hardware snapshot for the Hardware tab: host, GPUs (nvidia-smi + torch),
the environment the pipeline will run in, and the pipeline process itself."""

from __future__ import annotations

import os
import pathlib
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

from .util import free_disk_bytes, no_console_kwargs, pid_alive


def _int(s: str) -> Optional[int]:
    try:
        return int(float(s))
    except ValueError:
        return None


def gpus_from_smi() -> List[Dict[str, Any]]:
    smi = shutil.which("nvidia-smi") or ("/usr/lib/wsl/lib/nvidia-smi" if os.path.exists("/usr/lib/wsl/lib/nvidia-smi") else None)
    if not smi:
        return []
    try:
        out = subprocess.run([smi, "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                             **no_console_kwargs())
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            gpus.append({"index": _int(parts[0]), "name": parts[1], "vram_total_mib": _int(parts[2]),
                         "vram_used_mib": _int(parts[3]), "utilization_percent": _int(parts[4]),
                         "temperature_c": _int(parts[5]),
                         "power_w": _int(parts[6]) if len(parts) > 6 else None})
    return gpus


def env_versions() -> Dict[str, Any]:
    v: Dict[str, Any] = {"python": sys.version.split()[0], "executable": sys.executable}
    for mod in ("torch", "transformers", "gguf-connector", "huggingface_hub", "datasets", "safetensors", "accelerate", "pillow", "psutil"):
        try:
            import importlib.metadata as md

            v[mod] = md.version(mod)
        except Exception:
            v[mod] = None
    try:
        import torch

        v["cuda_available"] = torch.cuda.is_available()
        v["cuda_version"] = torch.version.cuda
        v["torch_devices"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                               "total_mib": torch.cuda.get_device_properties(i).total_memory // 2**20}
                              for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except Exception as e:
        v["cuda_available"] = False
        v["torch_error"] = str(e)
    return v


_env_cache: Optional[Dict[str, Any]] = None


def snapshot(project_path: Optional[str] = None, pid: Optional[int] = None) -> Dict[str, Any]:
    global _env_cache
    info: Dict[str, Any] = {"os": f"{platform.system()} {platform.release()}", "cpu_count": os.cpu_count(),
                            "hostname": platform.node()}
    try:
        import psutil

        vm = psutil.virtual_memory()
        info.update({"total_ram_mib": vm.total // 2**20, "available_ram_mib": vm.available // 2**20,
                     "cpu_percent": psutil.cpu_percent(interval=None)})
        if pid and pid_alive(pid):
            p = psutil.Process(pid)
            with p.oneshot():
                info["process"] = {"pid": pid, "cpu_percent": p.cpu_percent(interval=None),
                                   "rss_mib": p.memory_info().rss // 2**20,
                                   "threads": p.num_threads(), "create_time": p.create_time()}
    except ImportError:
        try:
            fields = {}
            for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
                k, _, rest = line.partition(":")
                fields[k.strip()] = rest.strip()
            info["total_ram_mib"] = int(fields["MemTotal"].split()[0]) // 1024
            info["available_ram_mib"] = int(fields["MemAvailable"].split()[0]) // 1024
        except (OSError, KeyError):
            pass
    info["gpus"] = gpus_from_smi()
    if _env_cache is None:
        _env_cache = env_versions()
    info["env"] = _env_cache
    if project_path:
        info["disk_free"] = free_disk_bytes(project_path)
    return info
