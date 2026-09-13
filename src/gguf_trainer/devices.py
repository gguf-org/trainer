"""Device selection shared by the stages."""

from __future__ import annotations

import torch


def pick_device(arg: str = "auto") -> torch.device:
    """'auto' -> the CUDA device with the most VRAM (torch orders devices
    fastest-first; never hardcode an index), else CPU."""
    if arg and arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        best = max(range(torch.cuda.device_count()),
                   key=lambda i: torch.cuda.get_device_properties(i).total_memory)
        return torch.device(f"cuda:{best}")
    return torch.device("cpu")


def device_index(dev: torch.device) -> int:
    if dev.type != "cuda":
        return -1
    return dev.index if dev.index is not None else torch.cuda.current_device()


def auto_mem_budgets(dev: torch.device, gpu_gib: float, cpu_gib: float):
    """Teacher placement budgets as accelerate max_memory strings."""
    if gpu_gib <= 0 and dev.type == "cuda":
        total = torch.cuda.get_device_properties(device_index(dev)).total_memory / 2**30
        gpu_gib = max(1.0, total - 1.5)
    if cpu_gib <= 0:
        try:
            import psutil

            cpu_gib = max(4.0, psutil.virtual_memory().available / 2**30 - 6.0)
        except ImportError:
            cpu_gib = 40.0
    return f"{gpu_gib:.1f}GiB", f"{cpu_gib:.1f}GiB"
