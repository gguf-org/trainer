"""Export stage: checkpoint -> f16 GGUF for ggk's --llm-adapter.

Everything the engine needs is derived from tensor shapes (`adapter.query`
[width, num_queries] selects the resampler variant, its absence the
token-aligned one; `adapter.t5_embed.weight` the seeded resampler (query
seed = the teacher tokenizer's ids); `adapter.vision_proj.weight` selects
the vision extension; head_dim 64).  Norms, biases, the query and vision_proj stay
f32, other 2-D weights go f16.  A resampler trained on standardized targets
has the standardization folded into out_proj so the engine emits
teacher-scale rows:  W' = diag(sigma) W,  b' = sigma * b + mu.

The same writer serves the end-of-run export (best.pt -> <name>-f16.gguf)
and the mid-run snapshots (best.pt or last.pt -> <name>-step<N>-f16.gguf,
see snapshot.py).  Provenance KVs (adapter.trained_steps, planned_steps,
checkpoint, val_cos, best_val_cos, snapshot) tell any reader how far along
the run the file was taken; the engine ignores them.
"""

from __future__ import annotations

import math
import pathlib
import shutil
from typing import Optional

import numpy as np
import torch
from gguf_connector.reader import GGUFReader
from gguf_connector.writer import GGUFWriter

from .adapter import KIND_RESAMPLER, KIND_SEEDED, KIND_TOKEN_VISION, AdapterConfig, build_adapter
from .util import replace_atomic


def load_folded_model(checkpoint: pathlib.Path):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = AdapterConfig.from_ck(ck["config"])
    model = build_adapter(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    if ck.get("standardized"):
        mu = ck["mu"].float()
        sigma = ck["sigma"].float()
        with torch.no_grad():
            model.out_proj.weight.mul_(sigma[:, None])
            model.out_proj.bias.mul_(sigma).add_(mu)
    return model, cfg, ck


def _keep_f32(name: str, a: np.ndarray) -> bool:
    return a.ndim == 1 or name in ("query", "vision_proj.weight") or ".ln_" in name or name.startswith("ln_")


def checkpoint_file(project, checkpoint: str) -> pathlib.Path:
    """'best' | 'last' -> checkpoints/<kind>.pt (must exist)."""
    if checkpoint not in ("best", "last"):
        raise ValueError(f"checkpoint must be 'best' or 'last', not {checkpoint!r}")
    ckpt = project.checkpoints_dir / f"{checkpoint}.pt"
    if not ckpt.exists():
        raise RuntimeError(f"no checkpoints/{checkpoint}.pt to export")
    return ckpt


def export_adapter(project, pack, log, checkpoint: str = "best", out: Optional[pathlib.Path] = None,
                   snapshot: bool = False, copy: bool = True) -> pathlib.Path:
    """Write the adapter GGUF from checkpoints/<checkpoint>.pt.  Default = the
    pipeline's export stage (best.pt -> project.adapter_path(), copied to
    export.copy_to).  Snapshots pass out=project.snapshot_path(step),
    snapshot=True and skip the copy."""
    ckpt = checkpoint_file(project, checkpoint)
    model, cfg, ck = load_folded_model(ckpt)
    log(f"export: loaded {ckpt} (step {ck['step']}, {model.num_params() / 1e6:.1f}M params, kind {cfg.kind})")
    out = pathlib.Path(out) if out is not None else project.adapter_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    planned = int((ck.get("train_config") or {}).get("steps") or project.config["train"]["steps"])
    val = ck.get("val") or {}
    val_cos = val.get("val_cos")

    w = GGUFWriter(str(tmp), "pig")
    w.add_name(project.config["name"])
    w.add_uint32("adapter.in_dim", cfg.in_dim)
    w.add_uint32("adapter.out_dim", cfg.out_dim)
    if cfg.kind == KIND_TOKEN_VISION:
        w.add_uint32("adapter.vis_dim", cfg.vis_dim)
    w.add_uint32("adapter.width", cfg.width)
    w.add_uint32("adapter.depth", cfg.depth)
    w.add_uint32("adapter.heads", cfg.heads)
    if cfg.kind in (KIND_RESAMPLER, KIND_SEEDED):
        w.add_uint32("adapter.num_queries", cfg.num_queries)
    w.add_uint32("adapter.trained_steps", int(ck["step"]))
    w.add_uint32("adapter.planned_steps", planned)
    w.add_string("adapter.checkpoint", checkpoint)
    w.add_bool("adapter.snapshot", bool(snapshot))
    if isinstance(val_cos, float) and math.isfinite(val_cos):
        w.add_float32("adapter.val_cos", float(val_cos))
    bvc = ck.get("best_val_cos")
    if isinstance(bvc, float) and math.isfinite(bvc) and bvc > -1.0:
        w.add_float32("adapter.best_val_cos", float(bvc))
    w.add_string("adapter.pack", pack.id)
    w.add_string("adapter.trainer", "gguf-trainer")
    for k, v in pack.export_kv(project).items():
        if isinstance(v, str):
            w.add_string(k, v)
        elif isinstance(v, bool):
            w.add_bool(k, v)
        elif isinstance(v, int):
            w.add_uint32(k, v)
        elif isinstance(v, float):
            w.add_float32(k, v)
    n_f16 = n_f32 = 0
    for name, t in model.state_dict().items():
        a = t.detach().cpu().float().numpy()
        if _keep_f32(name, a):
            w.add_tensor("adapter." + name, a.astype(np.float32))
            n_f32 += 1
        else:
            w.add_tensor("adapter." + name, a.astype(np.float16))
            n_f16 += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    r = GGUFReader(str(tmp))
    names = {t.name for t in r.tensors}
    assert len(r.tensors) == n_f16 + n_f32, "read-back tensor count mismatch"
    if cfg.kind == KIND_TOKEN_VISION:
        assert "adapter.query" not in names and {"adapter.skip.weight", "adapter.vision_proj.weight",
                                                 "adapter.vis_in.weight"} <= names, "vision-ext layout"
    elif cfg.kind == KIND_SEEDED:
        assert {"adapter.query", "adapter.t5_embed.weight"} <= names, "seeded resampler layout"
    else:
        assert "adapter.query" in names and "adapter.t5_embed.weight" not in names, "resampler layout"
    del r
    replace_atomic(tmp, out)
    log(f"export: wrote {out}: {n_f16} f16 + {n_f32} f32 tensors "
        f"({checkpoint}.pt, step {ck['step']}/{planned}"
        + (f", val cos {val_cos:.4f}" if isinstance(val_cos, float) and math.isfinite(val_cos) else "") + ")")
    copy_to = (project.config.get("export") or {}).get("copy_to") or ""
    if copy_to and copy:
        dst = pathlib.Path(copy_to).expanduser()
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, dst / out.name)
        log(f"export: copied to {dst / out.name}")
    return out


def gguf_adapter_model(path: pathlib.Path):
    """Rebuild the adapter from an exported GGUF (the engine's view of it):
    the kind is read off the tensor names exactly like llm_adapter.hpp does."""
    r = GGUFReader(str(path))
    kv = {f.name: f.contents() for f in r.fields.values() if f.name.startswith("adapter.")}
    sd = {}
    for t in r.tensors:
        shape = tuple(int(x) for x in reversed(t.shape))
        sd[t.name[len("adapter."):]] = torch.from_numpy(np.asarray(t.data).astype(np.float32).reshape(shape).copy())
    resampler = "query" in sd
    seeded = resampler and "t5_embed.weight" in sd
    cfg = AdapterConfig(in_dim=int(kv["adapter.in_dim"]), out_dim=int(kv["adapter.out_dim"]),
                        width=int(kv["adapter.width"]), depth=int(kv["adapter.depth"]),
                        num_queries=int(kv.get("adapter.num_queries", sd["query"].shape[0] if resampler else 0)),
                        kind=(KIND_SEEDED if seeded else KIND_RESAMPLER) if resampler else KIND_TOKEN_VISION,
                        vis_dim=0 if resampler else int(kv.get("adapter.vis_dim", sd["vision_proj.weight"].shape[1])),
                        seed_vocab=int(sd["t5_embed.weight"].shape[0]) if seeded else 0)
    model = build_adapter(cfg)
    model.load_state_dict(sd)
    return model.eval(), cfg, kv
