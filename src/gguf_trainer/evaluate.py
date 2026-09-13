"""Eval stage: what the DiT actually sees, measured on the val shards from
the EXPORTED GGUF (so the folded weights and the f16 cast are part of the
number), plus a checkpoint-vs-GGUF round-trip check.  Written to
<project>/eval.json (next to project.json, so several projects can share
one output folder without their evaluations overwriting each other).

resampler packs:
  cos_raw       plain per-row cosine (inflated by the rows' shared offset)
  cos_rms       cosine after per-row RMSNorm (== what the DiT consumes)
  cos_centered  cosine after subtracting the per-dim target mean (judge this)
  rel_mse       relative MSE (scale-aware)

token-aligned vision packs:
  cos           per-token cosine over every real token
  cos_vis / cos_txt   split by vision vs text positions
  cos_slice     positions >= the template start (64 edit / 34 t2i) — the
                conditioning the DiT consumes (judge this)
  rel_mse       relative MSE over real tokens
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .adapter import KIND_TOKEN_VISION
from .devices import pick_device
from .export import gguf_adapter_model, load_folded_model
from .shards import ValSet
from .util import write_json_atomic


def rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


@torch.no_grad()
def evaluate(project, pack, log) -> dict:
    dev = pick_device(project.config["train"].get("device", "auto"))
    gguf_path = project.adapter_path()
    model, cfg, kv = gguf_adapter_model(gguf_path)
    model.to(dev)
    ck_model, _, ck = load_folded_model(project.checkpoints_dir / "best.pt")
    ck_model.to(dev)
    val = ValSet(str(project.shards_dir / "val"), int(project.config["train"]["batch_size"]))
    if cfg.kind == KIND_TOKEN_VISION:
        res = _eval_token_aligned(model, ck_model, val, dev)
    else:
        res = _eval_resampler(model, ck_model, val, dev, ck["mu"].to(dev))
    res.update({"gguf": str(gguf_path), "kind": cfg.kind, "trained_steps": int(kv.get("adapter.trained_steps", 0)),
                "time": time.time(), "config": cfg.to_dict()})
    log("eval: " + ", ".join(f"{k} {v:.4f}" for k, v in res.items() if isinstance(v, float) and k != "time"))
    write_json_atomic(project.eval_path(), res)
    return res


def _eval_resampler(model, ck_model, val, dev, mu):
    sums = {"cos_raw": 0.0, "cos_rms": 0.0, "cos_centered": 0.0, "rel_mse": 0.0, "roundtrip_cos": 0.0}
    worst_rms = 1.0
    n = 0
    for b in val:
        h = b["qwen_hidden"].to(dev).float()
        keep = b["keep"].to(dev)
        t = b["target"].to(dev).float()
        p = model(h, keep).float()
        p_ck = ck_model(h, keep).float()
        sums["roundtrip_cos"] += F.cosine_similarity(p - mu, p_ck - mu, dim=-1).mean().item()
        sums["cos_raw"] += F.cosine_similarity(p, t, dim=-1).mean().item()
        c_rms = F.cosine_similarity(rms_norm(p), rms_norm(t), dim=-1)
        sums["cos_rms"] += c_rms.mean().item()
        worst_rms = min(worst_rms, c_rms.min().item())
        sums["cos_centered"] += F.cosine_similarity(p - mu, t - mu, dim=-1).mean().item()
        sums["rel_mse"] += (((p - t) ** 2).sum() / (t ** 2).sum()).item()
        n += 1
    res = {k: v / max(1, n) for k, v in sums.items()}
    res.update({"worst_row_cos_rms": worst_rms, "val_batches": n})
    return res


def _eval_token_aligned(model, ck_model, val, dev):
    from .vision_data import start_idx

    tok_sum = {"cos": 0.0, "cos_vis": 0.0, "cos_txt": 0.0, "roundtrip_cos": 0.0}
    tok_n = {"cos": 0.0, "cos_vis": 0.0, "cos_txt": 0.0, "roundtrip_cos": 0.0}
    err = ref = 0.0
    slice_cos = []
    n_samples = 0
    n_batches = 0
    for b in val:
        h = b["qwen_hidden"].to(dev).float()
        vis = b["vis"].to(dev).float() if b.get("vis") is not None else None
        keep = b["keep"].to(dev)
        t = b["target"].to(dev).float()
        p = model(h, keep, vis).float()
        p_ck = ck_model(h, keep, vis).float()
        m = keep.float()
        cos = F.cosine_similarity(p, t, dim=-1)
        rt = F.cosine_similarity(p, p_ck, dim=-1)
        is_vis = b["is_vis"].to(dev) & keep
        is_txt = keep & ~is_vis
        for key, sel in (("cos", keep), ("cos_vis", is_vis), ("cos_txt", is_txt)):
            tok_sum[key] += (cos * sel).sum().item()
            tok_n[key] += sel.sum().item()
        tok_sum["roundtrip_cos"] += (rt * keep).sum().item()
        tok_n["roundtrip_cos"] += keep.sum().item()
        err += (((p - t) ** 2) * m.unsqueeze(-1)).sum().item()
        ref += ((t ** 2) * m.unsqueeze(-1)).sum().item()
        for j, ni in enumerate(b["n_images"]):
            s0 = start_idx(ni)
            sl, ml = cos[j, s0:], m[j, s0:]
            slice_cos.append(((sl * ml).sum() / ml.sum().clamp_min(1.0)).item())
            n_samples += 1
        n_batches += 1
    res = {k: tok_sum[k] / max(tok_n[k], 1.0) for k in tok_sum}
    res.update({"rel_mse": err / max(ref, 1e-8),
                "cos_slice": sum(slice_cos) / max(1, len(slice_cos)),
                "worst_sample_cos_slice": min(slice_cos) if slice_cos else float("nan"),
                "val_samples": n_samples, "val_batches": n_batches})
    return res
