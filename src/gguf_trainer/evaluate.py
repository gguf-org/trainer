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

seeded resampler packs (PixArt T5):
  cos           per-slot cosine over the real teacher slots (judge this)
  cos_rms       cosine after per-row RMSNorm
  cos_eos       cosine at the EOS slot only (the hardest one)
  rel_mse       relative MSE over the real slots
  worst_row_cos the worst real slot in the val set

token-aligned vision packs:
  cos           per-token cosine over every real token
  cos_vis / cos_txt   split by vision vs text positions
  cos_slice     positions >= the template start (64 edit / 34 t2i) — the
                conditioning the DiT consumes (judge this)
  rel_mse       relative MSE over real tokens
"""

from __future__ import annotations

import pathlib
import time
from typing import Optional

import torch
import torch.nn.functional as F

from .adapter import KIND_SEEDED, KIND_TOKEN_VISION
from .devices import pick_device
from .export import checkpoint_file, gguf_adapter_model, load_folded_model
from .shards import ValSet
from .util import write_json_atomic


def rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


@torch.no_grad()
def evaluate(project, pack, log, gguf_path: Optional[pathlib.Path] = None, checkpoint: str = "best",
             out_path: Optional[pathlib.Path] = None, device: Optional[str] = None, write: bool = True) -> dict:
    """Default = the pipeline's eval stage: adapter_path() vs best.pt ->
    <project>/eval.json.  Snapshots evaluate their own GGUF against the
    checkpoint it came from and keep the result in the manifest (write=False)."""
    dev = pick_device(device or project.config["train"].get("device", "auto"))
    gguf_path = pathlib.Path(gguf_path) if gguf_path is not None else project.adapter_path()
    model, cfg, kv = gguf_adapter_model(gguf_path)
    model.to(dev)
    ck_model, _, ck = load_folded_model(checkpoint_file(project, checkpoint))
    ck_model.to(dev)
    val = ValSet(str(project.shards_dir / "val"), int(project.config["train"]["batch_size"]))
    if cfg.kind == KIND_TOKEN_VISION:
        res = _eval_token_aligned(model, ck_model, val, dev)
    elif cfg.kind == KIND_SEEDED:
        res = _eval_seeded(model, ck_model, val, dev)
    else:
        res = _eval_resampler(model, ck_model, val, dev, ck["mu"].to(dev))
    res.update({"gguf": str(gguf_path), "kind": cfg.kind, "trained_steps": int(kv.get("adapter.trained_steps", 0)),
                "checkpoint": checkpoint, "time": time.time(), "config": cfg.to_dict()})
    log("eval: " + ", ".join(f"{k} {v:.4f}" for k, v in res.items() if isinstance(v, float) and k != "time"))
    if write:
        write_json_atomic(out_path if out_path is not None else project.eval_path(), res)
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


def _eval_seeded(model, ck_model, val, dev):
    sums = {"cos": 0.0, "cos_rms": 0.0, "cos_eos": 0.0, "roundtrip_cos": 0.0}
    ns = {"cos": 0.0, "cos_rms": 0.0, "cos_eos": 0.0, "roundtrip_cos": 0.0}
    err = ref = 0.0
    worst = 1.0
    n_batches = 0
    for b in val:
        h = b["qwen_hidden"].to(dev).float()
        keep = b["keep"].to(dev)
        ids = b["seed_ids"].to(dev)
        m = b["seed_mask"].to(dev)
        t = b["target"].to(dev).float()
        p = model(h, keep, seed_ids=ids).float()
        p_ck = ck_model(h, keep, seed_ids=ids).float()
        mf = m.float()
        cos = F.cosine_similarity(p, t, dim=-1)
        rt = F.cosine_similarity(p, p_ck, dim=-1)
        c_rms = F.cosine_similarity(rms_norm(p), rms_norm(t), dim=-1)
        # the EOS slot = the last real slot of each row
        eos = torch.zeros_like(m)
        eos[torch.arange(m.shape[0], device=dev), m.long().sum(1).clamp_min(1) - 1] = True
        eos &= m
        for key, val_, sel in (("cos", cos, m), ("cos_rms", c_rms, m), ("cos_eos", cos, eos),
                               ("roundtrip_cos", rt, m)):
            sums[key] += (val_ * sel).sum().item()
            ns[key] += sel.sum().item()
        worst = min(worst, cos[m].min().item())
        err += (((p - t) ** 2) * mf.unsqueeze(-1)).sum().item()
        ref += ((t ** 2) * mf.unsqueeze(-1)).sum().item()
        n_batches += 1
    res = {k: sums[k] / max(ns[k], 1.0) for k in sums}
    res.update({"rel_mse": err / max(ref, 1e-8), "worst_row_cos": worst, "val_batches": n_batches})
    return res


def _eval_token_aligned(model, ck_model, val, dev):
    """Scores the rows the pack's contract supervises (batch["sup"]; every real
    token for MageFlow).  cos_vis / cos_txt = vision rows | text rows; when the
    image slots are not supervised the split is by sample instead and is
    reported as cos_edit (image samples) / cos_t2i (text-only samples)."""
    keys = ("cos", "cos_vis", "cos_txt", "cos_edit", "cos_t2i", "roundtrip_cos")
    tok_sum = {k: 0.0 for k in keys}
    tok_n = {k: 0.0 for k in keys}
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
        sup = b["sup"].to(dev) if b.get("sup") is not None else keep
        m = sup.float()
        cos = F.cosine_similarity(p, t, dim=-1)
        rt = F.cosine_similarity(p, p_ck, dim=-1)
        is_vis = b["is_vis"].to(dev) & sup
        is_txt = sup & ~is_vis
        has_img = torch.tensor([n > 0 for n in b["n_images"]], device=sup.device)[:, None]
        for key, sel in (("cos", sup), ("cos_vis", is_vis), ("cos_txt", is_txt),
                         ("cos_edit", sup & has_img), ("cos_t2i", sup & ~has_img)):
            tok_sum[key] += (cos * sel).sum().item()
            tok_n[key] += sel.sum().item()
        tok_sum["roundtrip_cos"] += (rt * keep).sum().item()
        tok_n["roundtrip_cos"] += keep.sum().item()
        err += (((p - t) ** 2) * m.unsqueeze(-1)).sum().item()
        ref += ((t ** 2) * m.unsqueeze(-1)).sum().item()
        for j, s0 in enumerate(b["sup_start"]):
            sl, ml = cos[j, s0:], m[j, s0:]
            slice_cos.append(((sl * ml).sum() / ml.sum().clamp_min(1.0)).item())
            n_samples += 1
        n_batches += 1
    # a split with no rows (e.g. cos_vis when the slots are unsupervised) is absent, not 0
    res = {k: tok_sum[k] / tok_n[k] for k in tok_sum if tok_n[k] > 0}
    res.update({"rel_mse": err / max(ref, 1e-8),
                "cos_slice": sum(slice_cos) / max(1, len(slice_cos)),
                "worst_sample_cos_slice": min(slice_cos) if slice_cos else float("nan"),
                "val_samples": n_samples, "val_batches": n_batches})
    return res
