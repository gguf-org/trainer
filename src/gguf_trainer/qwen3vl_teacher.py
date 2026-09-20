"""Qwen3-VL-4B-Instruct teacher, replicated the way ggk runs it for
MageFlow-Edit (trainer5 hf_models.py):

  - vision tower: main (merger) output only — NO deepstack (gk-diffuser.cpp
    ignores model.visual.deepstack_merger_list.*); pixels come through the
    engine's own preprocessing (vision_data.clip_preprocess);
  - text model: the vision embeds spliced into inputs_embeds at the
    <|image_pad|> runs, causal mask, position_ids=None (transformers expands
    arange over every M-RoPE section == ggk's all-equal M-RoPE == plain
    1-D rope), final-norm last_hidden_state (ggk mage_flow: out_layers = {});
  - placement: the text stack resident on the GPU when the budget allows,
    otherwise streamed through it with accelerate's cpu_offload (bf16
    weights in RAM, all compute on the GPU — faster than CPU layers).

The same splice helper feeds the student (pig_clip) with the identical
vision embeds mapped through the frozen vision_proj, and `fit_vision_proj`
computes that map (ridge least squares over the shared vocabulary).
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .vision_data import V_DIM, clip_preprocess, n_image_tokens, mageflow_target_size


# ---------------------------------------------------------------- vision

def patchify(img: np.ndarray, patch: int, merge: int, temporal: int):
    """[H, W, 3] float32 (already clip-normalized) -> ([n_patches, dim],
    grid_h, grid_w) exactly as Qwen2VLImageProcessorFast.patchify."""
    t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # C,H,W
    c, h, w = t.shape
    gh, gw = h // patch, w // patch
    p = t.reshape(c, gh // merge, merge, patch, gw // merge, merge, patch)
    p = p.permute(1, 4, 2, 5, 0, 3, 6)                      # gh/M, gw/M, M, M, C, P, P
    p = (p.unsqueeze(5).expand(-1, -1, -1, -1, -1, temporal, -1, -1)
         .reshape(gh * gw, c * temporal * patch * patch))
    return p, gh, gw


class VisionEncoder:
    """GPU-resident Qwen3-VL vision tower; returns the merged main embeds."""

    def __init__(self, visual, device, patch: int, merge: int, temporal: int, dtype=torch.bfloat16,
                 keep_deepstack: bool = False):
        self.m = visual.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.patch, self.merge, self.temporal = patch, merge, temporal
        # True: every returned main tensor carries `.deepstack` (list of
        # [n_tokens, D] bf16 cpu, one per deepstack layer) and `.grid` (h, w in
        # merged tokens) for a teacher that feeds them to the text stack
        self.keep_deepstack = keep_deepstack

    @torch.no_grad()
    def encode(self, imgs: Sequence[np.ndarray]) -> List[torch.Tensor]:
        """imgs: clip-normalized [H, W, 3] float32 arrays ->
        list of [n_tokens, V_DIM] bf16 cpu tensors (one tower pass)."""
        if not len(imgs):
            return []
        packs, grids = [], []
        for im in imgs:
            p, gh, gw = patchify(im, self.patch, self.merge, self.temporal)
            packs.append(p)
            grids.append((1, gh, gw))
        pixel = torch.cat(packs).to(self.device, self.dtype)
        grid_thw = torch.tensor(grids, dtype=torch.long, device=self.device)
        out = self.m(hidden_states=pixel, grid_thw=grid_thw)
        merged = out.pooler_output if hasattr(out, "pooler_output") else out[1]
        deep = []
        if self.keep_deepstack:
            deep = list(out.deepstack_features if hasattr(out, "deepstack_features") else out[2])
        outs, o = [], 0
        for (_, gh, gw) in grids:
            n = (gh // self.merge) * (gw // self.merge)
            main = merged[o: o + n].to(torch.bfloat16).cpu()
            if self.keep_deepstack:
                main.deepstack = [d[o: o + n].to(torch.bfloat16).cpu() for d in deep]
                main.grid = (gh // self.merge, gw // self.merge)
            outs.append(main)
            o += n
        assert o == merged.shape[0], (o, merged.shape)
        return outs


# ---------------------------------------------------------------- splice forward

def splice(embeds: torch.Tensor, vis_segments, vis_embeds) -> torch.Tensor:
    """In-place: embeds [L, D]; replace each (start, n) segment with the
    matching [n, D] vision tensor."""
    for (start, n), v in zip(vis_segments, vis_embeds):
        assert v.shape[0] == n, (v.shape, n)
        embeds[start: start + n] = v.to(embeds.dtype)
    return embeds


@torch.no_grad()
def batched_hidden(model, embed_weight: torch.Tensor, batch: List[dict], device, dtype=torch.bfloat16,
                   proj: Optional[torch.Tensor] = None, input_device=None) -> torch.Tensor:
    """One padded forward over a list of samples.

    model:        Qwen3VLTextModel or Qwen3Model (both: final-norm last_hidden_state)
    embed_weight: [vocab, D] embedding table (lookup happens on its device)
    batch:        dicts {ids: list[int], vis: [(start, n)], vis_embeds: [[n, V_DIM] tensors]}
    proj:         optional [D_out, V_DIM] map applied to the vision embeds before
                  the splice (the student's frozen vision_proj)
    -> [B, L, H] final-norm hidden states (dtype, on the model's device)
    """
    lens = [len(s["ids"]) for s in batch]
    L = max(lens)
    dev_e = embed_weight.device
    D = embed_weight.shape[1]
    embeds = torch.zeros(len(batch), L, D, dtype=dtype, device=dev_e)
    for j, s in enumerate(batch):
        ids = torch.as_tensor(s["ids"], dtype=torch.long, device=dev_e)
        e = embed_weight[ids].to(dtype)
        vv = s.get("vis_embeds") or []
        if vv:
            pv = [(v.to(dev_e).float() @ proj.T).to(dtype) if proj is not None else v.to(dev_e, dtype) for v in vv]
            e = splice(e, s["vis"], pv)
        embeds[j, : lens[j]] = e
    mask = (torch.arange(L, device=dev_e)[None, :] < torch.as_tensor(lens, device=dev_e)[:, None]).long()
    tgt = input_device or device
    out = model(inputs_embeds=embeds.to(tgt), attention_mask=mask.to(tgt))
    return out.last_hidden_state


# ---------------------------------------------------------------- sanity

def check_inv_freq_buffers(model) -> None:
    """transformers 5 builds models on the meta device; a rotary table that
    was not re-initialized shows up as garbage (see llada_compat).  Native
    Qwen3-VL recomputes its tables, but refuse to run rather than trust it."""
    from .llada_compat import rope_buffer_is_sane

    bad = [n for n, m in model.named_modules()
           if "inv_freq" in getattr(m, "_buffers", {}) and not rope_buffer_is_sane(m.inv_freq)]
    if bad:
        raise RuntimeError(f"rotary inv_freq buffers are not initialized in {bad}; refusing to run a "
                           f"teacher with scrambled positions")


# ---------------------------------------------------------------- teacher

class Qwen3VLTeacher:
    """Qwen3-VL-4B-Instruct as the engine runs it for MageFlow (see module
    docstring).  Qwen3VLFullTeacher below is the unreduced HF model."""

    def __init__(self, root: pathlib.Path, device: torch.device, gpu_mem_gib: float, teacher_mode: str = "auto",
                 log=print, dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        root = pathlib.Path(root)
        self.root = root
        self.dev = device
        self.dtype = dtype              # bf16 = the released weights; f32 only for reference checks
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(str(root))
        cfg = AutoConfig.from_pretrained(str(root))
        vc = cfg.vision_config
        self.patch = int(getattr(vc, "patch_size", 16))
        self.merge = int(getattr(vc, "spatial_merge_size", 2))
        self.temporal = int(getattr(vc, "temporal_patch_size", 2))
        self.vis_dim = int(getattr(vc, "out_hidden_size", V_DIM))
        self.out_dim = int(cfg.text_config.hidden_size)
        # the text stack alone is ~8 GB in bf16 (+ activations); the tower
        # ~1 GB.  Resident needs a comfortable budget, else stream it.
        text_gib = _param_bytes(root, exclude_prefix="model.visual.") / 2**30
        if teacher_mode == "auto":
            mode = "gpu" if (device.type == "cuda" and gpu_mem_gib >= text_gib + 3.0) else "offload"
        else:
            mode = teacher_mode
        if device.type != "cuda":
            mode = "cpu"
        self.mode = mode
        log(f"teacher {root.name}: text stack {text_gib:.1f} GiB bf16, GPU budget {gpu_mem_gib:.1f} GiB -> {mode}")
        m = AutoModel.from_pretrained(str(root), dtype=self.dtype)
        m.eval().requires_grad_(False)
        check_inv_freq_buffers(m)
        self.visual = VisionEncoder(m.visual, "cpu" if mode == "cpu" else device, self.patch, self.merge, self.temporal,
                                    dtype=self.dtype, keep_deepstack=getattr(self, "keep_deepstack", False))
        self.text = m.language_model
        self.embed_weight = self.text.embed_tokens.weight.detach()
        if mode == "gpu":
            self.text.to(device)
            self.embed_weight = self.text.embed_tokens.weight.detach()
        elif mode == "offload":
            from accelerate import cpu_offload

            self.embed_weight = self.embed_weight.clone()    # keep a plain copy: the hook streams the module's own
            cpu_offload(self.text, execution_device=torch.device(device))
        else:
            torch.set_num_threads(os.cpu_count() or 4)
        self.input_dev = torch.device("cpu") if mode == "cpu" else device
        log(f"teacher up in {time.time() - t0:.0f}s ({len(self.text.layers)} layers + final norm, deepstack "
            f"{'kept' if getattr(self, 'keep_deepstack', False) else 'dropped'})")

    def encode_images(self, imgs: Sequence[np.ndarray]) -> List[torch.Tensor]:
        return self.visual.encode(imgs)

    @torch.no_grad()
    def hidden(self, batch: List[dict]) -> torch.Tensor:
        """-> [B, L, out_dim] bf16 cpu final-norm hidden states."""
        h = batched_hidden(self.text, self.embed_weight, batch, self.dev, self.dtype, input_device=self.input_dev)
        return h.to(torch.bfloat16).cpu()

    def embed_table(self) -> torch.Tensor:
        return self.embed_weight.detach().float().cpu()


def mrope_positions(length: int, vis_segments, grids) -> torch.Tensor:
    """Qwen3-VL M-RoPE position ids of one sample -> [3, length] long
    (temporal, height, width).  Text tokens advance all three axes together;
    an image block of (h, w) merged tokens starting at position p gets
    t = p, h = p + row, w = p + col, and the text after it resumes at
    p + max(h, w) — Qwen3VLModel.get_rope_index for still images, written
    out so it does not depend on that method's version-specific signature
    (verified against the full HF forward)."""
    pos = torch.zeros(3, length, dtype=torch.long)
    nxt, cur = 0, 0
    for (start, n), (gh, gw) in zip(vis_segments, grids):
        assert gh * gw == n, (gh, gw, n)
        k = start - cur
        pos[:, cur:start] = torch.arange(nxt, nxt + k)[None]
        nxt += k
        rows = torch.arange(gh).repeat_interleave(gw)
        cols = torch.arange(gw).repeat(gh)
        pos[0, start: start + n] = nxt
        pos[1, start: start + n] = nxt + rows
        pos[2, start: start + n] = nxt + cols
        nxt += max(gh, gw)
        cur = start + n
    pos[:, cur:] = torch.arange(nxt, nxt + (length - cur))[None]
    return pos


class Qwen3VLFullTeacher(Qwen3VLTeacher):
    """Qwen3-VL as Hugging Face runs it — the reference the Qwen-Image 2.1 DiT
    was trained on, NOT the reduced path ggk runs for MageFlow:

      - deepstack ON: the tower's intermediate mergers are added to the first
        decoder layers at the image positions;
      - real M-RoPE: image tokens get (t, h, w) positions (mrope_positions);
      - tap = the last decoder layer BEFORE the final RMSNorm
        (diffusers QwenImage21Pipeline hooks the norm the same way; from
        transformers 5 `hidden_states[-1]` is the NORMALIZED state).

    The adapter still receives only the tower's main output (what the mmproj
    GGUF produces), so what deepstack contributed has to be learned from it.
    """

    keep_deepstack = True

    def __init__(self, root, device, gpu_mem_gib, teacher_mode="auto", log=print, dtype: torch.dtype = torch.bfloat16):
        super().__init__(root, device, gpu_mem_gib, teacher_mode, log, dtype=dtype)
        # a forward hook that returns the module's input replaces its output
        self.text.norm.register_forward_hook(lambda module, args, output: args[0])
        log("teacher tap: last decoder layer, pre final norm; deepstack + M-RoPE image positions on")

    @torch.no_grad()
    def hidden(self, batch: List[dict]) -> torch.Tensor:
        """-> [B, L, out_dim] bf16 cpu PRE-NORM last-layer hidden states."""
        lens = [len(s["ids"]) for s in batch]
        B, L = len(batch), max(lens)
        dev_e = self.embed_weight.device
        D = self.embed_weight.shape[1]
        embeds = torch.zeros(B, L, D, dtype=self.dtype, device=dev_e)
        pos = torch.ones(3, B, L, dtype=torch.long)
        vmask = torch.zeros(B, L, dtype=torch.bool)
        deep: List[List[torch.Tensor]] = []
        for j, s in enumerate(batch):
            ids = torch.as_tensor(s["ids"], dtype=torch.long, device=dev_e)
            e = self.embed_weight[ids].to(self.dtype)
            vv = s.get("vis_embeds") or []
            if vv:
                e = splice(e, s["vis"], [v.to(dev_e, self.dtype) for v in vv])
            embeds[j, : lens[j]] = e
            pos[:, j, : lens[j]] = mrope_positions(lens[j], s.get("vis") or [], [v.grid for v in vv])
            for (st, n), v in zip(s.get("vis") or [], vv):
                vmask[j, st: st + n] = True
                for k, d in enumerate(v.deepstack):
                    if len(deep) <= k:
                        deep.append([])
                    deep[k].append(d)
        tgt = self.input_dev
        mask = (torch.arange(L)[None, :] < torch.as_tensor(lens)[:, None]).long()
        kw = {}
        if deep:
            # row-major over [B, L]: the order the text stack reads the masked positions in
            kw = {"visual_pos_masks": vmask.to(tgt),
                  "deepstack_visual_embeds": [torch.cat(d).to(tgt, self.dtype) for d in deep]}
        out = self.text(inputs_embeds=embeds.to(tgt), attention_mask=mask.to(tgt), position_ids=pos.to(tgt), **kw)
        return out.last_hidden_state.to(torch.bfloat16).cpu()


def _param_bytes(root: pathlib.Path, exclude_prefix: str = "") -> int:
    """Bytes of the safetensors tensors under root, from the index / headers."""
    import struct

    total = 0
    for fp in sorted(root.glob("*.safetensors")):
        with open(fp, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__" or (exclude_prefix and name.startswith(exclude_prefix)):
                continue
            a, b = meta["data_offsets"]
            total += b - a
    return total


def load_embed_table(root: pathlib.Path) -> torch.Tensor:
    """The text embedding table straight from the safetensors (no model)."""
    from safetensors import safe_open

    candidates = ["model.language_model.embed_tokens.weight", "language_model.embed_tokens.weight",
                  "model.embed_tokens.weight"]
    idx = root / "model.safetensors.index.json"
    files = []
    if idx.is_file():
        wm = json.loads(idx.read_text())["weight_map"]
        for c in candidates:
            if c in wm:
                files = [root / wm[c]]
                break
    if not files:
        files = sorted(root.glob("*.safetensors"))
    for fp in files:
        with safe_open(str(fp), framework="pt") as sf:
            keys = set(sf.keys())
            for c in candidates:
                if c in keys:
                    return sf.get_tensor(c)
    raise RuntimeError(f"embed_tokens.weight not found under {root}")


def fit_vision_proj(teacher_table: torch.Tensor, student_table: torch.Tensor, ridge: float = 1e-4, log=print):
    """Ridge least squares  min_W ||E_teacher W^T - E_student||^2 + lambda ||W||^2
    over the shared vocabulary -> Linear weight [S_DIM, V_DIM] float32.
    Both models tokenize identically, so row i of both tables is the same
    token; the map that aligns them on 151936 real tokens is a sound fixed
    projection for the mmproj soft tokens (they live in the teacher's input
    embedding space by construction)."""
    et = teacher_table.to(torch.float64)
    es = student_table.to(torch.float64)
    if et.shape[0] != es.shape[0]:
        raise RuntimeError(f"vocabulary size mismatch: teacher {tuple(et.shape)} vs student {tuple(es.shape)}")
    g = et.T @ et
    lam = ridge * g.diagonal().mean()
    g += lam * torch.eye(g.shape[0], dtype=g.dtype)
    b = et.T @ es
    wt = torch.linalg.solve(g, b)
    w = wt.T.contiguous().to(torch.float32)
    pred = et.float() @ w.T
    cos = torch.nn.functional.cosine_similarity(pred, es.float(), dim=-1)
    rel = ((pred - es.float()) ** 2).sum() / (es.float() ** 2).sum()
    log(f"vision_proj vocab fit: cos mean {cos.mean():.4f} p05 {cos.quantile(0.05):.4f} | rel_mse {rel:.4f}")
    return w, float(cos.mean())


# ---------------------------------------------------------------- mock

class MockVisionTeacher:
    """Developer stand-in (GGUF_TRAINER_MOCK_TEACHER=1): synthetic but
    deterministic vision embeds and hidden states with the real shapes,
    so the whole image pipeline runs on a laptop without the 8 GB teacher."""

    def __init__(self, out_dim: int, vis_dim: int, log=print):
        g = torch.Generator().manual_seed(4321)
        self.out_dim, self.vis_dim = out_dim, vis_dim
        self.mode = "mock"
        self.pix_basis = torch.randn(48, vis_dim, generator=g) / 4
        self.tok_basis = torch.randn(64, out_dim, generator=g) / 4
        self.vis_mix = torch.randn(vis_dim, out_dim, generator=g) / vis_dim ** 0.5
        log("MOCK teacher in use: targets are synthetic, the adapter will be meaningless")

    def encode_images(self, imgs):
        outs = []
        for im in imgs:
            h, w = im.shape[:2]
            gh, gw = h // 32, w // 32
            cells = im.reshape(gh, 32, gw, 32, 3).mean(axis=(1, 3)).reshape(gh * gw, 3)
            feats = torch.zeros(gh * gw, 48)
            f = torch.from_numpy(cells.astype(np.float32))
            feats[:, :3] = f
            feats[:, 3:6] = f * f
            pos = torch.arange(gh * gw, dtype=torch.float32)[:, None]
            feats[:, 6:48] = torch.sin(pos * torch.arange(1, 43, dtype=torch.float32)[None, :] / 7.0)
            outs.append((feats @ self.pix_basis).to(torch.bfloat16))
        return outs

    def hidden(self, batch):
        B = len(batch)
        L = max(len(s["ids"]) for s in batch)
        out = torch.zeros(B, L, self.out_dim)
        for j, s in enumerate(batch):
            ids = torch.as_tensor(s["ids"], dtype=torch.long)
            feats = torch.zeros(len(ids), 64)
            feats[torch.arange(len(ids)), ids % 64] = 1.0
            feats[:, 0] += torch.arange(len(ids), dtype=torch.float32) / 100.0
            h = feats @ self.tok_basis
            for (st, n), v in zip(s.get("vis", []), s.get("vis_embeds") or []):
                h[st: st + n] += v.float() @ self.vis_mix
            out[j, : len(ids)] = h
        return out.to(torch.bfloat16)

    def embed_table(self):
        return None
