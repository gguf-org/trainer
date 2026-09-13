"""LLaDA-Image-Turbo text conditioning teacher, without diffusers.

QueryFormer and text_projection are re-implemented in plain torch (their
diffusers originals are ~150 lines of parameter-free norms + attention) and
load the HF safetensors directly, so the trainer needs no diffusers install
and no checkout of the LLaDA-Image reference repo.  The LLaDA2-MoE backbone
is loaded through transformers' trust_remote_code path from the snapshot's
own modeling file, spread over every CUDA device before spilling to CPU RAM
(accelerate device_map).

Batched exactly like LLaDAImagePipeline._encode_text (verified in trainer8
contract_test.py: cos 0.9999 against the single-prompt path): right padding,
key-padding mask, text->query block masked, positions = cumsum of the mask.

Ground truth: with the rotary tables repaired this teacher matches the
teacher-conditioned engine context of trainer8 (dumps/m1_gpu_full) at
centred cosine 0.998 under transformers 5.15 and 4.57 alike; without the
repair transformers 5 gives 0.887 (SHARD_CONTRACT 1, now rejected).
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    # diffusers RMSNorm(elementwise_affine=False): fp32 variance, cast back
    dtype = x.dtype
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + eps)).to(dtype)


def _attention(q, k, v, heads, attn_mask=None):
    # q/k/v [B, L, H*D] -> sdpa over [B, H, L, D]
    B, Lq, HD = q.shape
    D = HD // heads
    q = q.view(B, Lq, heads, D).transpose(1, 2)
    k = k.view(B, -1, heads, D).transpose(1, 2)
    v = v.view(B, -1, heads, D).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return o.transpose(1, 2).reshape(B, Lq, HD)


class QueryFormerBlock(nn.Module):
    def __init__(self, hidden, heads, inter, eps):
        super().__init__()
        self.hidden, self.heads, self.eps = hidden, heads, eps
        self.cross_attn = nn.Module()
        self.cross_attn.in_proj_weight = nn.Parameter(torch.zeros(3 * hidden, hidden))
        self.cross_attn.in_proj_bias = nn.Parameter(torch.zeros(3 * hidden))
        self.cross_attn.out_proj = nn.Linear(hidden, hidden)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden, inter)
        self.mlp.fc2 = nn.Linear(inter, hidden)

    def forward(self, q, enc, keep_mask):
        H = self.hidden
        q = F.layer_norm(q, (H,), eps=self.eps)
        enc = F.layer_norm(enc, (H,), eps=self.eps)
        w, b = self.cross_attn.in_proj_weight, self.cross_attn.in_proj_bias
        qq = F.linear(q, w[:H], b[:H])
        kk = F.linear(enc, w[H:2 * H], b[H:2 * H])
        vv = F.linear(enc, w[2 * H:], b[2 * H:])
        a = _attention(qq, kk, vv, self.heads, keep_mask[:, None, None, :])
        q = q + self.cross_attn.out_proj(a)
        q = F.layer_norm(q, (H,), eps=self.eps)
        return q + self.mlp.fc2(F.gelu(self.mlp.fc1(q), approximate="tanh"))


class QueryFormer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden = cfg.get("hidden_size", 2048)
        self.meta_queries = nn.Parameter(torch.zeros(cfg.get("num_queries", 256), hidden))
        self.query_blocks = nn.ModuleList(
            QueryFormerBlock(hidden, cfg.get("num_attention_heads", 16), cfg.get("intermediate_size", 8192),
                             cfg.get("norm_eps", 1e-6)) for _ in range(cfg.get("num_hidden_layers", 1)))

    def forward(self, inputs_embeds, keep_mask):
        q = self.meta_queries.unsqueeze(0).expand(inputs_embeds.shape[0], -1, -1)
        for blk in self.query_blocks:
            q = blk(q, inputs_embeds, keep_mask.bool())
        return q


class TextProjectionBlock(nn.Module):
    def __init__(self, hidden, inter, heads, eps):
        super().__init__()
        self.heads, self.eps = heads, eps
        self.head_dim = hidden // heads
        self.self_attn = nn.Module()
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.self_attn, n, nn.Linear(hidden, hidden))
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden, inter)
        self.mlp.fc2 = nn.Linear(inter, hidden)

    def forward(self, x):
        h = _rms_norm(x, self.eps)
        B, L, HD = h.shape
        q = self.self_attn.q_proj(h).view(B, L, self.heads, self.head_dim)
        k = self.self_attn.k_proj(h).view(B, L, self.heads, self.head_dim)
        v = self.self_attn.v_proj(h).view(B, L, self.heads, self.head_dim)
        q = _rms_norm(q, self.eps)
        k = _rms_norm(k, self.eps)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        x = x + self.self_attn.out_proj(a.transpose(1, 2).reshape(B, L, HD))
        h = _rms_norm(x, self.eps)
        return x + self.mlp.fc2(F.gelu(self.mlp.fc1(h), approximate="tanh"))


class TextProjection(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        hidden = cfg.get("hidden_size", 2048)
        self.layers = nn.ModuleList(
            TextProjectionBlock(hidden, cfg.get("intermediate_size", 8960), cfg.get("num_attention_heads", 32),
                                cfg.get("norm_eps", 1e-6)) for _ in range(cfg.get("num_hidden_layers", 6)))
        self.projector = nn.Linear(hidden, cfg.get("projection_dim", 2560))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.projector(x)


def load_diffusers_module(cls, subdir: pathlib.Path, device, dtype) -> nn.Module:
    """Build `cls` from <subdir>/config.json and load its safetensors strictly."""
    from safetensors.torch import load_file

    with open(subdir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    m = cls(cfg)
    sd = load_file(str(subdir / "diffusion_pytorch_model.safetensors"))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{cls.__name__}: missing {missing} unexpected {unexpected}")
    return m.to(device=device, dtype=dtype).eval().requires_grad_(False)


class LLaDATeacher:
    """texts -> [B, 256, 2560] bf16 (cpu): the QueryFormer rows of cap_feats."""

    def __init__(self, root: pathlib.Path, device: torch.device, gpu_mem: str, cpu_mem: str,
                 format_prompt: Callable[[str], str], max_len: int = 2048, log=print,
                 placement: str = "sequential"):
        from transformers import AutoModel, AutoTokenizer

        from .llada_compat import check_rope_buffers, ensure_transformers_compat, repair_rope_buffers

        ensure_transformers_compat()
        root = pathlib.Path(root)
        self.dev = device
        self.dtype = torch.bfloat16
        self.format_prompt = format_prompt
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
        self.tok.padding_side = "right"
        t0 = time.time()
        max_memory = {}
        if device.type == "cuda":
            max_memory[device.index if device.index is not None else torch.cuda.current_device()] = gpu_mem
            # every other CUDA device before the CPU: a CPU-resident MoE layer
            # runs its 256 experts eagerly and halves the throughput
            for i in range(torch.cuda.device_count()):
                if i not in max_memory:
                    free_gib = torch.cuda.get_device_properties(i).total_memory / 2**30 - 1.5
                    if free_gib > 2:
                        max_memory[i] = f"{free_gib:.1f}GiB"
        max_memory["cpu"] = cpu_mem
        # "auto" is NOT "use the budgets": transformers first balances the
        # model evenly over the GPUs (get_balanced_memory), which capped a
        # 5090 at ~15 GB of a 30 GB budget and pushed 12 MoE layers to the CPU
        # (2.4 p/s).  "sequential" fills the chosen device to its budget,
        # then the next GPU, then CPU RAM.
        device_map = "balanced" if placement == "balanced" else "sequential"
        log(f"teacher max_memory: {max_memory} (placement {device_map})")
        self.te = AutoModel.from_pretrained(str(root / "text_encoder"), dtype=self.dtype, trust_remote_code=True,
                                            device_map=device_map, max_memory=max_memory).eval()
        # transformers 5 leaves the remote model's non-persistent rotary
        # tables uninitialized (see llada_compat): repair, then refuse to run
        # with a bad table — a teacher with scrambled positions still emits
        # plausible rows, and the adapter would learn to ignore the prompt.
        repair_rope_buffers(self.te, log)
        check_rope_buffers(self.te)
        self.qf = load_diffusers_module(QueryFormer, root / "queryformer", device, self.dtype)
        self.proj = load_diffusers_module(TextProjection, root / "text_projection", device, self.dtype)
        self.embed = self.te.get_input_embeddings()
        self.embed_dev = self.embed.weight.device
        self.num_queries = self.qf.meta_queries.shape[0]
        dm = getattr(self.te, "hf_device_map", None) or {}
        placement = {}
        for v in dm.values():
            placement[str(v)] = placement.get(str(v), 0) + 1
        # transformers 5 keeps CPU-offloaded layers on the meta device and
        # streams them from the memory-mapped safetensors at forward time
        log(f"teacher up in {time.time() - t0:.0f}s; backbone modules per device {placement or 'n/a'}")

    def text_len(self, text: str) -> int:
        return len(self.tok(self.format_prompt(text), add_special_tokens=True).input_ids)

    @torch.no_grad()
    def __call__(self, texts: List[str]) -> torch.Tensor:
        enc = self.tok([self.format_prompt(t) for t in texts], add_special_tokens=True, padding=True,
                       truncation=True, max_length=self.max_len, return_tensors="pt")
        ids = enc.input_ids.to(self.embed_dev)
        attn = enc.attention_mask.bool()
        B, L = ids.shape
        NQ = self.num_queries
        emb = self.embed(ids)                                                   # [B, L, 2048]
        q = self.qf(emb.to(self.dev, self.dtype), attn.to(self.dev))            # [B, 256, 2048]
        emb = torch.cat([emb, q.to(self.embed_dev, emb.dtype)], 1)
        attn_full = torch.cat([attn, torch.ones(B, NQ, dtype=torch.bool)], 1).to(self.embed_dev)
        pos = attn_full.long().cumsum(1) - 1
        pos.masked_fill_(pos < 0, 0)
        neg = torch.finfo(emb.dtype).min
        m = torch.where(attn_full[:, None, None, :], torch.zeros((), dtype=emb.dtype, device=self.embed_dev),
                        torch.full((), neg, dtype=emb.dtype, device=self.embed_dev))
        m = m.expand(-1, 1, L + NQ, -1).clone()
        m[:, :, :L, L:] = neg                                                    # text cannot see queries
        hidden = self.te.model(inputs_embeds=emb, attention_mask=m, position_ids=pos,
                               return_dict=True).last_hidden_state
        cap = self.proj(hidden.to(self.dev, self.dtype))                        # [B, L+256, 2560]
        return cap[:, L:, :].to(torch.bfloat16).cpu()


class MockTeacher:
    """Developer stand-in (GGUF_TRAINER_MOCK_TEACHER=1): a fixed random
    projection of hashed prompt bytes, so the whole pipeline can be exercised
    on a laptop without the 33 GB teacher.  Never use for a real adapter."""

    def __init__(self, num_queries: int, out_dim: int, format_prompt, log=print):
        g = torch.Generator().manual_seed(1234)
        self.num_queries, self.out_dim = num_queries, out_dim
        self.format_prompt = format_prompt
        self.basis = torch.randn(64, num_queries * out_dim, generator=g) / 8
        self.offset = 2000.0 + torch.randn(out_dim, generator=g) * 50
        log("MOCK teacher in use: targets are synthetic, the adapter will be meaningless")

    def text_len(self, text: str) -> int:
        return len(self.format_prompt(text)) // 4 + 1

    def __call__(self, texts):
        out = []
        for t in texts:
            s = self.format_prompt(t).encode("utf-8")
            feats = torch.zeros(64)
            for i, b in enumerate(s[:512]):
                feats[(i * 7 + b) % 64] += (b - 96) / 32.0
            out.append((feats @ self.basis).view(self.num_queries, self.out_dim) + self.offset)
        return torch.stack(out).to(torch.bfloat16)


def use_mock_teacher() -> bool:
    return os.environ.get("GGUF_TRAINER_MOCK_TEACHER", "") not in ("", "0")
