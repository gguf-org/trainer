"""pig_clip -> teacher bridge adapters.  Two kinds, both under the engine's
"adapter." tensor prefix (ggk llm_adapter.hpp), head_dim 64, LayerNorm eps
1e-5, exact GELU:

RESAMPLER (trainer8 / LLaDA-Image), selected by the PRESENCE of `query` and
the absence of t5_embed*:

    kv  = in_proj(h)                        # student final-norm states, in_dim -> width
    q   = query (num_queries learned rows)  # no token-identity seed
    q   = Block(q, kv) * depth              # pre-LN self-attn, cross-attn to kv, exact-GELU MLP
    out = out_proj(ln_out(q))               # width -> out_dim

TOKEN-ALIGNED + VISION EXTENSION (trainer5 / MageFlow-Edit), selected by the
ABSENCE of `query`; the vision extension by the presence of `vision_proj.weight`:

    x   = in_proj(h) + vis_in(v)            # 1024 -> width, 2560 -> width (v = 0 at text positions)
    x   = TokenBlock(x) * depth             # pre-LN self-attn (bidirectional) + exact-GELU MLP
    out = out_proj(ln_out(x)) + skip(h)     # width -> out_dim, plus a direct linear path

SEEDED RESAMPLER (trainer / PixArt T5-XXL), selected by the presence of
BOTH `query` and `t5_embed.weight`:

    kv  = in_proj(h)
    q   = query + t5_embed[seed_ids]        # query i seeded with the teacher tokenizer's id at slot i
    q   = Block(q, kv) * depth
    out = out_proj(ln_out(q))               # one row per teacher slot (pads never supervised)

The seed is what lets a fixed query window follow the teacher's own
segmentation (T5 sentencepiece over Qwen BPE states); the engine keeps the
teacher tokenizer at inference and hands the adapter the same EOS + pad-0
window it builds for the mask, so the ids are free.

v is the RAW mmproj embed at vision positions, so vision fidelity does not
depend on what survives the 0.6B student.  vis_in is bias-free (zero input
-> zero contribution) and out_proj is zero-initialized (training starts
from the best linear map of the student states).  The module also carries
the FROZEN `vision_proj` [vis_dim -> in_dim] map the ENGINE applies to
every mmproj embed before the student LLM (fit once by least squares over
the shared vocabulary; the precomputed student states bake in the same map).
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

KIND_RESAMPLER = "resampler"
KIND_SEEDED = "seeded_resampler"
KIND_TOKEN_VISION = "token_aligned_vision"


@dataclass
class AdapterConfig:
    in_dim: int = 1024
    out_dim: int = 2560
    width: int = 1024
    depth: int = 6
    num_queries: int = 256          # resampler only
    mlp_ratio: float = 4.0
    kind: str = KIND_RESAMPLER
    vis_dim: int = 0                # token_aligned_vision only
    seed_vocab: int = 0             # seeded_resampler only: teacher tokenizer vocab (t5_embed rows)

    @property
    def heads(self):
        assert self.width % 64 == 0, "ggk assumes head_dim 64"
        return self.width // 64

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_ck(cls, d):
        d = dict(d)
        d.pop("heads", None)
        d.setdefault("kind", KIND_RESAMPLER)
        d.setdefault("vis_dim", 0)
        d.setdefault("seed_vocab", 0)
        return cls(**d)


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.o = nn.Linear(width, width)

    def forward(self, xq, xkv, keep_mask=None):
        B, Lq, W = xq.shape
        Lk = xkv.shape[1]
        hd = W // self.heads
        q = self.q(xq).view(B, Lq, self.heads, hd).transpose(1, 2)
        k = self.k(xkv).view(B, Lk, self.heads, hd).transpose(1, 2)
        v = self.v(xkv).view(B, Lk, self.heads, hd).transpose(1, 2)
        attn_mask = keep_mask[:, None, None, :] if keep_mask is not None else None
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.o(x.transpose(1, 2).reshape(B, Lq, W))


class Block(nn.Module):
    """Resampler block: self-attn over the queries, cross-attn to the student states, MLP."""

    def __init__(self, width, heads, mlp_ratio):
        super().__init__()
        self.ln_self = nn.LayerNorm(width)
        self.self_attn = Attention(width, heads)
        self.ln_q = nn.LayerNorm(width)
        self.ln_kv = nn.LayerNorm(width)
        self.cross_attn = Attention(width, heads)
        self.ln_mlp = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, q, kv, keep_mask):
        x = self.ln_self(q)
        q = q + self.self_attn(x, x)
        q = q + self.cross_attn(self.ln_q(q), self.ln_kv(kv), keep_mask)
        return q + self.mlp(self.ln_mlp(q))


class TokenBlock(nn.Module):
    """Token-aligned block: pre-LN bidirectional self-attention + MLP (ggk TokenBlock)."""

    def __init__(self, width, heads, mlp_ratio):
        super().__init__()
        self.ln_self = nn.LayerNorm(width)
        self.self_attn = Attention(width, heads)
        self.ln_mlp = nn.LayerNorm(width)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, x, keep_mask):
        h = self.ln_self(x)
        x = x + self.self_attn(h, h, keep_mask)
        return x + self.mlp(self.ln_mlp(x))


class ResamplerAdapter(nn.Module):
    """Resampler (kind resampler), or the seeded resampler (kind
    seeded_resampler) when cfg.seed_vocab > 0: the state-dict name of the
    seed table is `t5_embed` for every teacher, that is the engine's contract."""

    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.kind = KIND_SEEDED if cfg.seed_vocab > 0 else KIND_RESAMPLER
        self.grad_checkpoint = False
        self.query = nn.Parameter(torch.randn(cfg.num_queries, cfg.width) * 0.02)
        if cfg.seed_vocab > 0:
            self.t5_embed = nn.Embedding(cfg.seed_vocab, cfg.width)
            nn.init.normal_(self.t5_embed.weight, std=0.02)
        self.in_proj = nn.Linear(cfg.in_dim, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg.width, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.ln_out = nn.LayerNorm(cfg.width)
        self.out_proj = nn.Linear(cfg.width, cfg.out_dim)

    def forward(self, qwen_hidden, keep_mask=None, vis=None, seed_ids=None):
        # seed_ids [B, num_queries] int64: the teacher tokenizer's ids per slot
        # (seeded_resampler only; the plain resampler ignores them)
        B = qwen_hidden.shape[0]
        kv = self.in_proj(qwen_hidden)
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        if self.cfg.seed_vocab > 0:
            assert seed_ids is not None, "seeded_resampler needs seed_ids"
            q = q + self.t5_embed(seed_ids)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                q = torch.utils.checkpoint.checkpoint(blk, q, kv, keep_mask, use_reentrant=False)
            else:
                q = blk(q, kv, keep_mask)
        return self.out_proj(self.ln_out(q))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trained_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TokenAlignedVisionAdapter(nn.Module):
    kind = KIND_TOKEN_VISION

    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        assert cfg.vis_dim > 0, "token_aligned_vision needs vis_dim"
        self.cfg = cfg
        self.grad_checkpoint = False
        # frozen student-input map; fitted by qwen3vl_teacher.fit_vision_proj
        # and exported to the gguf for the engine
        self.vision_proj = nn.Linear(cfg.vis_dim, cfg.in_dim, bias=False)
        self.vision_proj.weight.requires_grad_(False)
        self.in_proj = nn.Linear(cfg.in_dim, cfg.width)
        self.vis_in = nn.Linear(cfg.vis_dim, cfg.width, bias=False)
        self.blocks = nn.ModuleList(TokenBlock(cfg.width, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.ln_out = nn.LayerNorm(cfg.width)
        self.out_proj = nn.Linear(cfg.width, cfg.out_dim)
        self.skip = nn.Linear(cfg.in_dim, cfg.out_dim)
        # start as the pure linear map; the residual branch grows from zero
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, qwen_hidden, keep_mask=None, vis=None):
        # qwen_hidden [B, L, in_dim] student final-norm states; vis [B, L, vis_dim]
        # raw mmproj embeds at vision positions, zeros elsewhere (None = text only)
        x = self.in_proj(qwen_hidden)
        if vis is not None:
            x = x + self.vis_in(vis)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, keep_mask, use_reentrant=False)
            else:
                x = blk(x, keep_mask)
        return self.out_proj(self.ln_out(x)) + self.skip(qwen_hidden)

    def set_vision_proj(self, weight: torch.Tensor):
        assert tuple(weight.shape) == tuple(self.vision_proj.weight.shape), weight.shape
        with torch.no_grad():
            self.vision_proj.weight.copy_(weight.to(self.vision_proj.weight.dtype))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trained_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_adapter(cfg: AdapterConfig) -> nn.Module:
    if cfg.kind == KIND_TOKEN_VISION:
        return TokenAlignedVisionAdapter(cfg)
    if cfg.kind == KIND_SEEDED:
        assert cfg.seed_vocab > 0, "seeded_resampler needs seed_vocab"
        return ResamplerAdapter(cfg)
    if cfg.kind == KIND_RESAMPLER:
        assert cfg.seed_vocab == 0, "a plain resampler carries no seed table"
        return ResamplerAdapter(cfg)
    raise ValueError(f"unknown adapter kind {cfg.kind}")


def adapter_config_for(pack, width: int, depth: int) -> AdapterConfig:
    return AdapterConfig(in_dim=pack.in_dim, out_dim=pack.out_dim, num_queries=pack.num_queries,
                         width=width, depth=depth, kind=pack.adapter_kind, vis_dim=pack.vis_dim,
                         seed_vocab=int(getattr(pack, "seed_vocab", 0) or 0))
