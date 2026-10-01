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

MING-IMAGE (gguf_trainer ming_teacher / Ming-Image 0.1 Design + Layer), selected
by the presence of `cap_query`:

    kv   = in_proj(h) ++ vis_in(v)                  # student states, plus one kv row per raw
                                                    # vision embed (2048, the Ling tower output)
    q    = cap_query (256 rows)                     # -> the connector's caption tokens
        ++ slot_query[:P] + seed_proj(seed_embed[ling_ids]) + vis_q(v at the image slots)
                                                    # one row per Ling prompt token (P of them)
    q    = Block(q, kv) * depth                     # joint self-attn over both banks, cross-attn to kv
    cap  = out_cap(ln_out(q[:256]))                 # width -> 2560
    dir  = out_proj(ln_out(q[256:]))                # width -> 3840

The direct rows keep the TEACHER's token count and identity (the engine
tokenizes the prompt with the Ling tokenizer for the seed ids, like the T5
path), so the DiT sees exactly the sequence it was trained on; `seed_embed`
is a frozen PCA of the Ling embedding table (unseen tokens still get a
meaningful seed), `seed_proj` is trained.  Image-slot queries carry their
own vision feature (vis_q), the kv rows carry all of them (vis_in).

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
KIND_MING = "ming_image"


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
    # ming_image only
    n_cap: int = 0                  # caption query rows (256)
    cap_dim: int = 0                # caption head width (2560)
    max_slots: int = 0              # slot_query rows = the longest Ling prompt supported
    seed_rank: int = 0              # seed_embed columns (PCA rank of the Ling embedding table)

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
        for k in ("n_cap", "cap_dim", "max_slots", "seed_rank"):
            d.setdefault(k, 0)
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

    def forward(self, q, kv, keep_mask, q_mask=None):
        x = self.ln_self(q)
        q = q + self.self_attn(x, x, q_mask)
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


class MingAdapter(nn.Module):
    """Ming-Image adapter (kind ming_image), see the module docstring."""

    kind = KIND_MING

    def __init__(self, cfg: AdapterConfig):
        super().__init__()
        assert cfg.n_cap > 0 and cfg.cap_dim > 0 and cfg.max_slots > 0 and cfg.seed_rank > 0 and cfg.seed_vocab > 0 \
            and cfg.vis_dim > 0, "ming_image needs n_cap, cap_dim, max_slots, seed_rank, seed_vocab, vis_dim"
        self.cfg = cfg
        self.grad_checkpoint = False
        self.cap_query = nn.Parameter(torch.randn(cfg.n_cap, cfg.width) * 0.02)
        self.slot_query = nn.Parameter(torch.randn(cfg.max_slots, cfg.width) * 0.02)
        # frozen PCA of the teacher's token embedding table (set_seed_table), exported f16
        self.seed_embed = nn.Embedding(cfg.seed_vocab, cfg.seed_rank)
        self.seed_embed.weight.requires_grad_(False)
        self.seed_proj = nn.Linear(cfg.seed_rank, cfg.width, bias=False)
        self.vis_q = nn.Linear(cfg.vis_dim, cfg.width, bias=False)
        self.in_proj = nn.Linear(cfg.in_dim, cfg.width)
        self.vis_in = nn.Linear(cfg.vis_dim, cfg.width, bias=False)
        self.blocks = nn.ModuleList(Block(cfg.width, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth))
        self.ln_out = nn.LayerNorm(cfg.width)
        self.out_cap = nn.Linear(cfg.width, cfg.cap_dim)
        self.out_proj = nn.Linear(cfg.width, cfg.out_dim)

    def set_seed_table(self, table: torch.Tensor):
        assert tuple(table.shape) == tuple(self.seed_embed.weight.shape), (table.shape, self.seed_embed.weight.shape)
        with torch.no_grad():
            self.seed_embed.weight.copy_(table.to(self.seed_embed.weight.dtype))

    def forward(self, qwen_hidden, keep_mask, seed_ids, slot_mask, vis=None, vis_keep=None, slot_vis_index=None):
        """qwen_hidden [B, L, in_dim] student final-norm states, keep_mask [B, L];
        seed_ids [B, P] Ling prompt ids (P <= max_slots), slot_mask [B, P] real
        slots; vis [B, V, vis_dim] raw vision embeds (one image per sample,
        zero rows past vis_keep [B, V]); slot_vis_index [B, P] long: the vis
        row feeding slot i, or V (a zero row) for non-image slots.
        -> (cap [B, n_cap, cap_dim], direct [B, P, out_dim])"""
        B, P = seed_ids.shape
        assert P <= self.cfg.max_slots, f"{P} Ling tokens exceed the adapter's {self.cfg.max_slots} slots"
        kv = self.in_proj(qwen_hidden)
        kv_mask = keep_mask
        if vis is not None:
            kv = torch.cat([kv, self.vis_in(vis)], dim=1)
            kv_mask = torch.cat([keep_mask, vis_keep], dim=1)
        q_cap = self.cap_query.unsqueeze(0).expand(B, -1, -1)
        q_slot = self.slot_query[:P].unsqueeze(0) + self.seed_proj(self.seed_embed(seed_ids))
        if vis is not None and slot_vis_index is not None:
            vpad = torch.cat([vis, torch.zeros_like(vis[:, :1])], dim=1)                   # row V = zeros
            gathered = torch.gather(vpad, 1, slot_vis_index.unsqueeze(-1).expand(-1, -1, vis.shape[-1]))
            q_slot = q_slot + self.vis_q(gathered)
        q = torch.cat([q_cap, q_slot], dim=1)
        q_mask = torch.cat([torch.ones(B, self.cfg.n_cap, dtype=torch.bool, device=q.device), slot_mask.bool()], dim=1)
        for blk in self.blocks:
            if self.grad_checkpoint and self.training:
                q = torch.utils.checkpoint.checkpoint(blk, q, kv, kv_mask, q_mask, use_reentrant=False)
            else:
                q = blk(q, kv, kv_mask, q_mask)
        q = self.ln_out(q)
        return self.out_cap(q[:, : self.cfg.n_cap]), self.out_proj(q[:, self.cfg.n_cap:])

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def num_trained_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_adapter(cfg: AdapterConfig) -> nn.Module:
    if cfg.kind == KIND_MING:
        return MingAdapter(cfg)
    if cfg.kind == KIND_TOKEN_VISION:
        return TokenAlignedVisionAdapter(cfg)
    if cfg.kind == KIND_SEEDED:
        assert cfg.seed_vocab > 0, "seeded_resampler needs seed_vocab"
        return ResamplerAdapter(cfg)
    if cfg.kind == KIND_RESAMPLER:
        assert cfg.seed_vocab == 0, "a plain resampler carries no seed table"
        return ResamplerAdapter(cfg)
    raise ValueError(f"unknown adapter kind {cfg.kind}")


def adapter_config_for(pack, width: int, depth: int, train_cfg: dict = None) -> AdapterConfig:
    """The adapter geometry of a pack at (width, depth); ming_image packs read
    max_slots / seed_rank from the project's train settings (pack defaults)."""
    tc = train_cfg or {}
    extra = {}
    if pack.adapter_kind == KIND_MING:
        extra = {"n_cap": int(pack.num_queries), "cap_dim": int(pack.cap_dim),
                 "max_slots": int(tc.get("max_slots") or getattr(pack, "max_slots", 0)),
                 "seed_rank": int(tc.get("seed_rank") or getattr(pack, "seed_rank", 0))}
    return AdapterConfig(in_dim=pack.in_dim, out_dim=pack.out_dim,
                         num_queries=0 if pack.adapter_kind == KIND_MING else pack.num_queries,
                         width=width, depth=depth, kind=pack.adapter_kind, vis_dim=pack.vis_dim,
                         seed_vocab=int(getattr(pack, "seed_vocab", 0) or 0), **extra)
