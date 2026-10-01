"""Ming-Image 0.1 (Design / Design-Layer) text-encoder teacher: the Ling-mini-2.0
MLLM (inclusionAI/Ming-Image-0.1-Design, mllm/ + connector/ + mlp/) as the
reference pipeline runs it for image generation
(modeling_bailingmm2.get_condition_embeds_for_image_gen).

Sequence (tokenization_bailing + processing_bailingmm2.apply_chat_template):

    <role>SYSTEM</role>你是一个友好的AI助手。\\n\\ndetailed thinking off<|role_end|>
    <role>HUMAN</role>[<image><imagePatch>*n</image>\\n]{text}<|role_end|><role>ASSISTANT</role>
    + <image><imagePatch>*256</image>            (the generation block, appended)

  * the input image (editing / layer decomposition) is spliced over its own
    <imagePatch> run: Qwen2.5-VL tower -> linear_proj (Linear, GELU, Linear)
    -> F.normalize; the 256 learned query_tokens over the generation block;
  * the thinker = BailingMoeV2 (20 layers, layer 0 dense, 256 experts top-8,
    group-limited 4/8, sigmoid + expert_bias routing, MultiRouter: image
    tokens - input image AND query block - route through image_gate), fused
    qkv with per-head q/k RMSNorm, partial (64/128) "video rope" at theta 6e5;
  * caption  = proj_out(connector(proj_in(h_final[query block])))  [256, 2560]
    (Qwen2-1.5B-shaped 28-layer stack, bidirectional, positions 0..255,
    connector_norm false -> no F.normalize);
  * direct   = proj_directvlm(cat(hidden_states[5], [12], [20]))[:n_prompt]
    [n_prompt, 3840] over EVERY prompt token (system prefix, image block,
    text, suffix); hidden_states[20] is the final-norm output.

The DiT consumes cap_embedder(caption) ++ direct.  Both are the targets of
the ming_image adapter (adapter.MingAdapter); the vision embeds (2048, unit
norm) are stored raw for the adapter's vis_in / vis_q paths, exactly what
the engine's pig_ming_vision GGUF produces.

Placement (precompute.teacher_mode): gpu = the thinker resident over the CUDA
devices (needs ~36 GiB in total), offload = weights in RAM streamed layer by
layer through the GPU (accelerate cpu_offload; RAM must hold ~34 GB), cpu =
no CUDA.  The vision tower, connector and projections sit on the device
when the budget allows.

Written against the upstream modeling code (modeling_bailing_moe_v2.py,
modeling_bailingmm2.py, MIT) rather than importing it: that code pins
transformers 4.57 and its own attention / cache classes.  Every equation
here is that file's, forward only.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import time
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen3vl_teacher import patchify

# ---------------------------------------------------------------- constants

SYSTEM_PREFIX = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|><role>HUMAN</role>"
SUFFIX = "<|role_end|><role>ASSISTANT</role>"
IMAGE_START_ID = 157158    # <image>
IMAGE_PATCH_ID = 157157    # <imagePatch>
IMAGE_END_ID = 157159      # </image>
LING_VOCAB = 157184
NUM_QUERIES = 256
HIDDEN = 2048
CAP_DIM = 2560
DIRECT_DIM = 3840
VIS_DIM = 2048             # linear_proj output (== thinker hidden)
DIRECT_LAYERS = (5, 12, 20)

# processing: loader smart_resize (factor 28, 4..1024 patches of 28x28), then the
# processor's fixed-area one (min = max = 451584 px); CLIP mean/std
VISION_FACTOR = 28
VISION_PIXELS = 451584
VISION_LOADER_MIN = 4 * 28 * 28
VISION_LOADER_MAX = 1024 * 28 * 28
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
PATCH, MERGE, TEMPORAL = 14, 2, 2


# ---------------------------------------------------------------- images

def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> Tuple[int, int]:
    """bailingmm_utils.smart_resize (Python round = half to even)."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return max(factor, h_bar), max(factor, w_bar)


def vision_input_size(height: int, width: int) -> Tuple[int, int]:
    """-> (h_bar, w_bar): the loader's smart_resize followed by the processor's
    fixed-area one (ggk MingImageTE::vision_input_size)."""
    h1, w1 = smart_resize(height, width, VISION_FACTOR, VISION_LOADER_MIN, VISION_LOADER_MAX)
    return smart_resize(h1, w1, VISION_FACTOR, VISION_PIXELS, VISION_PIXELS)


def n_vision_tokens(height: int, width: int) -> Tuple[int, int, int]:
    """-> (n_tokens, grid_h, grid_w) in merged (LLM) tokens for an image file size."""
    h_bar, w_bar = vision_input_size(height, width)
    gh, gw = h_bar // VISION_FACTOR, w_bar // VISION_FACTOR
    return gh * gw, gh, gw


def preprocess_vision_image(path) -> Tuple[np.ndarray, Tuple[int, int]]:
    """One bicubic (antialiased) resize straight to the final size, then the
    CLIP normalization -> ([H, W, 3] float32, (grid_h, grid_w)).  The engine
    resamples once as well (sd::ops::interpolate bicubic, antialias); the
    reference resizes twice, which differs by a fraction of a pixel."""
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        h_bar, w_bar = vision_input_size(h, w)
        rgb = rgb.resize((w_bar, h_bar), Image.BICUBIC)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return arr.astype(np.float32), (h_bar // VISION_FACTOR, w_bar // VISION_FACTOR)


def image_token_count(path) -> int:
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    return n_vision_tokens(h, w)[0]


# ---------------------------------------------------------------- tokens

def build_prompt_ids(tok, text: str, n_image: int) -> Tuple[List[int], int]:
    """-> (Ling token ids of the prompt, index of the first <imagePatch> or -1).
    Segments are tokenized separately; the boundaries are special tokens, so
    the ids equal the whole-string tokenization (checked once by
    `check_prompt_tokenization`)."""
    ids = list(tok(SYSTEM_PREFIX, add_special_tokens=False)["input_ids"])
    image_start = -1
    body = text or ""
    if n_image > 0:
        image_start = len(ids) + 1
        ids += [IMAGE_START_ID] + [IMAGE_PATCH_ID] * n_image + [IMAGE_END_ID]
        body = "\n" + body          # _expand_image_tokens: "</image>\n" precedes the text
    ids += list(tok(body + SUFFIX, add_special_tokens=False)["input_ids"])
    return ids, image_start


def check_prompt_tokenization(tok, log=print) -> None:
    """The segment-wise ids must equal the reference's whole-string ones."""
    for text, n in (("a sheep in sunglasses", 0), ("make it night", 4), ("", 0), ("Add snow, heavy\nLayer 1: x", 6)):
        a, _ = build_prompt_ids(tok, text, n)
        s = SYSTEM_PREFIX + ("<image>" + "<imagePatch>" * n + "</image>\n" if n else "") + text + SUFFIX
        b = list(tok(s, add_special_tokens=False)["input_ids"])
        if a != b:
            raise RuntimeError(f"Ling tokenization differs between segments and the whole template for {text!r}")
    for name, tid in (("<image>", IMAGE_START_ID), ("<imagePatch>", IMAGE_PATCH_ID), ("</image>", IMAGE_END_ID)):
        got = tok.convert_tokens_to_ids(name)
        if got != tid:
            raise RuntimeError(f"Ling tokenizer maps {name} to {got}, expected {tid}")
    log("Ling template tokenization checked (segments == whole string, image token ids)")


def load_ling_tokenizer(mllm_dir: pathlib.Path):
    """mllm/tokenizer.json (+ tokenizer_config.json) -> PreTrainedTokenizerFast.
    The BailingTokenizer class lives only in the GitHub repo; the fast
    tokenizer file is self-contained (added tokens included)."""
    from transformers import PreTrainedTokenizerFast

    mllm_dir = pathlib.Path(mllm_dir)
    tok = PreTrainedTokenizerFast(tokenizer_file=str(mllm_dir / "tokenizer.json"), eos_token="<|role_end|>",
                                  pad_token="<|role_end|>", bos_token="<|startoftext|>")
    return tok


def video_rope_positions(length: int, blocks: Sequence[Tuple[int, int, int]]) -> torch.Tensor:
    """get_t_scale_rope_index for one sample -> [3, length] (t, h, w).  Text
    advances all three axes; an image block (start, grid_h, grid_w) takes one
    temporal index t = the position after the preceding text, h = t + row -
    (gh-1)//2, w = t + col - (gw-1)//2, and the text after resumes at t + 1
    (max of the temporal row + 1).  grid_t is 1, so scale_factor is moot."""
    pos = torch.zeros(3, length, dtype=torch.long)
    st_idx, cur = 0, 0
    for start, gh, gw in blocks:
        n = gh * gw
        text_len = start - cur
        pos[:, cur:start] = torch.arange(text_len)[None] + st_idx
        t = text_len + st_idx
        rows = torch.arange(gh).repeat_interleave(gw)
        cols = torch.arange(gw).repeat(gh)
        pos[0, start: start + n] = t
        pos[1, start: start + n] = t + rows - (gh - 1) // 2
        pos[2, start: start + n] = t + cols - (gw - 1) // 2
        st_idx = t + 1
        cur = start + n
    pos[:, cur:] = torch.arange(length - cur)[None] + st_idx
    return pos


# ---------------------------------------------------------------- Bailing MoE (forward only)

class BailingRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(dt)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def video_rope_cos_sin(pos: torch.Tensor, rope_dim: int, theta: float, dtype: torch.dtype):
    """pos [3, B, L] -> cos, sin [B, L, rope_dim] with the video-rope section
    layout (apply_3d_rotary_pos_emb, rope_type video_rope, mrope_section
    [8, 12, 12]): of the rope_dim/2 frequencies, j < 24 alternate height /
    width (even / odd), j >= 24 carry the temporal index; the second half
    repeats the first (rotate_half pairs)."""
    half = rope_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=pos.device) / rope_dim))
    freqs = pos.float()[..., None] * inv_freq[None, None, None, :]         # [3, B, L, half]
    axis = torch.full((half,), 0, dtype=torch.long, device=pos.device)     # t
    spatial = torch.arange(half, device=pos.device) < 24
    axis[spatial] = torch.where(torch.arange(half, device=pos.device)[spatial] % 2 == 0, 1, 2)
    sel = freqs.permute(1, 2, 3, 0)                                          # [B, L, half, 3]
    sel = sel.gather(-1, axis[None, None, :, None].expand(sel.shape[0], sel.shape[1], half, 1)).squeeze(-1)
    emb = torch.cat([sel, sel], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class BailingAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, rope_dim: int, eps: float):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.rope_dim = heads, kv_heads, head_dim, rope_dim
        self.query_key_value = nn.Linear(hidden, (heads + 2 * kv_heads) * head_dim, bias=False)
        self.dense = nn.Linear(heads * head_dim, hidden, bias=False)
        self.q_norm = BailingRMSNorm(head_dim, eps)
        self.k_norm = BailingRMSNorm(head_dim, eps)

    def forward(self, x, cos, sin, mask):
        B, L, _ = x.shape
        qkv = self.query_key_value(x).view(B, L, self.heads + 2 * self.kv_heads, self.head_dim)
        q, k, v = qkv.split([self.heads, self.kv_heads, self.kv_heads], dim=2)
        q = self.q_norm(q.transpose(1, 2))
        k = self.k_norm(k.transpose(1, 2))
        v = v.transpose(1, 2)
        c, s = cos[:, None], sin[:, None]
        rd = self.rope_dim
        q = torch.cat([q[..., :rd] * c + rotate_half(q[..., :rd]) * s, q[..., rd:]], dim=-1)
        k = torch.cat([k[..., :rd] * c + rotate_half(k[..., :rd]) * s, k[..., rd:]], dim=-1)
        rep = self.heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        w = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(2, 3))
        w = w + mask
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(w, v).transpose(1, 2).reshape(B, L, -1)
        return self.dense(o)


class BailingMLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class BailingGate(nn.Module):
    def __init__(self, hidden: int, n_experts: int, top_k: int, n_group: int, topk_group: int, scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_experts, hidden))
        self.expert_bias = nn.Parameter(torch.zeros(n_experts), requires_grad=False)
        self.top_k, self.n_group, self.topk_group, self.scale = top_k, n_group, topk_group, scale

    def forward(self, x):
        """x [T, hidden] -> (topk_idx [T, k], topk_weight [T, k] float32)."""
        logits = F.linear(x.float(), self.weight.float())
        scores = torch.sigmoid(logits)
        routing = scores + self.expert_bias.float()
        T, E = routing.shape
        group_scores = routing.view(T, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = group_mask[:, :, None].expand(T, self.n_group, E // self.n_group).reshape(T, E)
        masked = routing.masked_fill(~score_mask.bool(), float("-inf"))
        topk_idx = torch.topk(masked, k=self.top_k, dim=-1, sorted=False)[1]
        w = torch.gather(scores, 1, topk_idx)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else w
        return topk_idx, w * self.scale


class BailingMoE(nn.Module):
    def __init__(self, hidden: int, moe_inter: int, n_experts: int, top_k: int, n_group: int, topk_group: int,
                 scale: float, n_shared: int):
        super().__init__()
        self.experts = nn.ModuleList(BailingMLP(hidden, moe_inter) for _ in range(n_experts))
        self.gate = BailingGate(hidden, n_experts, top_k, n_group, topk_group, scale)
        self.image_gate = BailingGate(hidden, n_experts, top_k, n_group, topk_group, scale)
        self.shared_experts = BailingMLP(hidden, moe_inter * n_shared) if n_shared > 0 else None
        self.top_k = top_k

    def forward(self, x, image_mask):
        B, L, H = x.shape
        flat = x.reshape(-1, H)
        idx, w = self.gate(flat)
        if image_mask is not None:
            i_idx, i_w = self.image_gate(flat)
            m = image_mask.reshape(-1, 1)
            idx = torch.where(m, i_idx, idx)
            w = torch.where(m, i_w, w)
        out = torch.zeros_like(flat, dtype=torch.float32)
        for e in torch.unique(idx).tolist():
            rows, slots = (idx == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            y = self.experts[e](flat[rows]).float() * w[rows, slots][:, None]
            out.index_add_(0, rows, y)
        y = out.to(x.dtype).view(B, L, H)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


class BailingLayer(nn.Module):
    def __init__(self, cfg: dict, layer_idx: int):
        super().__init__()
        hidden = cfg["hidden_size"]
        self.attention = BailingAttention(hidden, cfg["num_attention_heads"], cfg["num_key_value_heads"],
                                          cfg["head_dim"], int(cfg["head_dim"] * cfg["partial_rotary_factor"]),
                                          cfg["rms_norm_eps"])
        self.input_layernorm = BailingRMSNorm(hidden, cfg["rms_norm_eps"])
        self.post_attention_layernorm = BailingRMSNorm(hidden, cfg["rms_norm_eps"])
        self.dense = layer_idx < cfg["first_k_dense_replace"]
        if self.dense:
            self.mlp = BailingMLP(hidden, cfg["intermediate_size"])
        else:
            self.mlp = BailingMoE(hidden, cfg["moe_intermediate_size"], cfg["num_experts"], cfg["num_experts_per_tok"],
                                  cfg["n_group"], cfg["topk_group"], cfg["routed_scaling_factor"],
                                  cfg.get("num_shared_experts") or 0)

    def forward(self, x, cos, sin, mask, image_mask):
        x = x + self.attention(self.input_layernorm(x), cos, sin, mask)
        h = self.post_attention_layernorm(x)
        h = self.mlp(h) if self.dense else self.mlp(h, image_mask)
        return x + h


class BailingModel(nn.Module):
    """BailingMoeV2Model (checkpoint names model.model.*)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.word_embeddings = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList(BailingLayer(cfg, i) for i in range(cfg["num_hidden_layers"]))
        self.norm = BailingRMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

    def forward(self, embeds, cos, sin, mask, image_mask, capture: Sequence[int]):
        """-> (final normed states, {index: hidden_states[index]}) with HF numbering
        (index i = the input of layer i; num_layers = the final normed output)."""
        x = embeds
        out: Dict[int, torch.Tensor] = {}
        for i, layer in enumerate(self.layers):
            if i in capture:
                out[i] = x
            x = layer(x, cos, sin, mask, image_mask)
        x = self.norm(x)
        if len(self.layers) in capture:
            out[len(self.layers)] = x
        return x, out


class BailingThinker(nn.Module):
    """BailingMoeV2ForCausalLM minus lm_head (names model.model.*)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.model = BailingModel(cfg)


class MingMLLM(nn.Module):
    """BailingMM2NativeForConditionalGeneration's inference modules with the
    checkpoint's parameter names: vision.*, model.model.*, linear_proj.*."""

    def __init__(self, cfg: dict, build_vision: bool = True):
        super().__init__()
        llm = cfg["llm_config"]
        vc = cfg["vision_config"]
        self.vision = build_vision_tower(vc) if build_vision else None
        self.model = BailingThinker(llm)
        self.linear_proj = nn.Sequential(nn.Linear(vc["out_hidden_size"], llm["hidden_size"]), nn.GELU(),
                                         nn.Linear(llm["hidden_size"], llm["hidden_size"]))


def build_vision_tower(vc: dict):
    """The Qwen2.5-VL tower (qwen2_5_vit.py is transformers' module with a
    configurable out_hidden_size); state-dict names match vision.* verbatim."""
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

    cfg = Qwen2_5_VLVisionConfig(depth=vc["depth"], hidden_size=vc["hidden_size"], hidden_act=vc.get("hidden_act", "silu"),
                                 intermediate_size=vc["intermediate_size"], num_heads=vc["num_heads"],
                                 in_channels=vc.get("in_channels", 3), patch_size=vc["patch_size"],
                                 spatial_merge_size=vc["spatial_merge_size"], temporal_patch_size=vc["temporal_patch_size"],
                                 window_size=vc["window_size"], out_hidden_size=vc["out_hidden_size"],
                                 fullatt_block_indexes=list(vc["fullatt_block_indexes"]))
    cfg._attn_implementation = "sdpa"
    return Qwen2_5_VisionTransformerPretrainedModel(cfg)


# ---------------------------------------------------------------- Qwen2 connector (forward only)

class Qwen2RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(dt)


class Qwen2Attention(nn.Module):
    def __init__(self, hidden, heads, kv_heads):
        super().__init__()
        self.heads, self.kv_heads = heads, kv_heads
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden, kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(heads * self.head_dim, hidden, bias=False)

    def forward(self, x, cos, sin):
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.kv_heads, self.head_dim).transpose(1, 2)
        c, s = cos[None, None], sin[None, None]
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        rep = self.heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False)     # bidirectional (is_causal patched off)
        return self.o_proj(o.transpose(1, 2).reshape(B, L, -1))


class Qwen2Layer(nn.Module):
    def __init__(self, hidden, heads, kv_heads, inter, eps):
        super().__init__()
        self.self_attn = Qwen2Attention(hidden, heads, kv_heads)
        self.mlp = BailingMLP(hidden, inter)          # gate/up/down SiLU, same names
        self.input_layernorm = Qwen2RMSNorm(hidden, eps)
        self.post_attention_layernorm = Qwen2RMSNorm(hidden, eps)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Connector(nn.Module):
    """Qwen2ForCausalLM.model without embed_tokens (checkpoint names model.layers.*, model.norm)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(Qwen2Layer(cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"],
                                               cfg["intermediate_size"], cfg["rms_norm_eps"]) for _ in range(cfg["num_hidden_layers"]))
        self.norm = Qwen2RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

    def forward(self, x):
        B, L, _ = x.shape
        head_dim = self.cfg["hidden_size"] // self.cfg["num_attention_heads"]
        inv = 1.0 / (self.cfg["rope_theta"] ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) / head_dim))
        freqs = torch.arange(L, dtype=torch.float32, device=x.device)[:, None] * inv[None]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class DiffusersRMSNorm(nn.Module):
    """diffusers.models.normalization.RMSNorm (proj_directvlm.0)."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        h = x * torch.rsqrt(var + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            h = h.to(self.weight.dtype)
        return h * self.weight


# ---------------------------------------------------------------- loading helpers

def _safetensors_files(root: pathlib.Path) -> List[pathlib.Path]:
    idx = root / "model.safetensors.index.json"
    if idx.is_file():
        wm = json.loads(idx.read_text())["weight_map"]
        return [root / f for f in sorted(set(wm.values()))]
    return sorted(root.glob("*.safetensors"))


def load_named_tensors(root: pathlib.Path, wanted, dtype=None, map_name=lambda n: n) -> Dict[str, torch.Tensor]:
    """Load the tensors `wanted(name)` accepts from a safetensors dir."""
    from safetensors import safe_open

    out = {}
    for fp in _safetensors_files(root):
        with safe_open(str(fp), framework="pt") as sf:
            for k in sf.keys():
                if wanted(k):
                    t = sf.get_tensor(k)
                    out[map_name(k)] = t.to(dtype) if dtype is not None else t
    return out


def param_bytes(root: pathlib.Path, include=lambda n: True) -> int:
    import struct

    total = 0
    for fp in _safetensors_files(root):
        with open(fp, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__" or not include(name):
                continue
            a, b = meta["data_offsets"]
            total += b - a
    return total


# ---------------------------------------------------------------- teacher

class MingTeacher:
    """prompt (+ one input image) -> caption rows [256, 2560] and direct rows
    [n_prompt, 3840], both bf16 on the CPU; vision embeds [n, 2048]."""

    def __init__(self, root: pathlib.Path, device: torch.device, gpu_mem_gib: float, teacher_mode: str = "auto",
                 log=print, dtype: torch.dtype = torch.bfloat16):
        root = pathlib.Path(root)
        self.root = root
        self.dev = device
        self.dtype = dtype
        self.log = log
        t0 = time.time()
        mllm, connector, mlp = root / "mllm", root / "connector", root / "mlp"
        for d in (mllm, connector, mlp):
            if not d.is_dir():
                raise RuntimeError(f"teacher: {d} missing (Setup -> Materials)")
        self.cfg = json.loads((mllm / "config.json").read_text())
        llm = self.cfg["llm_config"]
        self.tok = load_ling_tokenizer(mllm)
        check_prompt_tokenization(self.tok, log)
        mlp_cfg = json.loads((mlp / "config.json").read_text())
        self.direct_layers = tuple(mlp_cfg.get("selected_hidden_states_layers") or DIRECT_LAYERS)
        self.connector_norm = bool(mlp_cfg.get("connector_norm", False))
        self.cap_dim = int(mlp_cfg.get("diffusion_c_input_dim", CAP_DIM))
        self.out_dim = int(mlp_cfg.get("diffusion_inner_dim", DIRECT_DIM))
        self.hidden = int(llm["hidden_size"])
        self.vis_dim = self.hidden

        thinker_gib = param_bytes(mllm, lambda n: n.startswith("model.model.")) / 2**30
        vision_gib = param_bytes(mllm, lambda n: n.startswith("vision.") or n.startswith("linear_proj.")) / 2**30
        conn_gib = param_bytes(connector, lambda n: not n.endswith("embed_tokens.weight")) / 2**30 / 2   # f32 on disk -> bf16
        if teacher_mode == "auto":
            budgets = self._cuda_budgets(gpu_mem_gib) if device.type == "cuda" else {}
            mode = "gpu" if sum(budgets.values()) >= thinker_gib + vision_gib + conn_gib + 3.0 else "offload"
        else:
            mode = teacher_mode
        if device.type != "cuda":
            mode = "cpu"
        self.mode = mode
        log(f"teacher {root.name}: thinker {thinker_gib:.1f} GiB, vision {vision_gib:.1f} GiB, connector {conn_gib:.1f} GiB "
            f"(bf16), GPU budget {gpu_mem_gib:.1f} GiB -> {mode}")

        # -- MLLM (vision tower + thinker + linear_proj), names == checkpoint --
        from accelerate import init_empty_weights

        with init_empty_weights():
            self.mllm = MingMLLM(self.cfg)
        self._place_mllm(mllm, mode, gpu_mem_gib, log)
        self.mllm.eval().requires_grad_(False)

        # -- connector + projections (small): device when the budget allows --
        small_dev = device if mode in ("gpu", "offload") and gpu_mem_gib >= vision_gib + conn_gib + 2.0 else torch.device("cpu")
        if mode == "cpu":
            small_dev = torch.device("cpu")
        self.small_dev = small_dev
        ccfg = json.loads((connector / "config.json").read_text())
        self.connector = Qwen2Connector(ccfg)
        sd = load_named_tensors(connector, lambda n: n.startswith("model.") and "embed_tokens" not in n, dtype,
                                map_name=lambda n: n[len("model."):])
        missing, unexpected = self.connector.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"connector weights do not match: missing {missing[:3]} unexpected {unexpected[:3]}")
        self.connector.to(small_dev, dtype).eval().requires_grad_(False)      # the shards are f32, run it in bf16 like the reference
        st = load_named_tensors(mlp, lambda n: True)
        self.query_tokens = st["query_tokens_dict.16x16"].to(small_dev, dtype)          # [256, hidden]
        self.proj_in = nn.Linear(self.hidden, ccfg["hidden_size"]).to(small_dev, dtype)
        self.proj_in.load_state_dict({"weight": st["proj_in.weight"], "bias": st["proj_in.bias"]})
        self.proj_out = nn.Linear(ccfg["hidden_size"], self.cap_dim).to(small_dev, dtype)
        self.proj_out.load_state_dict({"weight": st["proj_out.weight"], "bias": st["proj_out.bias"]})
        n_direct = len(self.direct_layers)
        self.proj_directvlm = nn.Sequential(DiffusersRMSNorm(self.hidden * n_direct, 1e-5),
                                            nn.Linear(self.hidden * n_direct, self.out_dim)).to(small_dev, dtype)
        self.proj_directvlm.load_state_dict({"0.weight": st["proj_directvlm.0.weight"], "1.weight": st["proj_directvlm.1.weight"],
                                             "1.bias": st["proj_directvlm.1.bias"]})
        for m in (self.proj_in, self.proj_out, self.proj_directvlm):
            m.eval().requires_grad_(False)
        self.num_queries = int(self.query_tokens.shape[0])
        self.llm_cfg = llm
        log(f"teacher up in {time.time() - t0:.0f}s: {llm['num_hidden_layers']} thinker layers, {ccfg['num_hidden_layers']} connector "
            f"layers, direct layers {list(self.direct_layers)}, {self.num_queries} queries, connector_norm {self.connector_norm}")

    # -- placement --
    def _cuda_budgets(self, main_gib: float) -> Dict[int, float]:
        """GiB per CUDA device: the project's budget on the chosen device, the
        rest of each other card minus headroom."""
        main = self.dev.index if self.dev.index is not None else torch.cuda.current_device()
        out = {}
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory / 2**30
            out[i] = main_gib if i == main else max(0.0, total - 1.5)
        return out

    def _place_mllm(self, mllm_dir: pathlib.Path, mode: str, gpu_mem_gib: float, log):
        from accelerate import cpu_offload, infer_auto_device_map, load_checkpoint_and_dispatch
        from accelerate.utils import load_checkpoint_in_model

        m = self.mllm
        if mode == "gpu":
            max_memory = {i: f"{g:.1f}GiB" for i, g in self._cuda_budgets(gpu_mem_gib).items()}
            dm = infer_auto_device_map(m, max_memory=max_memory, no_split_module_classes=["BailingLayer"], dtype=self.dtype)
            if any(str(d) in ("cpu", "disk") for d in dm.values()):
                log("teacher: the thinker does not fit the CUDA devices, streaming it from RAM instead")
                mode = "offload"
                self.mode = mode
            else:
                load_checkpoint_and_dispatch(m, str(mllm_dir), device_map=dm, dtype=self.dtype,
                                             no_split_module_classes=["BailingLayer"], strict=False)
                counts: Dict[str, int] = {}
                for v in dm.values():
                    counts[str(v)] = counts.get(str(v), 0) + 1
                log(f"teacher: device map {counts}")
                self.input_dev = torch.device(self.dev)
                return
        load_checkpoint_in_model(m, str(mllm_dir), device_map={"": "cpu"}, dtype=self.dtype, strict=False)
        if mode == "offload":
            # thinker layers streamed through the GPU one decoder layer at a time;
            # the tower + linear_proj resident when the budget has room for them
            cpu_offload(m.model, execution_device=torch.device(self.dev), preload_module_classes=["BailingLayer"])
            vision_gib = sum(p.numel() for p in m.vision.parameters()) * 2 / 2**30 + 0.1
            if gpu_mem_gib >= vision_gib + 4.0:
                m.vision.to(self.dev)
                m.linear_proj.to(self.dev)
            else:
                log(f"teacher: vision tower stays on the CPU (GPU budget {gpu_mem_gib:.1f} GiB)")
            self.input_dev = torch.device(self.dev)
        else:
            torch.set_num_threads(os.cpu_count() or 4)
            self.input_dev = torch.device("cpu")

    # -- API used by precompute --
    def prompt_ids(self, text: str, n_image: int) -> Tuple[List[int], int]:
        return build_prompt_ids(self.tok, text, n_image)

    def embed_table(self) -> torch.Tensor:
        """The thinker's token embedding table [vocab, hidden] float32 (cpu)."""
        w = self.mllm.model.model.word_embeddings.weight
        if w.device.type == "meta":     # offloaded: read it from the safetensors instead
            return load_named_tensors(self.root / "mllm", lambda n: n == "model.model.word_embeddings.weight")[
                "model.model.word_embeddings.weight"].float()
        return w.detach().float().cpu()

    @torch.no_grad()
    def encode_images(self, imgs: Sequence[np.ndarray]) -> List[torch.Tensor]:
        """CLIP-normalized [H, W, 3] arrays -> list of [n_tokens, 2048] bf16 cpu
        (linear_proj + F.normalize, i.e. what is spliced into the thinker)."""
        if not len(imgs):
            return []
        packs, grids = [], []
        for im in imgs:
            p, gh, gw = patchify(im, PATCH, MERGE, TEMPORAL)
            packs.append(p)
            grids.append((1, gh, gw))
        vdev = next(self.mllm.vision.parameters()).device
        pixel = torch.cat(packs).to(vdev, self.dtype)
        grid = torch.tensor(grids, dtype=torch.long, device=vdev)
        out = self.mllm.vision(hidden_states=pixel, grid_thw=grid)
        # transformers returns the pre-merger patch states as last_hidden_state and
        # the merged tokens (out_hidden_size) as pooler_output
        feats = getattr(out, "pooler_output", None)
        if feats is None:
            feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)
        assert feats.shape[-1] == self.mllm.linear_proj[0].in_features, (feats.shape, self.mllm.linear_proj[0].in_features)
        feats = self.mllm.linear_proj(feats.to(self.dtype))
        feats = F.normalize(feats, dim=-1)
        outs, o = [], 0
        for (_, gh, gw) in grids:
            n = (gh // MERGE) * (gw // MERGE)
            outs.append(feats[o: o + n].to(torch.bfloat16).cpu())
            o += n
        assert o == feats.shape[0], (o, feats.shape)
        return outs

    @torch.no_grad()
    def __call__(self, batch: List[dict]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """batch: dicts {ids: prompt ids, image_start: int, grid: (gh, gw) | None,
        vis_embeds: [n, 2048] | None} -> (caption [B, 256, cap_dim] bf16 cpu,
        [direct [n_prompt_i, out_dim] bf16 cpu, ...])."""
        nq = self.num_queries
        lens = [len(s["ids"]) for s in batch]
        totals = [n + nq + 2 for n in lens]
        B, L = len(batch), max(totals)
        emb_w = self.mllm.model.model.word_embeddings
        edev = emb_w.weight.device if emb_w.weight.device.type != "meta" else self.input_dev
        ids = torch.full((B, L), self.tok.pad_token_id, dtype=torch.long)
        image_mask = torch.zeros(B, L, dtype=torch.bool)
        pos = torch.ones(3, B, L, dtype=torch.long)
        splices: List[Tuple[int, int, torch.Tensor]] = []
        for j, s in enumerate(batch):
            n = lens[j]
            row = list(s["ids"]) + [IMAGE_START_ID] + [IMAGE_PATCH_ID] * nq + [IMAGE_END_ID]
            ids[j, : len(row)] = torch.tensor(row)
            blocks = []
            vv = s.get("vis_embeds")
            if vv is not None:
                gh, gw = s["grid"]
                assert vv.shape[0] == gh * gw, (vv.shape, s["grid"])
                blocks.append((int(s["image_start"]), gh, gw))
                splices.append((j, int(s["image_start"]), vv))
            blocks.append((n + 1, 1, nq))
            pos[:, j, : len(row)] = video_rope_positions(len(row), blocks)
            image_mask[j, : len(row)] = torch.tensor(row) == IMAGE_PATCH_ID
        # embeddings with the image features / query tokens spliced over their patch runs
        embeds = emb_w(ids.to(edev)).to(self.dtype)
        q = self.query_tokens.to(embeds.device, self.dtype)
        for j in range(B):
            embeds[j, lens[j] + 1: lens[j] + 1 + nq] = q
        for j, st, vv in splices:
            embeds[j, st: st + vv.shape[0]] = vv.to(embeds.device, self.dtype)
        # causal + right-padding mask, additive
        minv = torch.finfo(self.dtype).min
        keep = torch.arange(L)[None, :] < torch.tensor(totals)[:, None]
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
        allowed = causal[None] & keep[:, None, :]
        mask = torch.zeros(B, 1, L, L, dtype=self.dtype).masked_fill(~allowed[:, None], minv)
        llm = self.llm_cfg
        rope_dim = int(llm["head_dim"] * llm["partial_rotary_factor"])
        cos, sin = video_rope_cos_sin(pos, rope_dim, float(llm["rope_theta"]), self.dtype)
        capture = set(self.direct_layers)
        final, taps = self.mllm.model.model(embeds, cos.to(embeds.device), sin.to(embeds.device),
                                            mask.to(embeds.device), image_mask.to(embeds.device), capture)
        # direct condition: the prompt tokens' selected hidden states, concatenated
        sdev = self.small_dev
        direct_out = []
        for j in range(B):
            n = lens[j]
            cat = torch.cat([taps[i][j, :n].to(sdev, self.dtype) for i in self.direct_layers], dim=-1)
            direct_out.append(self.proj_directvlm(cat).to(torch.bfloat16).cpu())
        # caption: connector over the final states at the query positions
        gen = torch.stack([final[j, lens[j] + 1: lens[j] + 1 + nq] for j in range(B)]).to(sdev, self.dtype)
        cap = self.proj_out(self.connector(self.proj_in(gen)))
        if self.connector_norm:
            cap = F.normalize(cap, dim=-1)
        return cap.to(torch.bfloat16).cpu(), direct_out


# ---------------------------------------------------------------- mock

class MockMingTeacher:
    """Developer stand-in (GGUF_TRAINER_MOCK_TEACHER=1): the real Ling
    tokenizer when mllm/tokenizer.json is on disk (so the exported GGUF can
    run in the engine), else a hashed word tokenizer over the same template
    ids; synthetic but deterministic targets with the real shapes."""

    def __init__(self, root: Optional[pathlib.Path], log=print):
        g = torch.Generator().manual_seed(4321)
        self.mode = "mock"
        self.num_queries, self.cap_dim, self.out_dim, self.vis_dim, self.hidden = NUM_QUERIES, CAP_DIM, DIRECT_DIM, VIS_DIM, HIDDEN
        self.tok = None
        tj = pathlib.Path(root) / "mllm" / "tokenizer.json" if root else None
        if tj is not None and tj.is_file():
            self.tok = load_ling_tokenizer(tj.parent)
            check_prompt_tokenization(self.tok, log)
            log(f"MOCK teacher: using the real Ling tokenizer from {tj}")
        self.tok_basis = torch.randn(64, self.out_dim, generator=g) / 4
        self.cap_basis = torch.randn(64, self.cap_dim, generator=g) / 4
        self.pix_basis = torch.randn(48, self.vis_dim, generator=g) / 4
        self.vis_mix = torch.randn(self.vis_dim, self.out_dim, generator=g) / self.vis_dim ** 0.5
        log("MOCK teacher in use: targets are synthetic, the adapter will be meaningless")

    def _hash_ids(self, text: str) -> List[int]:
        return [1000 + (zlib.crc32(w.encode("utf-8")) % 150000) for w in text.split()]

    def prompt_ids(self, text: str, n_image: int) -> Tuple[List[int], int]:
        if self.tok is not None:
            return build_prompt_ids(self.tok, text, n_image)
        ids = [157151, 6, 157152, 200, 156895, 157151, 7, 157152]       # a fixed stand-in prefix
        start = -1
        if n_image > 0:
            start = len(ids) + 1
            ids += [IMAGE_START_ID] + [IMAGE_PATCH_ID] * n_image + [IMAGE_END_ID]
        ids += self._hash_ids(text) + [156895, 157151, 8, 157152]
        return ids, start

    def embed_table(self):
        return None

    def encode_images(self, imgs):
        outs = []
        for im in imgs:
            h, w = im.shape[:2]
            gh, gw = h // VISION_FACTOR, w // VISION_FACTOR
            cells = im.reshape(gh, VISION_FACTOR, gw, VISION_FACTOR, 3).mean(axis=(1, 3)).reshape(gh * gw, 3)
            feats = torch.zeros(gh * gw, 48)
            f = torch.from_numpy(cells.astype(np.float32))
            feats[:, :3] = f
            feats[:, 3:6] = f * f
            posn = torch.arange(gh * gw, dtype=torch.float32)[:, None]
            feats[:, 6:48] = torch.sin(posn * torch.arange(1, 43, dtype=torch.float32)[None, :] / 7.0)
            v = F.normalize(feats @ self.pix_basis, dim=-1)
            outs.append(v.to(torch.bfloat16))
        return outs

    def __call__(self, batch):
        B = len(batch)
        cap = torch.zeros(B, self.num_queries, self.cap_dim)
        direct = []
        for j, s in enumerate(batch):
            ids = torch.as_tensor(s["ids"], dtype=torch.long)
            feats = torch.zeros(len(ids), 64)
            feats[torch.arange(len(ids)), ids % 64] = 1.0
            feats[:, 0] += torch.arange(len(ids), dtype=torch.float32) / 100.0
            h = feats @ self.tok_basis
            vv = s.get("vis_embeds")
            if vv is not None:
                st = int(s["image_start"])
                h[st: st + vv.shape[0]] += vv.float() @ self.vis_mix
            direct.append(h.to(torch.bfloat16))
            summary = torch.zeros(64)
            summary[ids % 64] = 1.0
            c = (summary[None, :] * torch.arange(1, self.num_queries + 1, dtype=torch.float32)[:, None] / 64.0) @ self.cap_basis
            if vv is not None:
                c = c + (vv.float().mean(0) @ self.vis_mix)[: self.cap_dim][None, :] * 0.1
            cap[j] = c
        return cap.to(torch.bfloat16), direct
