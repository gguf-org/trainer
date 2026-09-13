"""The student: pig_clip as a torch model, loaded straight from its GGUF.

pig_clip (trainer7) is a native train/fine-tune, not a stock Qwen3 checkpoint,
but its GGUFs carry the plain Qwen3 tensor layout — HF tensor
names under "model.", 1-D/norm f32, 2-D f16 — so a Qwen3Model built from
the base config can take the weights directly.  Quantized variants are
dequantized with gguf-connector's vendored gguf-py quant code; the adapter is best trained against the f16 file
(what trainer8 did), and ggk's LLMEmbedder emits the same final-norm hidden
states the loader returns here (out_layers={}).

Tokenization is the engine's: Qwen BPE over the formatted string with no
special tokens (the <role>/<IMAGE1> markers are plain text there).
"""

from __future__ import annotations

import pathlib
from typing import List, Tuple

import numpy as np
import torch


def gguf_state_dict(path: pathlib.Path) -> dict:
    from gguf_connector.reader import GGUFReader
    from gguf_connector.const import GGMLQuantizationType as T

    try:
        from gguf_connector.quant import dequantize
    except ImportError:  # gguf-connector without the quant module
        dequantize = None
    r = GGUFReader(str(path))
    sd = {}
    for t in r.tensors:
        name = t.name[len("model."):] if t.name.startswith("model.") else t.name
        shape = tuple(int(x) for x in reversed(t.shape))
        if t.tensor_type in (T.F32, T.F16):
            a = np.asarray(t.data).astype(np.float32).reshape(shape)
        elif t.tensor_type == T.BF16:
            a = np.asarray(t.data).view(np.int16).reshape(shape)
            sd[name] = torch.from_numpy(a.copy()).view(torch.bfloat16).float()
            continue
        else:
            if dequantize is None:
                raise RuntimeError(f"{path.name}: {t.name} is {t.tensor_type.name}; gguf-connector cannot dequantize it")
            a = dequantize(np.asarray(t.data), t.tensor_type).astype(np.float32).reshape(shape)
        sd[name] = torch.from_numpy(np.ascontiguousarray(a))
    return sd


def load_student(gguf_path: pathlib.Path, hf_dir: pathlib.Path, device, dtype=None, log=print):
    """-> (Qwen3Model in eval mode, tokenizer)"""
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    cfg = AutoConfig.from_pretrained(str(hf_dir))
    model = AutoModel.from_config(cfg)
    sd = gguf_state_dict(pathlib.Path(gguf_path))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected]
    missing = [k for k in missing if "rotary" not in k and "inv_freq" not in k]
    if missing or unexpected:
        raise RuntimeError(f"student GGUF does not match the Qwen3 config: missing {missing[:5]} "
                           f"unexpected {unexpected[:5]}")
    model.eval().requires_grad_(False)
    if dtype is not None:
        model.to(dtype)
    model.to(device)
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    tok.padding_side = "right"
    log(f"student {pathlib.Path(gguf_path).name}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params "
        f"on {device}, hidden {cfg.hidden_size}")
    return model, tok


def tokenize_student(prompts: List[str], tok, format_prompt, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """-> ids [S, L] int32 right-padded, len [S] int32."""
    enc = tok([format_prompt(p) for p in prompts], padding=True, truncation=True, max_length=max_len,
              add_special_tokens=False, return_tensors="np")
    ids = enc["input_ids"].astype(np.int32)
    lens = enc["attention_mask"].sum(axis=1).astype(np.int32)
    assert (lens > 0).all()
    return ids, lens
