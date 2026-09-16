"""T5-XXL (v1.1) encoder teacher for the PixArt pack — the ./trainer recipe.

The encoder-only bf16 repack (callgg/t5-v1_1-xxl-encoder-bf16: config +
model.safetensors + the sentencepiece model) loads into transformers'
T5EncoderModel.  The target per prompt is `last_hidden_state` over the
window ggk's PixArtT5Embedder builds: sentencepiece ids + EOS, padded with 0
to the caption length (120), key-masked at the pads.  T5 is bidirectional
with a relative position bias, so the real positions do not depend on how
many pads follow them: the teacher runs on the longest real length of the
batch and only the real rows are stored (pads are never supervised — the
DiT never attends there).

Placement (precompute.teacher_mode): gpu = resident (9.5 GB bf16 + activations,
needs ~12 GiB), offload = weights in RAM streamed layer by layer through the
GPU with accelerate's cpu_offload (a 6 GB card), cpu = no CUDA.
"""

from __future__ import annotations

import os
import pathlib
import time
import zlib
from typing import List, Tuple

import numpy as np
import torch

T5_VOCAB = 32128     # the embedding table (32100 sentencepiece pieces + 28 unused rows)
T5_EOS = 1
T5_PAD = 0


def tokenize_t5(tok, prompts: List[str], window: int) -> Tuple[np.ndarray, np.ndarray]:
    """EOS appended, pad(0) to `window` — the exact sequence the engine's
    T5 tokenizer + pad_tokens produce.  -> ids [B, window] int32,
    len [B] int32 (real incl. EOS, >= 1: an empty prompt is [EOS])."""
    enc = tok(list(prompts), padding="max_length", max_length=window, truncation=True, return_tensors="np")
    ids = enc["input_ids"].astype(np.int32)
    lens = np.maximum(enc["attention_mask"].sum(axis=1).astype(np.int32), 1)
    return ids, lens


class T5Teacher:
    """prompts -> (ids [B, window], len [B]) and hidden rows [B, L, 4096] bf16 cpu."""

    def __init__(self, root: pathlib.Path, device: torch.device, gpu_mem_gib: float, window: int,
                 teacher_mode: str = "auto", log=print):
        from transformers import T5EncoderModel, T5TokenizerFast

        root = pathlib.Path(root)
        self.root = root
        self.dev = device
        self.window = window
        self.dtype = torch.bfloat16
        t0 = time.time()
        # spiece.model only (no tokenizer.json): the fast tokenizer is converted
        # at load time, which needs sentencepiece + protobuf installed
        self.tok = T5TokenizerFast.from_pretrained(str(root))
        weights = root / "model.safetensors"
        size_gib = weights.stat().st_size / 2**30 if weights.is_file() else 9.5
        if teacher_mode == "auto":
            mode = "gpu" if (device.type == "cuda" and gpu_mem_gib >= size_gib + 2.5) else "offload"
        else:
            mode = teacher_mode
        if device.type != "cuda":
            mode = "cpu"
        self.mode = mode
        log(f"teacher {root.name}: {size_gib:.1f} GiB bf16, GPU budget {gpu_mem_gib:.1f} GiB -> {mode}")
        m = T5EncoderModel.from_pretrained(str(root), dtype=self.dtype)
        m.eval().requires_grad_(False)
        self.out_dim = int(m.config.d_model)
        self.vocab = int(m.config.vocab_size)
        if mode == "gpu":
            m.to(device)
        elif mode == "offload":
            from accelerate import cpu_offload

            cpu_offload(m, execution_device=torch.device(device))
        else:
            torch.set_num_threads(os.cpu_count() or 4)
        self.m = m
        self.input_dev = torch.device("cpu") if mode == "cpu" else device
        log(f"teacher up in {time.time() - t0:.0f}s ({m.config.num_layers} layers, d_model {self.out_dim}, "
            f"vocab {self.vocab})")

    def tokenize(self, prompts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        return tokenize_t5(self.tok, prompts, self.window)

    def text_len(self, text: str) -> int:
        return int(min(self.window, len(self.tok(text).input_ids)))

    @torch.no_grad()
    def __call__(self, ids: np.ndarray, lens: np.ndarray) -> torch.Tensor:
        """ids [B, window] int32, len [B] -> [B, max(len), out_dim] bf16 cpu
        (rows past each sample's len are meaningless)."""
        L = int(lens.max())
        t_ids = torch.from_numpy(ids[:, :L].astype(np.int64)).to(self.input_dev)
        mask = (torch.arange(L)[None, :] < torch.from_numpy(lens.astype(np.int64))[:, None]).long().to(self.input_dev)
        if self.input_dev.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.m(input_ids=t_ids, attention_mask=mask).last_hidden_state
        else:
            out = self.m(input_ids=t_ids, attention_mask=mask).last_hidden_state
        return out.to(torch.bfloat16).cpu()


class MockT5Teacher:
    """Developer stand-in (GGUF_TRAINER_MOCK_TEACHER=1): a hashed word
    "tokenizer" over the same EOS + pad-0 window and a fixed random projection
    per slot, so the seeded pipeline runs on a laptop without the 9.5 GB
    teacher.  Never use for a real adapter."""

    def __init__(self, out_dim: int, window: int, vocab: int = T5_VOCAB, log=print):
        g = torch.Generator().manual_seed(4321)
        self.out_dim, self.window, self.vocab = out_dim, window, vocab
        self.mode = "mock"
        self.table = torch.randn(vocab, out_dim, generator=g) / 4
        self.pos = torch.randn(window, out_dim, generator=g) / 8
        log("MOCK teacher in use: targets are synthetic, the adapter will be meaningless")

    def _ids(self, text: str) -> List[int]:
        ids = [2 + (zlib.crc32(w.encode('utf-8')) % (self.vocab - 2)) for w in text.split()]
        return ids[: self.window - 1] + [T5_EOS]

    def tokenize(self, prompts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        ids = np.full((len(prompts), self.window), T5_PAD, dtype=np.int32)
        lens = np.zeros(len(prompts), dtype=np.int32)
        for i, p in enumerate(prompts):
            row = self._ids(p)
            ids[i, : len(row)] = row
            lens[i] = len(row)
        return ids, lens

    def text_len(self, text: str) -> int:
        return len(self._ids(text))

    def __call__(self, ids: np.ndarray, lens: np.ndarray) -> torch.Tensor:
        L = int(lens.max())
        t = torch.from_numpy(ids[:, :L].astype(np.int64))
        out = self.table[t] + self.pos[:L][None]
        # a little context mixing so a per-slot lookup cannot solve it
        out = out + 0.5 * torch.roll(out, 1, dims=1)
        return out.to(torch.bfloat16)
