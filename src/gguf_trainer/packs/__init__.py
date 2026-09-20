"""Trainer packs: one per (teacher text encoder -> pig_clip adapter) recipe."""

from __future__ import annotations

from typing import Dict, List

from .base import TrainerPack
from .llada_image import LLaDAImagePack
from .qwen3vl_mageflow import Qwen3VLMageFlowPack
from .qwen3vl_qwenimage import Qwen3VLQwenImagePack
from .t5xxl_pixart import T5XXLPixArtPack

_PACKS: Dict[str, TrainerPack] = {p.id: p for p in (LLaDAImagePack(), Qwen3VLMageFlowPack(), Qwen3VLQwenImagePack(),
                                                 T5XXLPixArtPack())}


def get_pack(pack_id: str) -> TrainerPack:
    if pack_id not in _PACKS:
        raise KeyError(f"unknown trainer pack '{pack_id}' (have: {', '.join(_PACKS)})")
    return _PACKS[pack_id]


def list_packs() -> List[Dict[str, object]]:
    return [p.describe() for p in _PACKS.values()]
