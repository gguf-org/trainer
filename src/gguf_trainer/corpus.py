"""Corpus stage.

Text packs: a text-only prompt corpus (the teacher never sees images).
Presets pull real user prompts from public HF datasets (the same sources the
trainer/trainer2 corpora were built from); local files (one prompt per
line, or .jsonl with a "text" field) can be added or used exclusively.
Dedupes, filters by length, holds out a val split, and injects ~1% empty
prompts so the adapter also learns the checkpoint's empty CFG prompt.
-> data/train.txt, data/val.txt

Image packs (trainer5 mix): (image, instruction) samples on the edit
template — instructions synthesized from the captions of an image preset
(Flickr30k, COCO) or of local folders — a small share of two-image samples
("Image 2:" header coverage), and text-only samples on the t2i template
from the same text presets.  Val images never appear in train.  Images
are copied (raw bytes, no re-encoding) into data/images/.
-> data/train.jsonl, data/val.jsonl  {"text": str, "images": [rel paths]}
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .util import replace_atomic

PRESETS: Dict[str, Dict[str, str]] = {
    # id: dataset id, config, split, column, note
    "gustavosta": {"repo": "Gustavosta/Stable-Diffusion-Prompts", "split": "train", "col": "Prompt",
                   "title": "Stable Diffusion prompts (Gustavosta, ~80k)"},
    "midjourney": {"repo": "succinctly/midjourney-prompts", "split": "train", "col": "text",
                   "title": "Midjourney prompts (succinctly, ~250k)"},
    "diffusiondb": {"repo": "poloclub/diffusiondb", "file": "metadata.parquet", "col": "prompt",
                    "title": "DiffusionDB prompt metadata (~1.5M unique, parquet)"},
    "vidprom": {"repo": "WenhaoWang/VidProM", "split": "train", "col": "prompt",
                "title": "VidProM text-to-video prompts (~1.6M)"},
}


def preset_list() -> List[Dict[str, str]]:
    return [{"id": k, **v} for k, v in PRESETS.items()]


def _iter_preset(name: str, token: str = "") -> Iterable[str]:
    p = PRESETS[name]
    if "file" in p:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(p["repo"], p["file"], repo_type="dataset", token=token or None)
        table = pq.read_table(path, columns=[p["col"]])
        for text in table.column(p["col"]).to_pylist():
            if text:
                yield text
        return
    from datasets import load_dataset

    ds = load_dataset(p["repo"], split=p["split"], token=token or None)
    col = p["col"] if p["col"] in ds.column_names else ds.column_names[0]
    for row in ds:
        text = row.get(col)
        if text:
            yield text


def _iter_file(path: str) -> Iterable[str]:
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line).get("text", "")
                    if t:
                        yield t
        else:
            for line in f:
                yield line.rstrip("\n")


def collect_prompts(cfg: dict, token: str = "", log: Callable[[str], None] = print,
                    progress: Callable[[str], None] = lambda s: None, required: bool = True) -> List[str]:
    """The deduplicated, length-filtered text pool from the selected presets
    and local files (in source order; shuffle is the caller's)."""
    min_c, max_c = int(cfg.get("min_chars", 8)), int(cfg.get("max_chars", 1200))
    seen = set()
    pool: List[str] = []

    def add(text: str):
        text = " ".join(str(text).split())
        if not (min_c <= len(text) <= max_c):
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        pool.append(text)

    sources: List[tuple] = [("preset", n) for n in cfg.get("presets", [])]
    sources += [("file", f) for f in cfg.get("extra_files", []) if f]
    if not sources:
        if required:
            raise RuntimeError("corpus: no presets selected and no local prompt files given")
        return pool
    for kind, src in sources:
        n0 = len(pool)
        progress(f"loading {kind} {src}")
        log(f"corpus: loading {kind} '{src}' ...")
        try:
            for t in (_iter_preset(src, token) if kind == "preset" else _iter_file(src)):
                add(t)
        except Exception as e:  # a dead preset must not kill the run
            log(f"corpus: WARNING {kind} '{src}' failed ({e}); continuing")
        log(f"corpus:   +{len(pool) - n0} prompts (total {len(pool)})")
    return pool


def _write_lines(path: pathlib.Path, rows: List[str]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for p in rows:
            f.write(p.replace("\n", " ") + "\n")
    replace_atomic(tmp, path)


def _write_jsonl(path: pathlib.Path, rows: List[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    replace_atomic(tmp, path)


def build_corpus(cfg: dict, out_dir: pathlib.Path, token: str = "", log: Callable[[str], None] = print,
                 progress: Callable[[str], None] = lambda s: None) -> Dict[str, pathlib.Path]:
    """Text-only corpus -> train.txt / val.txt."""
    rng = random.Random(int(cfg.get("seed", 8)))
    pool = collect_prompts(cfg, token, log, progress)
    n_train, n_val = int(cfg.get("n_train", 60000)), int(cfg.get("n_val", 1024))
    if len(pool) < n_val + 100:
        raise RuntimeError(f"corpus: only {len(pool)} usable prompts; need more than {n_val + 100}")
    rng.shuffle(pool)
    val = pool[:n_val]
    train = pool[n_val: n_val + n_train]
    frac = float(cfg.get("empty_frac", 0.01))
    n_empty = int(len(train) * frac)
    train = train[: len(train) - n_empty] + [""] * n_empty
    rng.shuffle(train)
    n_val_empty = max(1, int(len(val) * frac)) if frac > 0 else 0
    val = val[: len(val) - n_val_empty] + [""] * n_val_empty

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rows in (("train", train), ("val", val)):
        path = out_dir / f"{name}.txt"
        _write_lines(path, rows)
        paths[name] = path
        log(f"corpus: {path.name}: {len(rows)} prompts ({sum(1 for p in rows if not p)} empty)")
    return paths


# ---------------------------------------------------------------- image corpus

def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[-80:]


def build_image_corpus(project, pack, cfg: dict, token: str = "", log: Callable[[str], None] = print,
                       progress: Callable[[str], None] = lambda s: None) -> Dict[str, pathlib.Path]:
    """(image, instruction) + text-only corpus -> train.jsonl / val.jsonl."""
    from .vision_data import (IMAGE_PRESETS, ImageSource, clean, make_instruction, make_two_image_instruction,
                              subject_from_caption)

    out_dir = project.data_dir
    img_dir = out_dir / "images"
    train_path, val_path = out_dir / "train.jsonl", out_dir / "val.jsonl"
    if train_path.is_file() and val_path.is_file():
        log("corpus: train.jsonl + val.jsonl exist, nothing to do")
        return {"train": train_path, "val": val_path}
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(cfg.get("seed", 8)))

    # -- image sources --
    sources: List[ImageSource] = []
    for pid in cfg.get("image_presets") or []:
        spec = IMAGE_PRESETS.get(pid)
        if not spec:
            log(f"corpus: WARNING unknown image preset '{pid}', skipped")
            continue
        mat = pack.material(f"images_{pid}", project)
        root = pack.material_path(project, mat) if mat else None
        if not root or not pathlib.Path(root).is_dir():
            raise RuntimeError(f"corpus: image preset '{pid}' is not downloaded (Setup → Materials)")
        sources.append(ImageSource(pid, root, spec))
    for folder in cfg.get("image_folders") or []:
        if not folder:
            continue
        p = pathlib.Path(folder).expanduser()
        if not p.is_dir():
            raise RuntimeError(f"corpus: image folder {p} does not exist")
        sources.append(ImageSource(_safe_name(p.name) or "folder", p))
    n_image, n_two = int(cfg.get("n_image", 56000)), int(cfg.get("n_two", 3000))
    n_val_image = int(cfg.get("n_val_image", 1024))
    if not sources and (n_image or n_two or n_val_image):
        raise RuntimeError("corpus: no image preset selected and no image folder given")

    entries: List[Tuple[int, str, List[str]]] = []
    for si, src in enumerate(sources):
        progress(f"listing {src.name}")
        es = src.entries()
        log(f"corpus: {src.name}: {len(es)} images ({sum(1 for _, c in es if c)} captioned)")
        entries += [(si, key, caps) for key, caps in es]
    rng.shuffle(entries)
    if entries and len(entries) < n_val_image + 8:
        raise RuntimeError(f"corpus: only {len(entries)} images; need more than {n_val_image + 8} for the val split")
    val_pool = entries[:n_val_image]
    train_pool = entries[n_val_image:]
    copied: Dict[Tuple[int, str], str] = {}

    def image_rel(si: int, key: str) -> str:
        k = (si, key)
        if k in copied:
            return copied[k]
        data, suffix = sources[si].bytes(key)
        name = f"{sources[si].name}_{_safe_name(pathlib.Path(key).stem)}{suffix or '.jpg'}"
        dst = img_dir / name
        if not dst.is_file() or dst.stat().st_size != len(data):
            tmp = dst.with_name(dst.name + ".tmp")
            with open(tmp, "wb") as f:
                f.write(data)
            replace_atomic(tmp, dst)
        rel = f"images/{name}"
        copied[k] = rel
        return rel

    def cats_of(caps: List[str]) -> List[str]:
        cs = sorted({subject_from_caption(c) for c in caps if c})
        return [c for c in cs if c != "subject"] or []

    def image_samples(pool, n: int) -> List[dict]:
        out = []
        if not pool or n <= 0:
            return out
        for i in range(n):
            si, key, caps = pool[i % len(pool)]
            if i and i % 2000 == 0:
                progress(f"images {i}/{n}")
            out.append({"text": make_instruction(rng, caps, cats_of(caps)), "images": [image_rel(si, key)]})
        return out

    def two_image_samples(pool, n: int) -> List[dict]:
        out = []
        if len(pool) < 2 or n <= 0:
            return out
        for _ in range(n):
            (sa, ka, ca), (sb, kb, cb) = rng.sample(pool, 2)
            out.append({"text": make_two_image_instruction(rng, ca, cb),
                        "images": [image_rel(sa, ka), image_rel(sb, kb)]})
        return out

    # -- text-only share --
    n_text, n_val_text = int(cfg.get("n_text", 27000)), int(cfg.get("n_val_text", 512))
    text_pool: List[str] = []
    if n_text or n_val_text:
        text_pool = [clean(t) for t in collect_prompts(cfg, token, log, progress, required=False)]
        text_pool = [t for t in text_pool if t]
        rng.shuffle(text_pool)
        if len(text_pool) < n_val_text + 8:
            raise RuntimeError(f"corpus: only {len(text_pool)} text prompts; select a prompt preset or add a "
                               f"prompt file, or set the text counts to 0")
    frac = float(cfg.get("empty_frac", 0.015))

    def text_samples(rows: List[str]) -> List[dict]:
        rows = list(rows)
        if rows and frac > 0:
            for i in rng.sample(range(len(rows)), max(1, int(len(rows) * frac))):
                rows[i] = ""
        return [{"text": t, "images": []} for t in rows]

    val_texts = text_pool[:n_val_text]
    train_texts = text_pool[n_val_text: n_val_text + n_text]

    if not val_path.is_file():
        progress("building val.jsonl")
        val = image_samples(val_pool, len(val_pool)) + text_samples(val_texts)
        rng.shuffle(val)
        _write_jsonl(val_path, val)
        log(f"corpus: val.jsonl: {len(val)} samples ({len(val_pool)} image, {len(val_texts)} text)")
    if not train_path.is_file():
        progress("building train.jsonl")
        train = image_samples(train_pool, n_image)
        train += two_image_samples(train_pool, n_two)
        train += text_samples(train_texts)
        rng.shuffle(train)
        _write_jsonl(train_path, train)
        n_img = sum(1 for s in train if len(s["images"]) == 1)
        n_2 = sum(1 for s in train if len(s["images"]) == 2)
        log(f"corpus: train.jsonl: {len(train)} samples ({n_img} image over {len(train_pool)} distinct images, "
            f"{n_2} two-image, {len(train) - n_img - n_2} text); {len(copied)} images in {img_dir}")
    for s in sources:
        s.close()
    return {"train": train_path, "val": val_path}
