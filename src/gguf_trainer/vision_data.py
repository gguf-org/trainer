"""The MageFlow-Edit conditioning contract (ggk conditioner.hpp,
sd_version_is_mage_flow + core/util.cpp clip_preprocess), image-corpus
sources and the instruction synthesizer — trainer5's data_common.py +
build_corpus.py folded into the package.

Edit path (reference images present, prompt_template_encode_start_idx = 64):

    <|im_start|>system\\n{EDIT_SYSTEM}<|im_end|>\\n<|im_start|>user\\n     64 tok
    per image i:  "Image {i+1}: <|vision_start|>"                        6 tok
                  "<|image_pad|>" * n_img_tokens  "<|vision_end|>"
    {text}<|im_end|>\\n<|im_start|>assistant\\n

Text-to-image path (no image, start_idx = 34):

    <|im_start|>system\\n{T2I_SYSTEM}<|im_end|>\\n<|im_start|>user\\n      34 tok
    {text}<|im_end|>\\n<|im_start|>assistant\\n

Raw BPE (add_special_tokens=False), no padding.  Image resize: h/w rounded
to factor 32; if max(orig side) > 384, beta = 384/max_side, floored to
factor.  clip_preprocess: NEAREST resize (integer out*in//out index), center
crop, clamp 0..1, OpenAI-CLIP mean/std.  Do not "improve" the resize — an
antialiased resample changes every vision token the teacher saw.
"""

from __future__ import annotations

import csv
import io
import json
import os
import pathlib
import random
import re
import zipfile
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

T_DIM = 2560   # teacher / mmproj / adapter-out hidden
S_DIM = 1024   # student hidden
V_DIM = 2560   # raw vision embed (mmproj out_hidden_size)

FACTOR = 32          # patch 16 * spatial merge 2
MAX_SIDE = 384       # mage_flow max_pixels (a max SIDE, unlike qwen-image)
MAX_LEN = 1024       # engine cap is 2048+64; the corpus stays far below

EDIT_PREFIX = ("<|im_start|>system\nDescribe the key features of the input "
               "image (color, shape, size, texture, objects, background), "
               "then explain how the user's text instruction should alter or "
               "modify the image. Generate a new image that meets the user's "
               "requirements while maintaining consistency with the original "
               "input where appropriate.<|im_end|>\n<|im_start|>user\n")
T2I_PREFIX = ("<|im_start|>system\nDescribe the image by detailing the "
              "color, shape, size, texture, quantity, text, spatial "
              "relationships of the objects and background:<|im_end|>\n"
              "<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"

EDIT_START_IDX = 64  # engine hardcodes both; assert_template_counts checks
T2I_START_IDX = 34
IMAGE_HEADER_TOKENS = 6

IMAGE_PAD_ID = 151655      # <|image_pad|>
VISION_START_ID = 151652   # <|vision_start|>
VISION_END_ID = 151653     # <|vision_end|>

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ---------------------------------------------------------------- images

def mageflow_target_size(width: int, height: int) -> Tuple[int, int]:
    """conditioner.hpp mage_flow branch: -> (w_bar, h_bar), each a multiple
    of FACTOR.  round() first; the >MAX_SIDE downscale uses floor()."""
    f = float(FACTOR)
    h_bar = max(FACTOR, int(round(height / f) * f))
    w_bar = max(FACTOR, int(round(width / f) * f))
    max_side = max(height, width)
    if max_side > MAX_SIDE:
        beta = np.float32(MAX_SIDE) / np.float32(max_side)  # C++ float math
        h_bar = max(FACTOR, int(np.floor(np.float32(height) * beta / f)) * FACTOR)
        w_bar = max(FACTOR, int(np.floor(np.float32(width) * beta / f)) * FACTOR)
    return w_bar, h_bar


def n_image_tokens(w_bar: int, h_bar: int) -> int:
    return (w_bar // FACTOR) * (h_bar // FACTOR)


def nearest_resize(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """ggk sd::ops::interpolate default (Nearest): in_idx = out*in//out_size.
    img: [H, W, 3] float32."""
    in_h, in_w = img.shape[:2]
    ys = (np.arange(out_h, dtype=np.int64) * in_h) // out_h
    xs = (np.arange(out_w, dtype=np.int64) * in_w) // out_w
    return img[ys][:, xs]


def clip_preprocess(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """core/util.cpp clip_preprocess: aspect-preserving nearest resize
    (scale = max, sizes truncated from float32 products), center crop,
    clamp, CLIP mean/std.  img: [H, W, 3] float32 in 0..1."""
    in_h, in_w = img.shape[:2]
    width_scale = np.float32(target_w) / np.float32(in_w)
    height_scale = np.float32(target_h) / np.float32(in_h)
    scale = max(width_scale, height_scale)
    resized_w = int(np.float32(scale) * np.float32(in_w))
    resized_h = int(np.float32(scale) * np.float32(in_h))
    r = nearest_resize(img, resized_w, resized_h)
    if resized_h < target_h or resized_w < target_w:
        # float truncation can land the resize 1px short of the target on one
        # axis (the C++ engine reads past the row there); edge-replicate
        r = np.pad(r, ((0, max(0, target_h - resized_h)),
                       (0, max(0, target_w - resized_w)), (0, 0)),
                   mode="edge")
    h_off = max((resized_h - target_h) // 2, 0)
    w_off = max((resized_w - target_w) // 2, 0)
    r = r[h_off: h_off + target_h, w_off: w_off + target_w]
    r = np.clip(r, 0.0, 1.0)
    return (r - CLIP_MEAN) / CLIP_STD


def load_image(path) -> np.ndarray:
    """-> [H, W, 3] float32 in 0..1 (engine: sd_image_get_f32 scale=1/255)."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def preprocess_ref_image(path) -> Tuple[np.ndarray, int]:
    """Full engine path for one ref image -> ([h, w, 3] normalized float32,
    n_tokens)."""
    img = load_image(path)
    h, w = img.shape[:2]
    w_bar, h_bar = mageflow_target_size(w, h)
    return clip_preprocess(img, w_bar, h_bar), n_image_tokens(w_bar, h_bar)


def image_token_count(path) -> int:
    """n vision tokens of an image file without decoding its pixels."""
    from PIL import Image

    with Image.open(path) as im:
        w, h = im.size
    return n_image_tokens(*mageflow_target_size(w, h))


# ---------------------------------------------------------------- tokens

_template_checked: Dict[int, bool] = {}


def assert_template_counts(tok) -> None:
    """The engine hardcodes start_idx 64 (edit) / 34 (t2i) and the 6-token
    image header; fail loudly if the tokenizer disagrees."""
    if _template_checked.get(id(tok)):
        return
    n_edit = len(tok(EDIT_PREFIX, add_special_tokens=False)["input_ids"])
    n_t2i = len(tok(T2I_PREFIX, add_special_tokens=False)["input_ids"])
    if n_edit != EDIT_START_IDX:
        raise RuntimeError(f"edit prefix tokenizes to {n_edit} tokens, the engine hardcodes {EDIT_START_IDX}")
    if n_t2i != T2I_START_IDX:
        raise RuntimeError(f"t2i prefix tokenizes to {n_t2i} tokens, the engine hardcodes {T2I_START_IDX}")
    for i in range(3):
        hdr = tok(f"Image {i + 1}: <|vision_start|>", add_special_tokens=False)["input_ids"]
        if len(hdr) != IMAGE_HEADER_TOKENS:
            raise RuntimeError(f"image header {i + 1} tokenizes to {len(hdr)} tokens, expected {IMAGE_HEADER_TOKENS}")
    _template_checked[id(tok)] = True


def build_sample(tok, text: str, image_token_counts: Sequence[int]) -> Tuple[List[int], List[Tuple[int, int]]]:
    """-> (ids, vis_segments [(start, n)]) for one sample.

    image_token_counts empty -> t2i template; else edit template with one
    image block per count.  vis_segments give the positions of the
    <|image_pad|> runs (where vision embeds replace the input embeddings).
    Matches the engine: image_embed_idx starts at 64+6 and advances by
    1 + n + 6 per image.
    """
    assert_template_counts(tok)
    text = text or ""
    if not image_token_counts:
        ids = tok(T2I_PREFIX + text + SUFFIX, add_special_tokens=False)["input_ids"]
        return ids, []
    img_prompt = ""
    vis_segments = []
    idx = EDIT_START_IDX + IMAGE_HEADER_TOKENS
    for i, n in enumerate(image_token_counts):
        img_prompt += f"Image {i + 1}: <|vision_start|>" + "<|image_pad|>" * n + "<|vision_end|>"
        vis_segments.append((idx, n))
        idx += 1 + n + IMAGE_HEADER_TOKENS
    ids = tok(EDIT_PREFIX + img_prompt + text + SUFFIX, add_special_tokens=False)["input_ids"]
    for start, n in vis_segments:
        if ids[start - 1] != VISION_START_ID or ids[start + n] != VISION_END_ID \
                or any(t != IMAGE_PAD_ID for t in ids[start: start + n]):
            raise RuntimeError("vision_start/pad/end tokens are not where the engine expects them")
    return ids, vis_segments


def start_idx(n_images: int) -> int:
    return EDIT_START_IDX if n_images else T2I_START_IDX


# ---------------------------------------------------------------- image sources

IMAGE_PRESETS: Dict[str, Dict[str, object]] = {
    "flickr30k": {
        "repo": "nlphuji/flickr30k", "repo_type": "dataset", "subdir": "flickr30k",
        "patterns": ["flickr30k-images.zip", "flickr_annotations_30k.csv"], "format": "flickr_zip",
        "title": "Flickr30k (31k photos, 5 captions each, 4.4 GB)",
        "hint": "one zip + a CSV; images are copied out of the zip into <project>/data/images as needed",
    },
    "coco_captions": {
        "repo": "Multimodal-Fatima/COCO_captions_train", "repo_type": "dataset", "subdir": "coco_captions",
        "patterns": ["data/*.parquet"], "format": "parquet", "image_col": "image", "caption_col": "sentences_raw",
        "title": "COCO captions, Karpathy train split (113k photos, 5 captions each, 17 GB)",
        "hint": "38 parquet shards with the JPEGs embedded (the trainer5 reference corpus was COCO)",
    },
}


def image_preset_list() -> List[Dict[str, object]]:
    return [{"id": k, "repo": v["repo"], "title": v["title"], "hint": v.get("hint", "")} for k, v in IMAGE_PRESETS.items()]


class ImageSource:
    """Uniform access to a preset snapshot or a local folder:
    keys() -> [(key, captions)] cheaply, then bytes(key) for the chosen ones."""

    def __init__(self, name: str, root: pathlib.Path, spec: Optional[Dict[str, object]] = None):
        self.name = name
        self.root = pathlib.Path(root)
        self.spec = spec
        self._entries: Optional[List[Tuple[str, List[str]]]] = None
        self._zip: Optional[zipfile.ZipFile] = None
        self._zip_names: Dict[str, str] = {}
        self._parquet_index: Dict[str, Tuple[str, int, int]] = {}   # key -> (file, row_group, row)

    # -- enumeration --
    def entries(self) -> List[Tuple[str, List[str]]]:
        if self._entries is None:
            fmt = (self.spec or {}).get("format", "folder")
            if fmt == "flickr_zip":
                self._entries = self._flickr_entries()
            elif fmt == "parquet":
                self._entries = self._parquet_entries()
            else:
                self._entries = self._folder_entries()
        return self._entries

    def _flickr_entries(self):
        csv_path = next(self.root.glob("*.csv"), None)
        if csv_path is None:
            raise RuntimeError(f"{self.name}: no annotations CSV under {self.root}")
        out = []
        with open(csv_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    caps = json.loads(row.get("raw") or "[]")
                except ValueError:
                    caps = []
                fn = row.get("filename") or ""
                if fn:
                    out.append((fn, [str(c) for c in caps if c]))
        return out

    def _parquet_entries(self):
        import pyarrow.parquet as pq

        cap_col = str(self.spec.get("caption_col", "caption"))
        out = []
        files = sorted(self.root.glob("data/*.parquet")) or sorted(self.root.rglob("*.parquet"))
        for fp in files:
            pf = pq.ParquetFile(str(fp))
            cols = [c for c in (cap_col,) if c in pf.schema_arrow.names]
            for rg in range(pf.num_row_groups):
                tbl = pf.read_row_group(rg, columns=cols)
                caps_col = tbl.column(cap_col).to_pylist() if cap_col in tbl.column_names else [None] * tbl.num_rows
                for i in range(tbl.num_rows):
                    key = f"{fp.stem}-{rg}-{i}"
                    caps = caps_col[i]
                    if isinstance(caps, str):
                        caps = [caps]
                    caps = [str(c).strip() for c in (caps or []) if c]
                    self._parquet_index[key] = (str(fp), rg, i)
                    out.append((key, caps))
        return out

    def _folder_entries(self):
        out = []
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                caps = []
                sidecar = p.with_suffix(".txt")
                if sidecar.is_file():
                    caps = [l.strip() for l in sidecar.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
                out.append((str(p.relative_to(self.root)), caps))
        return out

    # -- bytes --
    def bytes(self, key: str) -> Tuple[bytes, str]:
        """-> (raw file bytes, suffix)"""
        fmt = (self.spec or {}).get("format", "folder")
        if fmt == "flickr_zip":
            if self._zip is None:
                zp = next(self.root.glob("*.zip"), None)
                if zp is None:
                    raise RuntimeError(f"{self.name}: no image zip under {self.root}")
                self._zip = zipfile.ZipFile(zp)
                self._zip_names = {os.path.basename(n): n for n in self._zip.namelist() if not n.endswith("/")}
            name = self._zip_names.get(key)
            if name is None:
                raise KeyError(f"{key} not in {self._zip.filename}")
            return self._zip.read(name), pathlib.Path(key).suffix.lower() or ".jpg"
        if fmt == "parquet":
            import pyarrow.parquet as pq

            fp, rg, i = self._parquet_index[key]
            col = str(self.spec.get("image_col", "image"))
            tbl = pq.ParquetFile(fp).read_row_group(rg, columns=[col])
            cell = tbl.column(col)[i].as_py()
            if isinstance(cell, dict):
                data = cell.get("bytes")
                path = cell.get("path") or ""
                if data is None and path:
                    data = open(path, "rb").read()
            else:
                data, path = cell, ""
            suffix = pathlib.Path(path).suffix.lower() if path else ""
            if not suffix:
                suffix = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
            return data, suffix
        p = self.root / key
        return p.read_bytes(), p.suffix.lower()

    def close(self):
        if self._zip is not None:
            self._zip.close()
            self._zip = None


# ---------------------------------------------------------------- instructions

ADD_THINGS = [
    "sunglasses", "a red hat", "a rainbow", "falling snow", "a small bird",
    "colorful balloons", "a neon sign", "fireworks", "a full moon", "fog",
    "autumn leaves", "a wooden fence", "string lights", "a cup of coffee",
    "a red scarf", "wildflowers", "a butterfly", "graffiti on the wall",
    "raindrops", "a pair of headphones", "a golden crown", "a tiny flag",
    "candles", "soap bubbles", "a shadow", "sparkles",
]
STYLES = [
    "night", "winter", "a watercolor painting", "an oil painting",
    "cyberpunk style", "black and white", "vintage sepia", "anime style",
    "pixel art", "a pencil sketch", "golden hour lighting",
    "a foggy morning", "neon lit", "a comic book style",
    "a low-poly 3d render", "a claymation scene", "impressionist style",
]
BACKGROUNDS = [
    "a beach at sunset", "a snowy mountain", "a dense forest",
    "a futuristic city", "a desert", "outer space", "a cozy living room",
    "an ancient temple", "a rainy street", "a field of flowers",
    "a library", "an underwater scene",
]
COLORS = ["red", "blue", "green", "golden", "purple", "pink", "white",
          "black", "orange", "turquoise"]
TWO_IMAGE_PATTERNS = [
    "combine the two images into one scene",
    "put the {cat} from Image 1 into Image 2",
    "blend Image 1 and Image 2 together",
    "apply the style of Image 2 to Image 1",
    "place the subject of Image 1 next to the subject of Image 2",
    "{cap}",
]
_DETERMINERS = {"a", "an", "the", "this", "that", "some", "one", "two", "three", "four", "five", "six",
                "several", "many", "few", "group", "of", "couple", "pair", "lots", "there", "is", "are"}
_ADJECTIVES = {"young", "old", "small", "large", "little", "big", "tall", "short", "tiny", "huge", "cute",
               "happy", "smiling", "elderly", "adult", "baby", "beautiful", "pretty", "asian", "african",
               "white", "black", "brown", "red", "blue", "green", "yellow", "orange", "pink", "purple",
               "gray", "grey", "dark", "light", "blond", "blonde", "shirtless", "bearded", "wet", "dirty",
               "long", "wooden", "busy", "crowded", "empty", "colorful", "bright"}


def clean(s: str) -> str:
    s = re.sub(r"[()\[\]]", "", str(s))  # engine parses ()/[] as attention syntax
    return re.sub(r"\s+", " ", s).strip()


def subject_from_caption(caption: str) -> str:
    """The head noun of a caption ('A young man in a blue shirt ...' -> 'man'),
    used where trainer5 used COCO instance categories."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", caption.lower())
    for w in words:
        if w in _DETERMINERS or w in _ADJECTIVES:
            continue
        if len(w) < 3:
            continue
        return w
    return "subject"


def make_instruction(rng: random.Random, caps: Sequence[str], cats: Sequence[str]) -> str:
    """trainer5's instruction mix: plain captions, add/remove/replace/restyle/
    recolor patterns, background swaps, caption+style, truncated captions,
    ~1.5% empty."""
    cap = clean(rng.choice(caps)) if caps else ""
    cat = rng.choice(list(cats)) if cats else "subject"
    r = rng.random()
    if r < 0.015:
        return ""
    if r < 0.30 and cap:
        return cap
    if r < 0.42:
        t = rng.choice(ADD_THINGS)
        return rng.choice([f"add {t}", f"add {t} to the image", f"the {cat} with {t}"])
    if r < 0.50:
        return rng.choice([f"remove the {cat}", f"remove the {cat} from the image"])
    if r < 0.56:
        return f"replace the {cat} with {rng.choice(ADD_THINGS)}"
    if r < 0.68:
        s = rng.choice(STYLES)
        return rng.choice([f"make it {s}", f"turn this into {s}", f"restyle the image as {s}"])
    if r < 0.76:
        return f"change the background to {rng.choice(BACKGROUNDS)}"
    if r < 0.82:
        return f"make the {cat} {rng.choice(COLORS)}"
    if r < 0.90 and cap:
        return f"{cap}, {rng.choice(STYLES)}"
    if r < 0.94 and cap:
        words = cap.split()
        return " ".join(words[: rng.randint(2, max(2, min(5, len(words))))])
    return cap or f"make it {rng.choice(STYLES)}"


def make_two_image_instruction(rng: random.Random, caps_a: Sequence[str], caps_b: Sequence[str]) -> str:
    pat = rng.choice(TWO_IMAGE_PATTERNS)
    cat = subject_from_caption(caps_a[0]) if caps_a else "subject"
    cap = clean(rng.choice(caps_b)) if caps_b else "combine the two images"
    return pat.format(cat=cat, cap=cap)
