"""pig_ming_vision-<quant>.gguf: the Ling Qwen2.5-VL vision tower + linear_proj
of the Ming-Image MLLM snapshot (mllm/model-*.safetensors, tensor names
`vision.*` and `linear_proj.*` kept verbatim), for ggk's `--llm_vision`.

Layout = the vision part of the Ming-Image text encoder GGUF (ggk loads
either under text_encoders.llm.): norms and biases f32, the 5-D
`vision.patch_embed.proj.weight` [1280, 3, 2, 14, 14] as BF16 (the engine's
folded-5-D Conv3d path reads it), the other 2-D weights q8_0 (gguf-py's
reference quantizer) or f16.  The engine never reads general.architecture.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import torch
from gguf_connector.const import GGMLQuantizationType as T
from gguf_connector.writer import GGUFWriter

from .ming_teacher import load_named_tensors
from .util import replace_atomic

QUANTS = ("q8_0", "f16")


def export_ming_vision(mllm_dir: pathlib.Path, out: pathlib.Path, quant: str = "q8_0", log=print) -> pathlib.Path:
    quant = quant.lower()
    if quant not in QUANTS:
        raise ValueError(f"vision_quant must be one of {QUANTS}, not {quant!r}")
    t0 = time.time()
    mllm_dir = pathlib.Path(mllm_dir)
    sd = load_named_tensors(mllm_dir, lambda n: n.startswith("vision.") or n.startswith("linear_proj."))
    if not sd or "vision.patch_embed.proj.weight" not in sd or "linear_proj.2.weight" not in sd:
        raise RuntimeError(f"{mllm_dir}: vision.* / linear_proj.* tensors not found")
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    w = GGUFWriter(str(tmp), "pig")
    w.add_name(out.stem)
    w.add_string("ming_vision.source", "inclusionAI/Ming-Image-0.1-Design mllm (vision + linear_proj)")
    w.add_string("ming_vision.quant", quant)
    counts = {"f32": 0, "f16": 0, "bf16": 0, "q8_0": 0}
    for name in sorted(sd):
        t = sd[name]
        if t.ndim == 5:
            # the Conv3d patch embed: keep the checkpoint's bf16 bits, 5-D shape
            bits = t.to(torch.bfloat16).contiguous().view(torch.int16).numpy()
            w.add_tensor(name, bits, raw_shape=list(t.shape), raw_dtype=T.BF16)
            counts["bf16"] += 1
            continue
        a = t.float().numpy()
        if a.ndim == 1:
            w.add_tensor(name, a.astype(np.float32))
            counts["f32"] += 1
        elif quant == "q8_0" and a.shape[-1] % 32 == 0:
            from gguf_connector.quant import quantize

            w.add_tensor(name, quantize(a, T.Q8_0), raw_dtype=T.Q8_0)   # byte-shaped data: the writer derives the element shape
            counts["q8_0"] += 1
        else:
            w.add_tensor(name, a.astype(np.float16))
            counts["f16"] += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    replace_atomic(tmp, out)
    log(f"vision: wrote {out} ({len(sd)} tensors: {counts}) in {time.time() - t0:.0f}s")
    return out
