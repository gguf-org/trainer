"""SigVQ vision encoder -> GGUF for ggk's --llm_vision (LLaDA-Image editing).

The engine loads the HF tensor names verbatim under its own "sigvq." prefix
and never reads a key/value, so the file is the safetensors content cast to
f16 (what the shipped pig_llada_sigvq-f16.gguf is: 491 tensors, all f16).
"""

from __future__ import annotations

import pathlib

import numpy as np

from .util import replace_atomic


def export_sigvq(safetensors_path: pathlib.Path, out: pathlib.Path, log=print) -> pathlib.Path:
    import torch
    from gguf_connector.reader import GGUFReader
    from gguf_connector.writer import GGUFWriter
    from safetensors import safe_open

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    w = GGUFWriter(str(tmp), "pig")
    w.add_name(out.stem)
    n = 0
    with safe_open(str(safetensors_path), framework="pt") as f:
        for name in sorted(f.keys()):
            t = f.get_tensor(name)
            a = t.to(torch.float32).numpy().astype(np.float16)
            w.add_tensor(name, a)
            n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    r = GGUFReader(str(tmp))
    assert len(r.tensors) == n, (len(r.tensors), n)
    del r
    replace_atomic(tmp, out)
    log(f"sigvq: wrote {out} ({n} f16 tensors)")
    return out
