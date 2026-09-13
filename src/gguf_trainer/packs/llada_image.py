"""LLaDA-Image (inclusionAI/LLaDA-Image-Turbo) pack — the trainer8 recipe.

Teacher: LLaDA2-MoE (16B-A1.4B) text encoder + QueryFormer (256 learned
queries appended to the token embeddings, text masked from seeing them) +
6-layer text_projection -> cap_feats.  The DiT conditioned on the 256 query
rows alone reproduces the full result (trainer8 M0 ablation), so the target
per prompt is exactly those rows: [256, 2560].

The reference image never enters the text encoder: SigVQ handles it, and
ggk loads that encoder from its own GGUF (`--llm_vision`), which this pack
can export from the same HF snapshot.  The adapter is therefore text-only
and works unchanged for text-to-image and editing.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

from .base import Material, TrainerPack, student_materials

HF_REPO = "inclusionAI/LLaDA-Image-Turbo"
TEXT_STACK_PATTERNS = ["model_index.json", "text_encoder/*", "queryformer/*", "text_projection/*",
                       "tokenizer/*"]
SIGVQ_PATTERNS = ["sigvq/*"]

PREFIX = "<role>HUMAN</role> Generate an image: "
PREFIX_EMPTY = "<role>HUMAN</role> Generate an image."
SUFFIX = "\n<role>ASSISTANT</role>\n<IMAGE1>"


class LLaDAImagePack(TrainerPack):
    id = "llada_image"
    title = "LLaDA-Image-Turbo (LLaDA2-MoE text encoder)"
    description = ("Replaces the 16B LLaDA2-MoE text stack of LLaDA-Image-Turbo with pig_clip + a "
                   "256-query resampler adapter (trainer8 recipe). Pairs with the SigVQ vision "
                   "encoder for image editing.")
    adapter_kind = "resampler"
    out_dim = 2560
    num_queries = 256
    max_len_teacher = 2048
    default_name = "llada_adapter"
    eval_keys = ["cos_centered", "cos_rms", "worst_row_cos_rms", "rel_mse", "cos_raw", "roundtrip_cos",
                 "trained_steps", "val_batches"]
    hints = {
        "corpus": ("The teacher never sees images, so the corpus is text only. ~1% empty prompts teach the "
                   "checkpoint's empty CFG prompt. 60k prompts took ~3.7 h of teacher time on an RTX 5090 "
                   "(4.5 prompts/s)."),
        "precompute": ("The teacher is spread over the CUDA devices and then CPU RAM (accelerate device_map). "
                       "sequential fills the chosen device up to its memory budget, then the other GPUs, then RAM — "
                       "on a 5090 + 4050 the whole 30 GB backbone stays on the GPUs. balanced is transformers' "
                       "default even split: it caps the big card at half the model and offloads the rest to the CPU, "
                       "where each MoE layer runs its 256 experts eagerly (2.4 vs ~4+ prompts/s). If sequential runs "
                       "out of VRAM, lower the GPU memory budget by 2–3 GiB instead of switching. Token budget caps "
                       "batch × (text tokens + 256 query rows)."),
        "train": ("trainer8 recipe: width 1024, depth 6, 20k steps, batch 32, lr 2e-4, warmup 1k, cosine to 10%, "
                  "whitened MSE + 0.5·(1−cos) on standardized targets. Judge by val centred cosine (0.965 on the "
                  "5090 run, ~55 min); below ~0.9 widen (1536) or deepen before touching the recipe."),
        "eval": ("Judge by cos_centered / rel_mse: the teacher rows share a large per-dim offset, so plain cosine "
                 "is ~0.99 even for a zero prediction. cos_rms is what the DiT consumes (per-row RMSNorm). "
                 "roundtrip_cos compares the exported f16 GGUF against the checkpoint (expect ≥ 0.999)."),
        "output": ("Text-to-image works with every student quantization; editing wants pig_clip at q8_0 or better "
                   "and prefers the f16 DiT (trainer8 A/B)."),
    }
    # 2 = rotary tables repaired after the transformers-5 meta-device load
    # (contract-1 shards were computed with uninitialized RoPE and are wrong)
    shard_contract = "llada_image/2"

    def materials(self, project=None) -> List[Material]:
        return student_materials() + [
            Material("teacher", "Teacher: LLaDA-Image-Turbo text stack", "hf_snapshot", required=True,
                     repo=HF_REPO, subdir="LLaDA-Image-Turbo", patterns=TEXT_STACK_PATTERNS,
                     hint="LLaDA2-MoE text encoder (bf16, ~33 GB) + QueryFormer + text_projection + "
                     "tokenizer. Needs ~34 GB of GPU+CPU memory to run."),
            Material("sigvq", "Vision encoder: SigVQ (for editing)", "hf_snapshot", required=False,
                     repo=HF_REPO, subdir="LLaDA-Image-Turbo", patterns=SIGVQ_PATTERNS,
                     hint="~2.4 GB, optional. Exported to <name>_sigvq-f16.gguf for --llm_vision."),
        ]

    def format_prompt(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return PREFIX_EMPTY + SUFFIX
        return PREFIX + text + SUFFIX

    def teacher_dir(self, project) -> pathlib.Path:
        return self.material_path(project, self.material("teacher"))

    def build_teacher(self, project, device, gpu_mem, cpu_mem, log):
        from ..llada_teacher import LLaDATeacher

        placement = (project.config.get("precompute") or {}).get("placement", "sequential")
        return LLaDATeacher(self.teacher_dir(project), device, gpu_mem, cpu_mem, self.format_prompt,
                            max_len=self.max_len_teacher, log=log, placement=placement)

    def build_mock_teacher(self, log):
        from ..llada_teacher import MockTeacher

        return MockTeacher(self.num_queries, self.out_dim, self.format_prompt, log)

    def export_kv(self, project) -> Dict[str, Any]:
        return {"adapter.qwen_hidden_source": "final_norm",
                "adapter.target": "llada_image_turbo text_projection query rows",
                "adapter.teacher": HF_REPO}

    def extra_exports(self, project, log) -> List[pathlib.Path]:
        if not (project.config.get("export") or {}).get("export_sigvq", True):
            return []
        root = self.material_path(project, self.material("sigvq"))
        st = root / "sigvq" / "diffusion_pytorch_model.safetensors" if root else None
        if not st or not st.is_file():
            log("sigvq: safetensors not downloaded, skipping vision-encoder export")
            return []
        from ..sigvq_export import export_sigvq

        out = project.output_dir / f"{project.export_base()}_sigvq-f16.gguf"
        if out.is_file() and out.stat().st_mtime >= st.stat().st_mtime:
            log(f"sigvq: {out.name} up to date")
            return [out]
        export_sigvq(st, out, log)
        return [out]

    def engine_command(self, student: str, adapter: str, extras: Optional[Dict[str, str]] = None) -> str:
        vis = (extras or {}).get("sigvq", "<sigvq.gguf>")
        return (f"ggk diffuser engine -- --diffusion-model LLaDA-image-turbo-nvfp4.gguf "
                f"--vae pig_flux2_vae_fp32-f16.gguf --llm {student} --llm-adapter {adapter} "
                f"--llm_vision {vis} -p \"a sheep in sunglasses\" --cfg-scale 1.0 --steps 4 "
                f"--sampling-method euler --diffusion-fa -o out.png")
