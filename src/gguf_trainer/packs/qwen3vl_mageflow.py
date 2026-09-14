"""MageFlow-Edit pack — the trainer5 recipe with pig_clip as the student.

MageFlow-Edit conditions its DiT on Qwen3-VL-4B-Instruct final-norm hidden
states with the reference image spliced in as mmproj vision tokens
(`--llm qwen3vl-4b-it-q4_k_m.gguf --llm_vision mmproj-qwen3vl-4b-it-f16.gguf`).
This pack distills that conditioning into pig_clip + a token-aligned adapter
with a vision extension, so the same generation runs with

    --llm pig_clip-<quant>.gguf --llm-adapter pig_qwen3vl_4b_adapter-f16.gguf
    --llm_vision mmproj-qwen3vl-4b-it-f16.gguf

The 4B mmproj stays exactly as the teacher uses it; the adapter owns the
2560 <-> 1024 bridge in both directions: a frozen `vision_proj` (ridge least
squares over the shared vocabulary, applied by the ENGINE to every mmproj
embed before the student) and a trained `vis_in` that hands the adapter the
RAW mmproj embeds, so vision fidelity does not depend on what survives the
0.6B student.  Teacher tap: final norm (out_layers = {}), ggk drops
deepstack, all-equal M-RoPE == plain rope, nearest resize + CLIP norm.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

from ..vision_data import IMAGE_PRESETS, V_DIM
from .base import Material, TrainerPack, student_materials

HF_REPO = "Qwen/Qwen3-VL-4B-Instruct"
TEACHER_PATTERNS = ["config.json", "generation_config.json", "model.safetensors.index.json", "model-*.safetensors",
                    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "chat_template.json"]


class Qwen3VLMageFlowPack(TrainerPack):
    id = "qwen3vl_mageflow"
    title = "MageFlow-Edit (Qwen3-VL-4B-Instruct + mmproj)"
    description = ("Replaces the Qwen3-VL-4B-Instruct text stack of MageFlow-Edit with pig_clip + a "
                   "token-aligned adapter with a vision extension (trainer5 recipe). Pairs with the "
                   "unchanged mmproj-qwen3vl-4b-it-f16.gguf vision encoder for image editing and "
                   "text-to-image.")
    adapter_kind = "token_aligned_vision"
    out_dim = 2560
    vis_dim = V_DIM
    num_queries = 0
    needs_images = True
    default_name = "qwen3vl_4b_adapter"
    max_len_student = 1024
    shard_contract = "qwen3vl_mageflow/1"
    eval_keys = ["cos_slice", "cos_vis", "cos_txt", "cos", "rel_mse", "worst_sample_cos_slice", "roundtrip_cos",
                 "trained_steps", "val_samples"]
    hints = {
        "corpus": ("(image, instruction) samples on the edit template plus text-only prompts on the "
                   "text-to-image template — the mix trainer5 used (56k single-image, 3k two-image, 27k text). "
                   "Instructions are synthesized from the captions (add / remove / replace / restyle / recolor / "
                   "background, plain and truncated captions, ~1.5% empty). Images are copied into "
                   "<project>/data/images; a local folder works too (optional <name>.txt captions next to each image)."),
        "precompute": ("Vision tower (GPU-resident) + Qwen3-VL text stack + pig_clip run once per shard. The text stack "
                       "(~8 GB bf16) stays on the GPU when the budget allows, otherwise it is streamed through it with "
                       "accelerate cpu_offload (RAM must hold it; ~3.5 samples/s on a 6 GB card, ~44 on a 5090 resident). "
                       "Budgets at 0 pick values for the card. Each sample stores the teacher and student states of every "
                       "token plus the raw vision embeds (~2.4 MB per image sample) — plan disk accordingly."),
        "train": ("trainer5 recipe: width 1024 (the gate winner), depth 4, 20k steps, batch 32, lr 2e-4, warmup 1k, "
                  "whitened MSE + 0.5·(1−cos) masked to real tokens, out_proj zero-init, vision_proj frozen. "
                  "Reference run: val cos 0.915 (vision positions 0.80, text 0.98) and the A/B edits were near-identical "
                  "to the teacher — the 4-step DiT forgives far more than the cosine suggests."),
        "eval": ("Judge by cos_slice (positions ≥ the template start — what the DiT consumes) and the vision/text split: "
                 "a lagging cos_vis means the vis_in path or width is the limiter. roundtrip_cos compares the exported "
                 "f16 GGUF against the checkpoint (expect ≥ 0.999)."),
        "output": ("Editing wants pig_clip at q8_0 or better; text-to-image works with every quantization. The mmproj "
                   "file is the teacher's own — it is not exported here."),
    }

    def defaults(self) -> Dict[str, Any]:
        return {
            "corpus": {"image_presets": ["flickr30k"], "image_folders": [], "n_image": 56000, "n_two": 3000,
                       "n_text": 27000, "n_val_image": 1024, "n_val_text": 512, "empty_frac": 0.015,
                       "presets": ["gustavosta", "midjourney"]},
            "precompute": {"shard_size": 512, "val_shard_size": 256, "teacher_batch": 0, "tok_budget": 0,
                           "student_batch": 0, "student_tok_budget": 0, "vis_tok_budget": 0, "teacher_mode": "auto"},
            "train": {"width": 1024, "depth": 4, "steps": 20000, "batch_size": 32},
            "export": {"export_sigvq": False},
        }

    def materials(self, project=None) -> List[Material]:
        mats = student_materials() + [
            Material("teacher", "Teacher: Qwen3-VL-4B-Instruct", "hf_snapshot", required=True,
                     repo=HF_REPO, subdir="Qwen3-VL-4B-Instruct", patterns=TEACHER_PATTERNS,
                     hint="bf16 safetensors (~8.3 GB) + tokenizer + config; the vision tower is the one "
                     "mmproj-qwen3vl-4b-it-f16.gguf was converted from."),
        ]
        presets = ["flickr30k"]
        if project is not None:
            presets = list((project.config.get("corpus") or {}).get("image_presets") or [])
        for pid in presets:
            spec = IMAGE_PRESETS.get(pid)
            if not spec:
                continue
            mats.append(Material(f"images_{pid}", f"Images: {spec['title']}", "hf_snapshot", required=True,
                                 repo=str(spec["repo"]), repo_type=str(spec.get("repo_type", "dataset")),
                                 subdir=str(spec["subdir"]), patterns=list(spec["patterns"]),
                                 hint=str(spec.get("hint", ""))))
        return mats

    def teacher_dir(self, project) -> pathlib.Path:
        return self.material_path(project, self.material("teacher"))

    def build_teacher(self, project, device, gpu_mem, cpu_mem, log):
        from ..qwen3vl_teacher import Qwen3VLTeacher

        pc = project.config.get("precompute") or {}
        gpu_gib = float(str(gpu_mem).replace("GiB", "")) if isinstance(gpu_mem, str) else float(gpu_mem or 0)
        return Qwen3VLTeacher(self.teacher_dir(project), device, gpu_gib, pc.get("teacher_mode", "auto"), log)

    def build_mock_teacher(self, log):
        from ..qwen3vl_teacher import MockVisionTeacher

        return MockVisionTeacher(self.out_dim, self.vis_dim, log)

    def export_kv(self, project) -> Dict[str, Any]:
        return {"adapter.qwen_hidden_source": "final_norm",
                "adapter.teacher": HF_REPO,
                "adapter.teacher_tap": "final_norm"}

    def engine_command(self, student: str, adapter: str, extras: Optional[Dict[str, str]] = None) -> str:
        return (f"ggk diffuser engine -- --diffusion-model mageflow-edit-turbo-nvfp4.gguf "
                f"--vae pig_mageflow_vae_fp32-f16.gguf --llm {student} --llm-adapter {adapter} "
                f"--llm_vision mmproj-qwen3vl-4b-it-f16.gguf --ref-image sheep.png "
                f"-p \"a sheep in sunglasses\" --cfg-scale 1.0 --steps 4 --sampling-method euler "
                f"--diffusion-fa -o out.png")

    def sample_bytes(self, config: Dict[str, Any]) -> int:
        c = config.get("corpus") or {}
        n_img = float(c.get("n_image", 0)) + float(c.get("n_two", 0))
        n_txt = float(c.get("n_text", 0))
        tot = max(1.0, n_img + n_txt)
        # ~235 tokens + 144 vision rows per image sample, ~60 tokens per text sample
        per_img = 235 * (self.out_dim + self.in_dim) * 2 + 144 * self.vis_dim * 2 + 235 * 4
        per_txt = 60 * (self.out_dim + self.in_dim) * 2 + 60 * 4
        return int((per_img * n_img + per_txt * n_txt) / tot)
