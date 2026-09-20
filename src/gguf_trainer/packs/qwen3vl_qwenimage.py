"""Qwen-Image 2.1 pack — replaces the Qwen3-VL-8B-Instruct text stack.

Qwen-Image 2.1 conditions its DiT on Qwen3-VL-8B-Instruct: the LAST decoder
layer before the final RMSNorm, with the reference images of an edit read
through the vision tower (`--llm qwen3vl-8b-it-<quant>.gguf --llm_vision
mmproj-qwen3vl-8b-it-f16.gguf`).  This pack distills that conditioning into
pig_clip + a token-aligned adapter with a vision extension, so the same
generation runs with

    --llm pig_clip-<quant>.gguf --llm-adapter pig_qwen3vl_8b_adapter-f16.gguf
    [--llm_vision mmproj-qwen3vl-8b-it-f16.gguf  -r ref.png]

It is the 8B sibling of the MageFlow pack (same adapter kind, same student,
same frozen-vision_proj / trained-vis_in bridge, here 4096 <-> 1024), with
three differences that come from the engine contract
(vision_data.QwenImage21Contract) rather than from the model size:

  * the teacher is the UNREDUCED Hugging Face model — deepstack on, real
    M-RoPE image positions, pre-norm tap (qwen3vl_teacher.Qwen3VLFullTeacher).
    The adapter only ever sees the mmproj's main output, so whatever
    deepstack told the teacher is learned from that;
  * only the text rows the DiT consumes are trained and scored: the DiT
    substitutes reference latents for the rows under the image slots, and
    rows before the template start are dropped.  Target statistics are taken
    over the same rows (a pre-norm tap has attention-sink rows that would
    otherwise own the whitening);
  * the vision token count follows the render size (4 reference-latent
    tokens per slot): 144 tokens at 384^2 up to 1024 at 1024^2, so samples
    are drawn over `corpus.ref_areas`.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

from ..vision_data import IMAGE_PRESETS, QI21_DEFAULT_REF_AREAS, QwenImage21Contract
from .base import Material, TrainerPack, student_materials

HF_REPO = "Qwen/Qwen3-VL-8B-Instruct"
TEACHER_PATTERNS = ["config.json", "generation_config.json", "model.safetensors.index.json", "model-*.safetensors",
                    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                    "merges.txt", "chat_template.json"]


class Qwen3VLQwenImagePack(TrainerPack):
    id = "qwen3vl_qwenimage"
    title = "Qwen-Image 2.1 (Qwen3-VL-8B-Instruct + mmproj)"
    description = ("Replaces the Qwen3-VL-8B-Instruct text stack of Qwen-Image 2.1 with pig_clip + a "
                   "token-aligned adapter with a vision extension. Text-to-image needs only the adapter; "
                   "editing pairs it with the unchanged mmproj-qwen3vl-8b-it-f16.gguf vision encoder.")
    adapter_kind = "token_aligned_vision"
    out_dim = 4096
    vis_dim = 4096
    num_queries = 0
    needs_images = True
    teacher_control = "teacher_mode"
    budget_scale = 0.5             # twice the 4B teacher's weights and activations on the same card
    default_name = "qwen3vl_8b_adapter"
    # room for two 1024^2 references (2 x ~1040 slots + markers) + a long prompt, so corpus.two_image_max_area
    # can be raised to the render area; the default keeps two-image samples at 512^2 each for cost
    max_len_student = 2304
    shard_contract = "qwen3vl_qwenimage/1"
    eval_keys = ["cos_slice", "cos_edit", "cos_t2i", "cos", "rel_mse", "worst_sample_cos_slice", "roundtrip_cos",
                 "trained_steps", "val_samples"]
    hints = {
        "corpus": ("Text-only prompts (the text-to-image path, and the cheap majority) plus (image, instruction) "
                   "samples for editing, all on the one Qwen-Image 2.1 template. Each image sample draws its reference "
                   "area from corpus.ref_areas — the engine resizes a reference to the render area, so the adapter has "
                   "to see 144-token (384²) through 1024-token (1024²) references. Two-image samples stay at or below "
                   "corpus.two_image_max_area each (the engine gives EVERY reference the full render area, so raise "
                   "it to cover two-reference edits above 512²). Instructions are synthesized from the captions as in the MageFlow pack."),
        "precompute": ("Vision tower + the full Qwen3-VL-8B text stack (deepstack on, ~16.4 GB bf16) + pig_clip, once per "
                       "shard. Resident on a 24 GB+ card; otherwise streamed through the GPU with accelerate cpu_offload "
                       "(RAM must hold it) — expect that to be slow at 1024-token references. Shards keep every student "
                       "row and the raw vision embeds but only the teacher rows outside the image slots: about 0.6 MB per "
                       "text sample and 3–11 MB per image sample depending on the reference area."),
        "train": ("MageFlow recipe (width 1024, depth 4, 20k steps, batch 32, lr 2e-4, whitened MSE + 0.5·(1−cos), "
                  "out_proj zero-init, vision_proj frozen) with the loss restricted to the text rows the DiT reads. "
                  "Batches with 1024-token references are heavy: grad checkpointing turns itself on below 12 GB, and "
                  "batch 16 is a reasonable fallback. No reference run exists yet for this pack."),
        "eval": ("Judge by cos_slice (rows ≥ 14 outside the image slots — exactly what the DiT consumes), split into "
                 "cos_t2i (text-only samples) and cos_edit (text rows of image samples: how well what the teacher read "
                 "off the image — deepstack included — survives the mmproj-main-output-only path). A pre-norm tap has "
                 "a few very large dimensions, so plain cosine runs high; compare runs by rel_mse as well."),
        "output": ("Text-to-image works with every pig_clip quantization; editing wants q8_0 or better. The mmproj file is "
                   "the teacher's own and is not exported here. Needs ggk 0.6.6+ (Qwen-Image 2.1 with the adapter "
                   "reference path)."),
    }

    def defaults(self) -> Dict[str, Any]:
        return {
            "corpus": {"image_presets": ["flickr30k"], "image_folders": [], "n_image": 30000, "n_two": 2000,
                       "n_text": 40000, "n_val_image": 512, "n_val_text": 512, "empty_frac": 0.015,   # the val set lives in RAM (~4 GB)
                       "presets": ["gustavosta", "midjourney"],
                       "ref_areas": [[a, w] for a, w in QI21_DEFAULT_REF_AREAS],
                       "two_image_max_area": 512 * 512},
            "precompute": {"shard_size": 256, "val_shard_size": 128, "teacher_batch": 0, "tok_budget": 0,
                           "student_batch": 0, "student_tok_budget": 0, "vis_tok_budget": 0, "teacher_mode": "auto"},
            "train": {"width": 1024, "depth": 4, "steps": 20000, "batch_size": 32},
            "export": {"export_sigvq": False},
        }

    def contract(self, project=None):
        c = (project.config.get("corpus") or {}) if project is not None else {}
        return QwenImage21Contract(ref_areas=c.get("ref_areas") or None,
                                   two_image_max_area=int(c.get("two_image_max_area") or 512 * 512))

    def materials(self, project=None) -> List[Material]:
        mats = student_materials() + [
            Material("teacher", "Teacher: Qwen3-VL-8B-Instruct", "hf_snapshot", required=True,
                     repo=HF_REPO, subdir="Qwen3-VL-8B-Instruct", patterns=TEACHER_PATTERNS,
                     hint="bf16 safetensors (~17.5 GB) + tokenizer + config; the vision tower is the one "
                     "mmproj-qwen3vl-8b-it-f16.gguf was converted from."),
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
        from ..qwen3vl_teacher import Qwen3VLFullTeacher

        pc = project.config.get("precompute") or {}
        gpu_gib = float(str(gpu_mem).replace("GiB", "")) if isinstance(gpu_mem, str) else float(gpu_mem or 0)
        return Qwen3VLFullTeacher(self.teacher_dir(project), device, gpu_gib, pc.get("teacher_mode", "auto"), log)

    def build_mock_teacher(self, log):
        from ..qwen3vl_teacher import MockVisionTeacher

        return MockVisionTeacher(self.out_dim, self.vis_dim, log)

    def export_kv(self, project) -> Dict[str, Any]:
        return {"adapter.qwen_hidden_source": "final_norm",          # the STUDENT side: pig_clip final norm
                "adapter.teacher": HF_REPO,
                "adapter.teacher_tap": "pre_norm_last_layer",
                "adapter.teacher_deepstack": True,
                "adapter.vision_contract": "qwen_image_2_1",
                "adapter.supervised_rows": "consumed_text"}

    def engine_command(self, student: str, adapter: str, extras: Optional[Dict[str, str]] = None) -> str:
        return (f"ggk diffuser engine -- --diffusion-model qwen-image-2.1-<quant>.gguf "
                f"--vae qwen_image_2.1_vae-f16.gguf --llm {student} --llm-adapter {adapter} "
                f"-p \"a red fox sitting in snow\" --cfg-scale 1.0 --steps 40 -W 1024 -H 1024 "
                f"--diffusion-fa -o out.png"
                f"   (editing: add --llm_vision mmproj-qwen3vl-8b-it-f16.gguf -r ref.png)")

    def sample_bytes(self, config: Dict[str, Any]) -> int:
        c = config.get("corpus") or {}
        n_img = float(c.get("n_image", 0)) + float(c.get("n_two", 0))
        n_txt = float(c.get("n_text", 0))
        tot = max(1.0, n_img + n_txt)
        areas = [(float(a), float(w)) for a, w in (c.get("ref_areas") or QI21_DEFAULT_REF_AREAS) if float(w) > 0]
        wsum = sum(w for _, w in areas) or 1.0
        n_vis = sum(a / 1024.0 * w for a, w in areas) / wsum        # one slot per 32x32 px
        n_txt_rows = 75.0                                           # 14 + markers + instruction + suffix
        # student rows everywhere + raw vision embeds + teacher rows outside the slots
        per_img = (n_vis + n_txt_rows) * self.in_dim * 2 + n_vis * self.vis_dim * 2 + n_txt_rows * self.out_dim * 2
        per_txt = 70 * (self.out_dim + self.in_dim) * 2
        return int((per_img * n_img + per_txt * n_txt) / tot)
