"""Ming-Image 0.1 pack (Design and Design-Layer) — replaces the Ling-mini-2.0 MLLM.

Ming-Image conditions its Z-Image-style DiT on two things its 16B Bailing-MoE
text encoder produces (ggk `--llm ming_image_0.1_ling_mini_2.0-*.gguf`):

  * 256 caption tokens: learned query tokens appended to the prompt, read
    back after the thinker and refined by a 28-layer Qwen2 connector (2560);
  * one "direct" row per prompt token: hidden states 5 / 12 / 20 of every
    prompt token (system prefix, input image block, text, suffix)
    concatenated, RMSNorm + Linear to the DiT width (3840).

This pack distills both into pig_clip + a `ming_image` adapter (adapter.py
MingAdapter): a resampler whose query bank is the 256 caption rows plus one
seeded slot per Ling prompt token — the engine keeps the Ling tokenizer
(embedded in the adapter GGUF as `tokenizer_json`, like the text encoder
GGUF does) so the DiT sees exactly the token count and identity it was
trained on.  Editing / layer decomposition show the input image to the
adapter through the Ling Qwen2.5-VL vision tower + linear_proj (2048-d unit
vectors), exported from the same MLLM snapshot to `pig_ming_vision-*.gguf`:

    --llm pig_clip-<quant>.gguf --llm-adapter pig_ming_adapter-f16.gguf
    [--llm_vision pig_ming_vision-f16.gguf --ref-image ref.png]

The student reads the raw prompt text (Qwen BPE, no template); the vision
embeds enter the adapter directly (kv rows + per-slot query seeds), never
the student, so no cross-vocabulary vision_proj is needed.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np

from ..vision_data import IMAGE_PRESETS
from .base import Material, TrainerPack, student_materials

HF_REPO = "inclusionAI/Ming-Image-0.1-Design"
TEACHER_PATTERNS = ["mllm/config.json", "mllm/model-*.safetensors", "mllm/model.safetensors.index.json",
                    "mllm/preprocessor_config.json", "mllm/tokenizer.json", "mllm/tokenizer_config.json",
                    "mllm/special_tokens_map.json",
                    "connector/config.json", "connector/model-*.safetensors", "connector/model.safetensors.index.json",
                    "mlp/config.json", "mlp/model.safetensors"]


class MingImagePack(TrainerPack):
    id = "ming_image"
    title = "Ming-Image 0.1 Design / Layer (Ling-mini-2.0 MLLM)"
    description = ("Replaces the 16B Ling-mini-2.0 (Bailing-MoE) text encoder of Ming-Image 0.1 Design and "
                   "Design-Layer with pig_clip + a caption-bank / Ling-token-seeded resampler adapter. Text-to-image "
                   "needs only the adapter; editing and layer decomposition add the exported pig_ming_vision GGUF.")
    adapter_kind = "ming_image"
    in_dim = 1024
    out_dim = 3840                 # direct rows (DiT width)
    cap_dim = 2560                 # caption rows (cap_embedder input)
    vis_dim = 2048                 # linear_proj output (unit norm)
    num_queries = 256              # caption bank rows
    seed_vocab = 157184            # Ling tokenizer vocab (seed table rows)
    seed_rank = 512                # default train.seed_rank (PCA of the 2048-d embedding table)
    max_slots = 1280               # default train.max_slots (Ling prompt tokens: ~30 prefix + 578 image + text)
    needs_images = True
    teacher_control = "teacher_mode"
    budget_scale = 0.5
    default_name = "ming_adapter"
    max_len_student = 512
    shard_contract = "ming_image/1"
    eval_keys = ["cos_centered", "cos_centered_cap", "cos_centered_dir", "cos_rms", "rel_mse", "roundtrip_cos",
                 "worst_row_cos_rms", "trained_steps", "val_batches"]
    hints = {
        "corpus": ("Text-only prompts (text-to-image, the cheap majority) plus (image, instruction) samples for editing "
                   "and layer decomposition — the Ming template shows the text encoder ONE input image, resized to "
                   "~672x672 on a 28 px grid (576 image tokens whatever the source size), so two-image samples are "
                   "off (n_two 0). Instructions are synthesized from the captions as in the MageFlow pack; layer "
                   "prompts ('Decompose this image into N layers ...') are ordinary multi-line text."),
        "precompute": ("The 16B Bailing-MoE thinker (~32 GB bf16) + the Qwen2.5-VL tower + the Qwen2 connector: gpu keeps "
                       "the thinker resident over the CUDA devices (~36 GiB in total), offload streams it layer by layer from "
                       "RAM through the GPU (RAM must hold ~34 GB). Per sample the shards keep the student rows, 256 caption "
                       "rows (bf16 2560), one 3840-wide row per Ling prompt token and, for image samples, the 576 raw "
                       "vision embeds: ~0.3 MB per text sample, ~7 MB per image sample. The precompute also fits the frozen "
                       "seed table (PCA of the Ling embedding table) once per project. Budget the corpus by teacher time: an image "
                       "sample costs the thinker ~610 tokens against ~35 for a text prompt; the default 8k image + 16k text "
                       "samples are 94 train shards (the earlier 30k + 40k came to 274 shards, over 18 h)."),
        "train": ("Resampler recipe (width 1024, depth 6, 20k steps, batch 32, lr 2e-4, warmup 1k, cosine to 10%): whitened "
                  "MSE + 0.5·(1−cos) on standardized targets for both heads, the direct head masked to the real Ling slots. "
                  "train.max_slots caps the Ling prompt length the adapter (and the engine) accept; train.seed_rank is "
                  "the seed table rank (file size ~157k x rank x 2 bytes). No reference run exists yet for this pack."),
        "eval": ("Judge by cos_centered (mean of the caption and direct heads' centred cosines; plain cosine is inflated by "
                 "the rows' shared offsets) and rel_mse; cos_rms is what the DiT's RMSNorm sees. cos_centered_dir covers "
                 "the prompt rows the DiT reads per token, cos_centered_cap the 256 connector tokens."),
        "output": ("The adapter GGUF embeds the Ling tokenizer (12 MB) so ggk can build the seed ids without the 16B "
                   "encoder. The companion pig_ming_vision-f16.gguf (Qwen2.5-VL tower + linear_proj from the same "
                   "snapshot, ~1.3 GB; export.vision_quant q8_0 halves it) is what --llm_vision takes for editing / "
                   "layers — text-to-image does not need it. Needs ggk 0.7.7+."),
    }

    def defaults(self) -> Dict[str, Any]:
        return {
            # 8k image + 16k text samples = 94 train shards of 256 (the first 30k + 40k default came
            # to 274 shards, > 18 h of teacher time: the 16B thinker sees ~610 tokens per image sample)
            "corpus": {"image_presets": ["flickr30k"], "image_folders": [], "n_image": 8000, "n_two": 0,
                       "n_text": 16000, "n_val_image": 256, "n_val_text": 384, "empty_frac": 0.015,
                       "presets": ["gustavosta", "midjourney"]},
            "precompute": {"shard_size": 256, "val_shard_size": 128, "teacher_batch": 0, "tok_budget": 0,
                           "student_batch": 0, "student_tok_budget": 0, "vis_tok_budget": 0, "teacher_mode": "auto"},
            "train": {"width": 1024, "depth": 6, "steps": 20000, "batch_size": 32, "seed_rank": self.seed_rank,
                      "max_slots": self.max_slots},
            "export": {"export_sigvq": False, "export_vision": True, "vision_quant": "f16"},
        }

    def val_split_labels(self, project=None) -> List[str]:
        return ["caption", "direct"]

    def materials(self, project=None) -> List[Material]:
        mats = student_materials() + [
            Material("teacher", "Teacher: Ming-Image 0.1 MLLM (mllm + connector + mlp)", "hf_snapshot", required=True,
                     repo=HF_REPO, subdir="Ming-Image-0.1-Design", patterns=TEACHER_PATTERNS,
                     hint="Ling-mini-2.0 Bailing-MoE + Qwen2.5-VL tower (bf16, ~32 GB), the Qwen2 connector (~5.8 GB) "
                     "and the projections (~120 MB) + tokenizer. The DiT and VAE are not needed."),
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

    def format_prompt(self, text: str) -> str:
        # the student reads the raw prompt (the engine's Ming adapter path hands pig_clip the text as is)
        return text or ""

    def teacher_dir(self, project) -> Optional[pathlib.Path]:
        return self.material_path(project, self.material("teacher"))

    def build_teacher(self, project, device, gpu_mem, cpu_mem, log):
        from ..ming_teacher import MingTeacher

        pc = project.config.get("precompute") or {}
        gpu_gib = float(str(gpu_mem).replace("GiB", "")) if isinstance(gpu_mem, str) else float(gpu_mem or 0)
        return MingTeacher(self.teacher_dir(project), device, gpu_gib, pc.get("teacher_mode", "auto"), log)

    def build_mock_teacher(self, log, project=None):
        from ..ming_teacher import MockMingTeacher

        root = self.teacher_dir(project) if project is not None else None
        return MockMingTeacher(root if root and pathlib.Path(root).is_dir() else None, log)

    def export_kv(self, project) -> Dict[str, Any]:
        return {"adapter.qwen_hidden_source": "final_norm",
                "adapter.teacher": HF_REPO,
                "adapter.target": "ming_image caption tokens (connector proj_out, 256 x 2560) + directvlm rows (3840 per Ling token)",
                "adapter.student_prompt": "raw_text",
                "adapter.vision": "ling qwen2.5-vl tower + linear_proj, L2-normalised (pig_ming_vision)"}

    def export_extra_tensors(self, project, log) -> Dict[str, Any]:
        """The Ling tokenizer.json as F16 byte values (the engine's read_embedded_json layout)."""
        root = self.teacher_dir(project)
        tj = pathlib.Path(root) / "mllm" / "tokenizer.json" if root else None
        if tj is None or not tj.is_file():
            log("export: WARNING mllm/tokenizer.json is not downloaded; the adapter GGUF will carry no Ling tokenizer "
                "(the engine then needs the tokenizer from the Ming text encoder GGUF)")
            return {}
        raw = np.frombuffer(tj.read_bytes(), dtype=np.uint8).astype(np.float16)
        log(f"export: embedding {tj.name} ({raw.size / 1e6:.1f} MB) as tokenizer_json")
        return {"tokenizer_json": raw}

    def extra_exports(self, project, log) -> List[pathlib.Path]:
        ex = project.config.get("export") or {}
        if not ex.get("export_vision", True):
            return []
        root = self.teacher_dir(project)
        mllm = pathlib.Path(root) / "mllm" if root else None
        if not mllm or not (mllm / "model.safetensors.index.json").is_file():
            log("vision: the MLLM snapshot is not downloaded, skipping the pig_ming_vision export")
            return []
        quant = str(ex.get("vision_quant") or "f16").lower()
        out = project.output_dir / f"{project.export_base()}_vision-{quant}.gguf"
        newest = max(f.stat().st_mtime for f in mllm.glob("model-*.safetensors"))
        if out.is_file() and out.stat().st_mtime >= newest:
            log(f"vision: {out.name} up to date")
            return [out]
        from ..ming_vision_export import export_ming_vision

        export_ming_vision(mllm, out, quant, log)
        return [out]

    def engine_command(self, student: str, adapter: str, extras: Optional[Dict[str, str]] = None) -> str:
        vis = (extras or {}).get("vision", "pig_ming_vision-f16.gguf")
        return (f"ggk diffuser engine -- --diffusion-model ming-image-0.1-design-nvfp4.gguf "
                f"--vae pig_ming_image_vae_bf16.gguf --llm {student} --llm-adapter {adapter} "
                f"-p \"a sheep in sunglasses\" --steps 12 --diffusion-fa --offload-to-cpu -o out.png"
                f"   (editing: add --llm_vision {vis} --ref-image ref.png; "
                f"layers: the -layer DiT + --prompt-file spec.txt -W 512 -H 512)")

    def sample_bytes(self, config: Dict[str, Any]) -> int:
        c = config.get("corpus") or {}
        n_img = float(c.get("n_image", 0)) + float(c.get("n_two", 0))
        n_txt = float(c.get("n_text", 0))
        tot = max(1.0, n_img + n_txt)
        cap = self.num_queries * self.cap_dim * 2
        per_txt = cap + 70 * self.out_dim * 2 + 40 * self.in_dim * 2
        per_img = cap + (576 + 70) * self.out_dim * 2 + 576 * self.vis_dim * 2 + 40 * self.in_dim * 2
        return int((per_img * n_img + per_txt * n_txt) / tot)
