"""PixArt / T5-XXL pack — the ./trainer recipe with pig_clip as the student.

PixArt's DiT cross-attends to T5-XXL v1.1 encoder output over a 120-slot
caption window (`caption_projection` takes 4096-dim rows; pads carry a
-10000 attention bias, so only the real tokens matter).  This pack distills
that conditioning into pig_clip + a seeded resampler: 120 learned queries,
each seeded with the T5 sentencepiece embedding of its slot, cross-attend
over the student's final-norm states and are projected to 4096.  The engine
keeps the T5 *tokenizer* (for the pad mask, prompt weights and the seed ids)
but never loads the 9.5 GB encoder:

    --llm pig_clip-<quant>.gguf --llm-adapter pig_t5_adapter-f16.gguf

replaces `--t5xxl t5xxl.gguf` for pixart-*.gguf.

Contract (trainer/adapter.py, verified against the engine at cos 0.999999):
  * student ids = the raw prompt through the Qwen BPE, no special tokens,
    an empty prompt = the single pad token;
  * seed ids = T5 sentencepiece + EOS, pad-0 to 120 (what PixArtT5Embedder
    builds); query_i = query[i] + t5_embed[id_i];
  * target = T5EncoderModel.last_hidden_state at the real slots; the loss
    is masked to them (whitened MSE + 0.5·(1−cos) on RAW targets with the
    sigma floor 0.03 that fixed T5's ~120 near-constant dims);
  * export layout == the shipped pig_t5_adapter-f16.gguf (adapter.query f32,
    adapter.t5_embed.weight f16 [32128, width], explicit q/k/v/o linears).
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional

from .base import Material, TrainerPack, student_materials

HF_REPO = "callgg/t5-v1_1-xxl-encoder-bf16"
TEACHER_PATTERNS = ["config.json", "generation_config.json", "model.safetensors", "special_tokens_map.json",
                    "spiece.model", "tokenizer_config.json"]
T5_WINDOW = 120          # PixArt caption length (y_embedding [4096, 120])
T5_VOCAB = 32128         # T5 v1.1 embedding rows


class T5XXLPixArtPack(TrainerPack):
    id = "t5xxl_pixart"
    title = "PixArt (T5-XXL v1.1 encoder)"
    description = ("Replaces the 4.7B T5-XXL text encoder of PixArt (--t5xxl) with pig_clip + a "
                   "120-query resampler seeded with the T5 token ids (the ./trainer recipe). Text-only; "
                   "the T5 tokenizer stays at inference, the encoder is never loaded.")
    adapter_kind = "seeded_resampler"
    out_dim = 4096
    num_queries = T5_WINDOW
    seed_vocab = T5_VOCAB
    max_len_student = 256
    teacher_control = "teacher_mode"
    default_name = "t5_adapter"
    shard_contract = "t5xxl_pixart/1"
    eval_keys = ["cos", "cos_rms", "rel_mse", "worst_row_cos", "cos_eos", "roundtrip_cos", "trained_steps",
                 "val_batches"]
    hints = {
        "corpus": ("Text only: the teacher is a text encoder. Prompts longer than the 120-slot T5 window are "
                   "truncated (max_chars 1200 keeps most under it). ~1% empty prompts cover the empty negative; "
                   "PixArt's own null caption (y_embedding) stays untouched. The reference runs used 117k–1.6M "
                   "unique prompts (Gustavosta + Midjourney + DiffusionDB)."),
        "precompute": ("T5-XXL is one dense 9.5 GB bf16 encoder: gpu keeps it resident (needs a ~12 GiB budget), "
                       "offload streams the 24 layers from RAM through the GPU (a 6 GB card; RAM must hold the "
                       "weights). Only the real slots of each prompt are stored (~40 rows × 4096 on average, "
                       "~0.4 MB per prompt). Token budget counts real T5 tokens per batch."),
        "train": ("trainer recipe: width 1024, depth 6, 20k steps, batch 32–64, lr 2e-4, warmup 1k, cosine to 10%, "
                  "whitened MSE + 0.5·(1−cos) masked to the real slots, sigma floor 0.03 (T5-XXL has ~120 "
                  "near-constant dims that otherwise carry half the loss as bf16 noise). Reference: val cos 0.827 "
                  "at 12.5k steps on 117k prompts; the curve flattens around 0.80–0.83 (the 0.6B student's "
                  "representation is the ceiling, not data)."),
        "eval": ("Judge by cos over the real slots (pads are never supervised or consumed) and rel_mse. cos_rms "
                 "is the per-row-normalized view; cos_eos isolates the EOS slot (~0.99 once the seed path works). "
                 "Baselines: predicting the per-dim mean scores ~0.42, the shipped pig_t5_adapter 0.82 on the "
                 "reference val prompts — anything under ~0.6 means the seed or tokenizer path is broken. "
                 "roundtrip_cos compares the exported f16 GGUF against the checkpoint (expect ≥ 0.999)."),
        "output": ("Use pig_clip at f16 or q8_0: the 4-bit student's hidden states lose ~0.09 cosine before the "
                   "adapter even runs (trainer measurements). The adapter is ~1024-wide, 176 tensors, ~350 MB f16."),
    }

    def defaults(self) -> Dict[str, Any]:
        return {
            "corpus": {"n_train": 120000, "n_val": 2048, "presets": ["gustavosta", "midjourney"]},
            "precompute": {"shard_size": 1024, "val_shard_size": 512, "teacher_batch": 0, "tok_budget": 0,
                           "student_batch": 0, "teacher_mode": "auto"},
            "train": {"width": 1024, "depth": 6, "steps": 20000, "batch_size": 32, "sigma_floor": 0.03},
            "export": {"export_sigvq": False},
        }

    def materials(self, project=None) -> List[Material]:
        return student_materials() + [
            Material("teacher", "Teacher: T5-XXL v1.1 encoder (bf16)", "hf_snapshot", required=True,
                     repo=HF_REPO, subdir="t5-v1_1-xxl-encoder-bf16", patterns=TEACHER_PATTERNS,
                     hint="Encoder-only bf16 safetensors (~9.5 GB) + the sentencepiece tokenizer PixArt uses. "
                     "Resident needs ~12 GiB of VRAM; otherwise streamed from RAM."),
        ]

    def format_prompt(self, text: str) -> str:
        # the engine's PixArt path hands the LLM the raw prompt (no template)
        return " ".join((text or "").split())

    def teacher_dir(self, project) -> pathlib.Path:
        return self.material_path(project, self.material("teacher"))

    def build_teacher(self, project, device, gpu_mem, cpu_mem, log):
        from ..t5_teacher import T5Teacher

        pc = project.config.get("precompute") or {}
        gpu_gib = float(str(gpu_mem).replace("GiB", "")) if isinstance(gpu_mem, str) else float(gpu_mem or 0)
        return T5Teacher(self.teacher_dir(project), device, gpu_gib, self.num_queries,
                         pc.get("teacher_mode", "auto"), log)

    def build_mock_teacher(self, log):
        from ..t5_teacher import MockT5Teacher

        return MockT5Teacher(self.out_dim, self.num_queries, self.seed_vocab, log)

    def export_kv(self, project) -> Dict[str, Any]:
        return {"adapter.t5_vocab": self.seed_vocab,
                "adapter.qwen_hidden_source": "final_norm",
                "adapter.target": "t5_v1_1_xxl encoder last_hidden_state, 120-slot PixArt window",
                "adapter.teacher": HF_REPO}

    def engine_command(self, student: str, adapter: str, extras: Optional[Dict[str, str]] = None) -> str:
        return (f"ggk diffuser engine -- --diffusion-model pixart-nvfp4.gguf "
                f"--vae pig_pixart_vae_fp16-f16.gguf --llm {student} --llm-adapter {adapter} "
                f"-p \"close-up portrait of a young lady\" --diffusion-fa -s 42 -o out.png")

    def sample_bytes(self, config: Dict[str, Any]) -> int:
        # ~40 real T5 slots + ~30 student tokens per prompt, bf16 rows + the id window
        return 40 * self.out_dim * 2 + 30 * self.in_dim * 2 + self.num_queries * 4
