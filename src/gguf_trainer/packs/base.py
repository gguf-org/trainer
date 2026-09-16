"""The pack interface.  A pack knows

  * which materials a run needs (HF snapshots of models or datasets, local
    GGUF files) and where they live inside the project;
  * how the target engine formats a prompt (and, for vision packs, how it
    preprocesses reference images and splices their embeddings);
  * how to build the teacher (produces the target rows per sample);
  * the adapter kind and geometry (student dim -> target dim, query rows
    or token-aligned, vision extension);
  * what to write into the exported GGUF, plus optional extra exports
    (e.g. the paired vision encoder);
  * its own defaults for the corpus / precompute / training settings.

The student side (pig_clip = a native train/fine-tune GGUF in the Qwen3 layout
+ the callgg/pig-clip-tokenizer snapshot it shares the tokenizer with) is
shared by every pack.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Dict, List, Optional

ADAPTER_KINDS = ("resampler", "seeded_resampler", "token_aligned_vision")


@dataclasses.dataclass
class Material:
    id: str
    title: str
    kind: str                      # hf_snapshot | local_file
    required: bool = True
    repo: str = ""                 # hf_snapshot
    patterns: List[str] = dataclasses.field(default_factory=list)
    subdir: str = ""               # destination under <project>/materials
    suffixes: List[str] = dataclasses.field(default_factory=list)  # local_file
    hint: str = ""
    default_path: str = ""         # local_file: a path to try first
    repo_type: str = "model"       # hf_snapshot: model | dataset

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def student_materials() -> List[Material]:
    """The two materials every pack shares: the pig_clip GGUF and the
    tokenizer/config snapshot it was trained with."""
    return [
        Material("student", "Student: pig_clip GGUF", "local_file", required=True,
                 suffixes=[".gguf"], hint="pig_clip-f16.gguf (native train/fine-tune). f16 is what "
                 "the adapter is trained against; quantized files are dequantized on load."),
        Material("student_tokenizer", "Student tokenizer + config (callgg/pig-clip-tokenizer)", "hf_snapshot",
                 required=True, repo="callgg/pig-clip-tokenizer", subdir="pig-clip-tokenizer",
                 # the six files the repo ships (no special_tokens_map.json: a
                 # pattern that matches nothing would never count as complete)
                 patterns=["config.json", "generation_config.json", "tokenizer.json",
                           "tokenizer_config.json", "vocab.json", "merges.txt"],
                 hint="~11 MB: the tokenizer + config pig_clip was trained with. "
                 "The pig_clip GGUF carries no tokenizer."),
    ]


class TrainerPack:
    id = "base"
    title = "base"
    description = ""
    adapter_kind = "resampler"     # resampler (query rows) | seeded_resampler (query rows seeded with the
    #                                teacher tokenizer's ids, pads unsupervised) | token_aligned_vision
    #                                (position i -> i, mmproj ext)
    in_dim = 1024                  # pig_clip hidden size
    out_dim = 0
    vis_dim = 0                    # token_aligned_vision: raw mmproj embed dim
    num_queries = 0                # resampler kinds: query rows (= the teacher window for seeded)
    seed_vocab = 0                 # seeded_resampler: teacher tokenizer vocab (t5_embed rows)
    needs_images = False           # corpus = (image(s), instruction) samples
    # GUI: which precompute placement control applies to the teacher
    #   "placement"    accelerate device_map over GPUs + CPU RAM (LLaDA)
    #   "teacher_mode" resident | streamed through the GPU (single dense model)
    teacher_control = "placement"
    default_name = "adapter"       # project / GGUF name suggestion (pig_ prefix added)
    max_len_student = 512
    # Bumped whenever the teacher targets change meaning; shards written
    # under another contract are stale and get recomputed (precompute.py).
    shard_contract = "1"
    # GUI copy per pack: corpus | precompute | train | eval | output
    hints: Dict[str, str] = {}
    # eval.json keys shown first in the Output tab
    eval_keys: List[str] = []

    def defaults(self) -> Dict[str, Any]:
        """Config overrides merged over project.DEFAULT_CONFIG at create()."""
        return {}

    def describe(self, project=None) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "description": self.description,
                "adapter_kind": self.adapter_kind, "needs_images": self.needs_images,
                "in_dim": self.in_dim, "out_dim": self.out_dim, "vis_dim": self.vis_dim,
                "num_queries": self.num_queries, "seed_vocab": self.seed_vocab,
                "teacher_control": self.teacher_control, "default_name": self.default_name,
                "sample_bytes": self.sample_bytes(project.config if project is not None else self.defaults()),
                "hints": dict(self.hints), "eval_keys": list(self.eval_keys),
                "materials": [m.to_dict() for m in self.materials(project)],
                "engine_command": self.engine_command("<pig_clip.gguf>", "<adapter.gguf>")}

    def materials(self, project=None) -> List[Material]:
        raise NotImplementedError

    def material(self, mat_id: str, project=None) -> Optional[Material]:
        return next((m for m in self.materials(project) if m.id == mat_id), None)

    def material_path(self, project, mat: Material) -> Optional[pathlib.Path]:
        """Resolve where a material lives for this project."""
        override = (project.config.get("materials") or {}).get(mat.id, {}).get("path")
        if override:
            return pathlib.Path(override).expanduser()
        if mat.kind == "hf_snapshot":
            return project.materials_dir / (mat.subdir or mat.repo.split("/")[-1])
        if mat.default_path and pathlib.Path(mat.default_path).expanduser().exists():
            return pathlib.Path(mat.default_path).expanduser()
        return None

    def format_prompt(self, text: str) -> str:
        """Resampler packs: the engine's template around a text prompt."""
        raise NotImplementedError

    def build_teacher(self, project, device, gpu_mem: str, cpu_mem: str, log):
        """resampler: callable(list[str]) -> [B, num_queries, out_dim] (cpu, bf16), with .text_len(str)
        seeded_resampler: .tokenize(list[str]) -> (ids [B, num_queries], len [B]) and
                          callable(ids, len) -> [B, max(len), out_dim] (cpu, bf16) (see t5_teacher.py)
        token_aligned_vision: an object with .encode_images() / .hidden() (see qwen3vl_teacher.py)"""
        raise NotImplementedError

    def build_mock_teacher(self, log):
        raise NotImplementedError

    def export_kv(self, project) -> Dict[str, Any]:
        """Extra GGUF key/values describing the target."""
        return {}

    def extra_exports(self, project, log) -> List[pathlib.Path]:
        """Optional companion exports after the adapter (e.g. vision encoder)."""
        return []

    def engine_command(self, student: str, adapter: str, extras: Optional[Dict[str, str]] = None) -> str:
        return ""

    def sample_bytes(self, config: Dict[str, Any]) -> int:
        """Rough shard bytes per corpus sample (GUI disk estimate)."""
        return self.num_queries * self.out_dim * 2 + 60 * self.in_dim * 2
