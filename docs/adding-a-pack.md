# Adding a pack

A **trainer pack** is one (teacher text encoder → pig_clip adapter) recipe.
Subclass `gguf_trainer.packs.base.TrainerPack`, then register the instance in
`gguf_trainer/packs/__init__.py`. The two shipped packs,
`packs/llada_image.py` and `packs/qwen3vl_mageflow.py`, are the references.

## What a pack declares

```python
from gguf_trainer.packs.base import Material, TrainerPack, student_materials

class MyPack(TrainerPack):
    id = "my_pack"                    # stored in project.json, GGUF metadata, shard meta
    title = "My model (its text encoder)"
    description = "..."
    adapter_kind = "resampler"        # or "token_aligned_vision"
    in_dim = 1024                     # pig_clip hidden size; leave it
    out_dim = 2560                    # target rows' width
    num_queries = 256                 # resampler only; 0 otherwise
    vis_dim = 0                       # token_aligned_vision: raw mmproj embed width
    needs_images = False              # True -> (image, instruction) corpus
    default_name = "my_adapter"       # project / GGUF name suggestion (pig_ prefix added)
    max_len_student = 512
    shard_contract = "my_pack/1"      # bump whenever the targets change meaning
    hints = {"corpus": "...", "precompute": "...", "train": "...", "eval": "...", "output": "..."}
    eval_keys = ["cos_centered", "rel_mse", ...]   # shown first in the Output tab
```

### Methods

| Method | Purpose |
| --- | --- |
| `defaults()` | Config overrides merged over `project.DEFAULT_CONFIG` when a project is created (corpus sizes, shard sizes, width / depth, export flags). |
| `materials(project=None)` | The `Material` list. Start with `student_materials()` and add the teacher and extras. Materials can depend on the project config (MageFlow adds one per selected image preset). |
| `format_prompt(text)` | Resampler packs: wrap a prompt in the engine's exact template. Used by both the teacher and the student tokenization. |
| `build_teacher(project, device, gpu_mem, cpu_mem, log)` | Return the teacher. Resampler: a callable `list[str] -> [B, num_queries, out_dim]` bf16 CPU tensor with a `.text_len(str)` method (used for the token budget). Token-aligned vision: an object with `encode_images(list[np.ndarray]) -> list[[n, vis_dim]]`, `hidden(batch) -> [B, L, out_dim]`, `embed_table()`, and a `tok` attribute for the tokenizer check. |
| `build_mock_teacher(log)` | A deterministic synthetic teacher for `GGUF_TRAINER_MOCK_TEACHER=1`. |
| `export_kv(project)` | Extra GGUF key/values (`str`, `bool`, `int`, `float`) describing the target. |
| `extra_exports(project, log)` | Companion files written after the adapter (the LLaDA pack exports SigVQ). Return the paths. |
| `engine_command(student, adapter, extras)` | The ggk command shown in the Output tab. |
| `sample_bytes(config)` | Rough shard bytes per sample for the GUI's disk estimate. |
| `material_path(project, mat)` | Usually inherited: override in `project.json` → `<project>/materials/<subdir>` → the material's `default_path`. |

### `Material`

```python
Material(id, title, kind,            # kind: "hf_snapshot" | "local_file"
         required=True,
         repo="org/name", repo_type="model" | "dataset",
         patterns=["text_encoder/*", "config.json"],   # allow_patterns; every pattern must match ≥ 1 file
         subdir="name",              # destination under <project>/materials
         suffixes=[".gguf"],         # local_file filter for Browse
         hint="shown in the GUI", default_path="")
```

A pattern that matches nothing would never count as complete, so list only
files the repository ships. Optional materials are downloaded by **Download
missing** only while `materials.wanted()` says so; extend that function if a
new optional material depends on a config flag.

## Choosing the adapter kind

* **`resampler`**: the target is a fixed number of rows per prompt
  (`[num_queries, out_dim]`), independent of the token count. The loss
  standardizes the targets per dimension and the export folds the
  standardization into `out_proj`. Use it when the DiT consumes a
  fixed-size conditioning (learned queries, pooled features).
* **`token_aligned_vision`**: one target row per student token, plus raw
  vision embeds spliced at known positions. The corpus is `(image,
  instruction)` jsonl; precompute runs a vision tower, the teacher text
  stack and the student per shard, and fits a frozen `vision_proj` from
  the two embedding tables (the vocabularies must match row for row). Use
  it when the DiT consumes the text encoder's per-token hidden states.

A new kind means extending `adapter.py`, `train.py`, `precompute.py`,
`shards.py`, `export.py` and `evaluate.py` together, plus the engine.

## Contract discipline

Everything the engine does to the prompt and the image must be replicated
bit-for-bit in the pack: template strings, special-token handling, image
resize and normalisation, which layer is tapped, which outputs are dropped.
The shipped packs verify their contracts against engine dumps
(`SD_DUMP_COND`, trainer8 `context.bin`); do the same before training at
scale, and bump `shard_contract` whenever a fix changes the targets so
existing shards are recomputed rather than resumed.

## Registering

```python
# gguf_trainer/packs/__init__.py
_PACKS = {p.id: p for p in (LLaDAImagePack(), Qwen3VLMageFlowPack(), MyPack())}
```

The GUI lists packs from `/api/status`; no frontend change is needed for a
pack that uses existing config fields. New fields need their inputs in
`static/index.html` / `static/app.js` and their defaults in
`project.DEFAULT_CONFIG`.
