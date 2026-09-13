# Projects and configuration

A **project** is one directory holding everything a run needs, so a reboot
loses nothing and `gguf-trainer run --project DIR` (or the GUI's Resume)
continues exactly where the pipeline stopped.

## Project folder layout

```
<project>/
  project.json          configuration (pack, materials, hyper-parameters)
  state.json            live pipeline state, written atomically by the pipeline
  pipeline.log          stdout/stderr of the detached pipeline process
  pipeline.pid          pid of the running pipeline (stale after a reboot)
  STOP                  request file: the pipeline saves and exits when it appears
  materials/            downloaded teacher / tokenizer / vision encoder / image datasets
    .download-<id>.{pid,log,status}   bookkeeping of detached download jobs
  data/                 train.txt + val.txt  (text packs)
                        train.jsonl + val.jsonl + images/  (image packs)
  shards/val/           val-00000.npz ...  + CONTRACT
  shards/train/         train-00000.npz ... + CONTRACT
  checkpoints/          last.pt, last.json, best.pt, best.json, log.csv
  checkpoints.stale-<time>/   archived after a shard-contract change
  eval.json             evaluation of the exported GGUF
  vision_proj.pt        vision packs: the frozen mmproj -> student map
```

Exported GGUFs land **next to** the project folder by default:
`test-trainer/pig_llada_adapter-f16.gguf` for the project
`test-trainer/llada_adapter/`. Every adapter trained under one parent folder
therefore ends up side by side. `export.output_dir` changes that. The
evaluation stays inside the project folder so projects sharing one output
folder never overwrite each other's `eval.json`.

## project.json

`project.json` is merged over the built-in defaults on every load, so a file
with only a few keys is valid. The pack's own defaults are applied once, when
the project is created. Full reference (values are the built-in defaults;
pack overrides are noted):

### Top level

| Key | Default | Meaning |
| --- | --- | --- |
| `pack` | `"llada_image"` | Trainer pack id: `llada_image` or `qwen3vl_mageflow`. |
| `name` | `"pig_llada_adapter"` | Export name; the GGUF is `<name>-f16.gguf`. Names ending in `_adapter` share the prefix with companion exports (`pig_llada_sigvq-f16.gguf`). |
| `auto_resume` | `true` | Allow `gguf-trainer serve --auto-resume` to relaunch this project after an interruption. |
| `hf_token` | `""` | Hugging Face token for downloads and datasets. |
| `materials` | `{}` | Per-material overrides: `{"<id>": {"path": "..."}}`. Written by Browse and by automatic linking. |

### corpus

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | `"presets"` | `presets` (datasets + extra files) or `files` (use `train_file` / `val_file` as they are). Text packs only. |
| `presets` | `["gustavosta", "midjourney"]` | Prompt dataset ids: `gustavosta`, `midjourney`, `diffusiondb`, `vidprom`. |
| `extra_files` | `[]` | Local `.txt` / `.jsonl` prompt files added to the pool. |
| `train_file`, `val_file` | `""` | Ready-made corpus files for `mode: files`. |
| `n_train` | `60000` | Training prompts (text packs). |
| `n_val` | `1024` | Validation prompts (text packs). |
| `empty_frac` | `0.01` (MageFlow: `0.015`) | Share of empty prompts injected so the adapter learns the empty CFG prompt. |
| `min_chars`, `max_chars` | `8`, `1200` | Length filter on prompts after whitespace normalisation. |
| `seed` | `8` | Shuffle / synthesis seed. |
| `image_presets` | `[]` (MageFlow: `["flickr30k"]`) | Image dataset ids: `flickr30k`, `coco_captions`. They are downloaded as materials. |
| `image_folders` | `[]` | Local folders of images, optional `<name>.txt` captions next to each image. |
| `n_image` | `56000` | Single-image training samples. |
| `n_two` | `3000` | Two-image training samples. |
| `n_text` | `27000` | Text-only training samples on the text-to-image template. |
| `n_val_image`, `n_val_text` | `1024`, `512` | Validation image / text samples. Val images never appear in train. |

### precompute

| Key | Default | Meaning |
| --- | --- | --- |
| `device` | `"auto"` | Torch device string, or `auto` = the CUDA device with the most VRAM, else CPU. |
| `placement` | `"sequential"` | LLaDA teacher device map. `sequential` fills the chosen GPU to its budget, then the other GPUs, then CPU RAM. `balanced` is accelerate's even split (about half the throughput). |
| `shard_size` | `1024` (MageFlow: `512`) | Samples per training shard. |
| `val_shard_size` | `512` (MageFlow: `256`) | Samples per validation shard. |
| `teacher_batch` | `24` (MageFlow: `0`) | Max prompts per teacher forward. `0` = auto for the card (vision packs). |
| `tok_budget` | `12288` (MageFlow: `0`) | Cap on batch × tokens per teacher forward. LLaDA counts text tokens + 256 query rows. `0` = auto. |
| `student_batch` | `64` (MageFlow: `0`) | Max prompts per student forward. |
| `student_tok_budget` | `0` | Vision packs: batch × tokens cap for the student. `0` = auto. |
| `vis_tok_budget` | `0` | Vision packs: merged vision tokens per vision-tower call. `0` = auto. |
| `teacher_mode` | `"auto"` | Vision packs: `gpu` keeps the text stack resident, `offload` streams it through the GPU with accelerate, `auto` picks `gpu` when the GPU budget is at least the text stack + 3 GiB. On CPU it is always `cpu`. |
| `gpu_mem_gib` | `0` | Teacher GPU budget. `0` = total VRAM of the chosen device − 1.5 GiB. |
| `cpu_mem_gib` | `0` | Teacher CPU budget. `0` = available RAM − 6 GiB (min 4). |
| `student_dtype` | `"bf16"` | `bf16`, `f16` or `f32` for the student forward. `f16` on CPU falls back to `f32`. |

Auto budgets for vision packs depend on the card: a GPU with more than 24 GB
gets teacher batch 512 / 65 536 tokens, student batch 256 / 131 072 tokens,
8 192 vision tokens; anything smaller gets 128 / 12 288, 64 / 24 576, 1 024.

### train

| Key | Default | Meaning |
| --- | --- | --- |
| `device` | `"auto"` | As above. |
| `width` | `1024` | Adapter width. Must be a multiple of 64 (head_dim 64). |
| `depth` | `6` (MageFlow: `4`) | Number of adapter blocks. |
| `steps` | `20000` | Optimizer steps. |
| `batch_size` | `32` | Samples per step. |
| `lr` | `2e-4` | Peak learning rate. Linear warmup, then cosine decay to 10 %. |
| `warmup` | `1000` | Warmup steps. |
| `weight_decay` | `0.01` | AdamW weight decay (betas 0.9 / 0.95). |
| `grad_clip` | `1.0` | Gradient-norm clip. |
| `cos_weight` | `0.5` | Weight of the `1 − cosine` term next to the whitened MSE. |
| `sigma_floor` | `0.03` | Floor on the per-dimension target standard deviation used for whitening. |
| `grad_checkpoint` | `"auto"` | `auto` enables activation checkpointing on CUDA cards under 12 GB; otherwise `on` / `off`. |
| `log_every` | `25` | Steps between metric rows in `log.csv` and state updates. |
| `val_every` | `500` | Steps between validation passes. `best.pt` is written when validation cosine improves. |
| `save_every` | `250` | Steps between `last.pt` saves (also on Stop and at the end). |
| `seed` | `42` | Torch seed and shard-stream seed. |

Changing `width` or `depth` after training started makes `last.pt`
incompatible; the train stage refuses to resume and tells you to reset
training or restore the settings.

### export

| Key | Default | Meaning |
| --- | --- | --- |
| `output_dir` | `""` | Where `<name>-f16.gguf` goes. Empty = the parent of the project folder. |
| `copy_to` | `""` | Optional extra directory the GGUF is copied to after export (for example your ggk model folder). |
| `export_sigvq` | `true` (MageFlow: `false`) | LLaDA: also export the SigVQ vision encoder as `<prefix>_sigvq-f16.gguf`. Turning it off also makes the SigVQ material "not wanted". |
| `run_eval` | `true` | Run the evaluation stage after export. |

## state.json

Written atomically by the pipeline (temp file + fsync + rename). Shape:

```json
{
  "status": "running",            // idle | running | stopped | failed | done
  "stage": "train",               // stage currently running
  "pid": 12345, "host": "box", "started": 1757600000.0, "finished": null,
  "error": null,
  "stages": {
    "corpus":           {"status": "done", "skipped": true, "started": ..., "finished": ..., "detail": "..."},
    "precompute_val":   {"status": "done", "done_shards": 2, "total_shards": 2, "prompts": 1024, "prompts_per_s": 4.5, "eta_s": 0},
    "train":            {"status": "running", "step": 1250, "steps": 20000, "loss": 0.41, "cos": 0.93,
                         "val_cos": 0.94, "best_val_cos": 0.94, "lr": 1.9e-4, "prompts_per_s": 310, "eta_s": 3100},
    "export":           {"status": "done", "path": "...gguf", "extras": ["...sigvq-f16.gguf"]},
    "eval":             {"status": "done"}
  },
  "updated": 1757600123.4
}
```

The GUI and `status` command derive **runtime status** from it: `running` if
the pid is alive and belongs to `gguf_trainer.pipeline`; `interrupted` if
the file says running but no such process exists; otherwise the recorded
status. Stage completion is always re-derived from the files on disk
(`stage_artifacts`), so a stale state after a reboot never hides real
progress, and clearing the state never loses work.

## Global settings

`~/.gguf-trainer/settings.json` holds `last_project` (the path the GUI
reopens and `--auto-resume` relaunches). Nothing else is stored globally.
