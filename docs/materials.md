# Materials

Materials are the files a pack needs before it can run: the student, its
tokenizer, the teacher, and pack extras. They live under
`<project>/materials/` unless a copy is found elsewhere and linked.

## What each pack needs

Every pack shares the two **student** materials:

| Id | Kind | Source | Size | Notes |
| --- | --- | --- | --- | --- |
| `student` | local file | your `pig_clip-f16.gguf` | ~1.2 GB | `pig_clip` is a native train / fine-tune, not a stock Qwen3 checkpoint; it shares the Qwen3 tokenizer and tensor layout. Quantized files are dequantized on load; the recipes were trained against f16. |
| `student_tokenizer` | HF snapshot | `callgg/pig-clip-tokenizer` | ~11 MB | `config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`. The GGUF carries no tokenizer. |

**LLaDA-Image-Turbo** pack:

| Id | Kind | Source | Size | Required |
| --- | --- | --- | --- | --- |
| `teacher` | HF snapshot | `inclusionAI/LLaDA-Image-Turbo`: `model_index.json`, `text_encoder/*`, `queryformer/*`, `text_projection/*`, `tokenizer/*` | ~33 GB | yes |
| `sigvq` | HF snapshot | same repo, `sigvq/*` | ~2.4 GB | no; wanted only while `export.export_sigvq` is true |

**MageFlow-Edit** pack:

| Id | Kind | Source | Size | Required |
| --- | --- | --- | --- | --- |
| `teacher` | HF snapshot | `Qwen/Qwen3-VL-4B-Instruct`: config, safetensors shards + index, preprocessor config, tokenizer files, chat template | ~8.3 GB | yes |
| `images_flickr30k` | HF dataset snapshot | `nlphuji/flickr30k`: `flickr30k-images.zip`, `flickr_annotations_30k.csv` | 4.4 GB | when selected under Corpus |
| `images_coco_captions` | HF dataset snapshot | `Multimodal-Fatima/COCO_captions_train`: `data/*.parquet` | 17 GB | when selected under Corpus |

Image materials appear in the list only for the presets ticked in
`corpus.image_presets`.

## Where a material is looked for

Resolution order for a material's path:

1. `materials.<id>.path` in `project.json` (set by Browse or by linking);
2. for HF snapshots, `<project>/materials/<subdir>`;
3. for local files, the material's default path if it exists.

## Automatic linking of existing copies

When a project is opened, created, refreshed, or a download is requested,
the trainer searches for complete copies of anything still missing and links
them by writing `materials.<id>.path` instead of downloading again. Search
roots, most specific first:

1. `$GGUF_TRAINER_MATERIALS`, if set;
2. the current working directory and `./materials`;
3. the project folder and its parent;
4. every other project's `materials/` under `~/gguf-trainer/projects`.

For each root both `<root>/<subdir>` and `<root>` itself are checked. A
snapshot counts as complete when every pattern matches at least one file
and no `.incomplete` blob is left under its `.cache/huggingface/download/`.
For the student, any `pig_clip*.gguf` in those roots (plus the project's
`materials/`) is a candidate; an `f16` file is preferred, then the largest.

The 1.5 s GUI poll does not do this scan (`online=0`); Open, Create,
Refresh and Download missing do.

## Download jobs

Each HF material downloads through `huggingface_hub.snapshot_download` with
`local_dir=<materials>/<subdir>` and the material's `allow_patterns`. That is
resumable by itself: partial blobs live under `<local_dir>/.cache/huggingface`
and a rerun continues them.

In the GUI a download is a **detached** child
(`python -m gguf_trainer.materials --project DIR --id ID`) with its pid, log
and final status in `<project>/materials/.download-<id>.{pid,log,status}`.
It survives the GUI closing and is found again on restart. A finished
status file outranks a pid that merely looks alive, because pids get
recycled. The `HF_TOKEN` environment variable is passed from the project's
`hf_token`.

Progress is measured as bytes on disk against the sizes reported by the Hub
API (cached per server process), so an interrupted download shows its real
remaining size. Statuses: `missing`, `partial`, `downloading`, `ready`. When
the Hub cannot be reached and sizes were never listed, readiness is judged
by the files on disk alone.

`gguf-trainer download --project DIR` does the same in the foreground, one
material after another, and is the right tool for a headless machine.

## Choosing the student file

Select the file with Browse, or drop a `pig_clip*.gguf` next to the project
or in the directory you start the GUI from and press Refresh. The precompute
stage refuses to start without it.
