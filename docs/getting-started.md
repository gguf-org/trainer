# Getting started

## Requirements

* Python 3.10 or newer.
* PyTorch 2.1+ with CUDA strongly recommended. The trainer runs on CPU, but
  the teachers are large (see [Hardware](hardware.md)).
* The Python dependencies are installed automatically: `transformers`,
  `accelerate`, `safetensors`, `huggingface_hub`, `datasets`, `pyarrow`,
  `gguf-connector`, `pillow`, `numpy`, `psutil`.
* Disk: the LLaDA teacher is ~33 GB, the Qwen3-VL teacher ~8.3 GB, and the
  MageFlow image corpus can reach ~200 GB of shards at the full reference
  size. Pick corpus sizes for your disk.
* A local `pig_clip-f16.gguf` (the student). Quantized `pig_clip` files are
  dequantized on load, but the recipes were trained against f16.

## Install

From PyPI:

```bash
pip install gguf-trainer
```

From this repository (the build backend is flit):

```bash
pip install .            # or: pip install -e .
```

`torch` is declared as a plain dependency. If you need a specific CUDA build,
install PyTorch first following pytorch.org, then install `gguf-trainer`.

## Launch the GUI

```bash
gguf-trainer
```

This starts a local HTTP server on `http://127.0.0.1:8655/` and opens it in
your browser. Useful flags:

```bash
gguf-trainer serve --port 0            # pick a free port
gguf-trainer serve --host 0.0.0.0      # reachable from other machines (no auth: use with care)
gguf-trainer serve --no-browser        # do not open a browser tab
gguf-trainer serve --auto-resume       # relaunch the last project if a reboot interrupted it
```

Closing the server does **not** stop a running pipeline. Use the Stop button
or `gguf-trainer stop` for that.

## First run, step by step

1. **Create a project.** Setup tab, Project section, **+ New project**. Choose
   a folder, a name and a **trainer pack**. Projects default to
   `~/gguf-trainer/projects/<name>/`. The GGUF name gets a `pig_` prefix if
   you did not add one.
2. **Materials.** The pack lists the files it needs. Use **Browse** to point
   at your `pig_clip-f16.gguf`; the tokenizer snapshot and the teacher are
   fetched by **Download missing**. Files you already have nearby are found
   and linked instead of downloaded again (see [Materials](materials.md)).
   Downloads run detached and resume after interruptions.
3. **Corpus.** Keep the default prompt presets or add your own `.txt` /
   `.jsonl` files. Image packs additionally select an image dataset or a
   local image folder (see [Corpus](corpus.md)).
4. **Precompute, Training, Output.** The pack's reference hyper-parameters
   are prefilled. Set the device and memory budgets if the defaults do not
   fit your machine. Press **Save settings**.
5. **Train tab, Start / Resume.** The pipeline runs
   corpus → precompute (val) → precompute (train) → train → export → eval.
   The stage strip shows progress and ETA; the chart shows loss, cosine and
   validation cosine.
6. **Output tab.** The exported GGUF(s), the evaluation, and a copy-ready
   `ggk diffuser engine` command.

## Headless run

Every step has a CLI equivalent. Create the project in the GUI once (or write
`project.json` by hand, see [Projects](project.md)), then:

```bash
gguf-trainer download --project ~/gguf-trainer/projects/llada_adapter   # fetch missing materials
gguf-trainer run      --project ~/gguf-trainer/projects/llada_adapter   # foreground pipeline
gguf-trainer start    --project ~/gguf-trainer/projects/llada_adapter   # detached pipeline
gguf-trainer status   --project ~/gguf-trainer/projects/llada_adapter
gguf-trainer stop     --project ~/gguf-trainer/projects/llada_adapter   # saves, then exits
```

## Smoke test without the teacher

Set `GGUF_TRAINER_MOCK_TEACHER=1` (or tick **Dev smoke test: mock teacher**
under Precompute) to run the whole pipeline with synthetic targets. The
student and its tokenizer are still required. The resulting adapter is
meaningless; use this only to exercise the machinery on a laptop.

```bash
GGUF_TRAINER_MOCK_TEACHER=1 gguf-trainer run --project ./smoke
```

Shards written with the mock teacher carry `"mock_teacher": true` in their
metadata.

## Where things end up

| What | Where |
| --- | --- |
| Project settings and state | `<project>/project.json`, `<project>/state.json` |
| Downloaded materials | `<project>/materials/` (or linked from elsewhere) |
| Corpus | `<project>/data/` |
| Precomputed shards | `<project>/shards/{val,train}/` |
| Checkpoints and training log | `<project>/checkpoints/` |
| Evaluation | `<project>/eval.json` |
| Exported GGUFs | **next to** the project folder by default (`export.output_dir` overrides) |
| GUI settings (last project) | `~/.gguf-trainer/settings.json` |
