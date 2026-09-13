# gguf-trainer documentation

`gguf-trainer` trains **pig_clip adapters**: small bridge networks that let the
`pig_clip` GGUF (the student, a native train / fine-tune that shares the Qwen3
tokenizer and layout) stand in for a diffusion model's
original text encoder in the **ggk** engine. It ships as a browser GUI with a
local backend, a headless CLI, resumable downloads and a checkpointed
pipeline that exports an f16 GGUF for `--llm-adapter`.

Version documented: `0.0.6` (see `src/gguf_trainer/__init__.py`).

## Contents

| Page | What it covers |
| --- | --- |
| [Getting started](getting-started.md) | Install, launch the GUI, first run, headless run, smoke test |
| [CLI reference](cli.md) | Every `gguf-trainer` subcommand and flag, exit codes, module entry points |
| [GUI guide](gui.md) | The Setup / Train / Hardware / Logs / Output tabs and what each control does |
| [Projects and configuration](project.md) | The project folder, `project.json` reference, `state.json`, global settings |
| [Pipeline](pipeline.md) | The six stages, what "done" means, stop / resume / force / reset, the shard contract |
| [Materials](materials.md) | What each pack downloads, where files are found, detached and foreground downloads |
| [Corpus](corpus.md) | Text presets and local files; the image corpus and instruction synthesis |
| [Pack: LLaDA-Image-Turbo](packs/llada-image.md) | The trainer8 recipe: teacher, template, adapter, metrics, engine command |
| [Pack: MageFlow-Edit](packs/mageflow-edit.md) | The trainer5 recipe: Qwen3-VL teacher, vision bridge, engine contract |
| [Adapters and export](adapters.md) | The two adapter architectures, the losses, the GGUF layout and metadata |
| [Evaluation](evaluation.md) | Every key in `eval.json` and which numbers to judge a run by |
| [Data formats](data-formats.md) | Corpus files, shard `.npz` layout, checkpoints, `log.csv`, `vision_proj.pt` |
| [HTTP API](api.md) | The JSON endpoints the GUI uses |
| [Hardware](hardware.md) | Requirements, device selection, memory budgets, teacher placement |
| [Troubleshooting](troubleshooting.md) | Common errors and what they mean |
| [Adding a pack](adding-a-pack.md) | How to implement a new `TrainerPack` |

## The short version

```bash
pip install gguf-trainer
gguf-trainer                      # GUI at http://127.0.0.1:8655/
```

1. **Setup tab**: create a project, pick a trainer pack, point at your
   `pig_clip-f16.gguf`, press **Download missing**.
2. Choose a corpus, keep or edit the pack's reference hyper-parameters,
   **Save settings**.
3. **Train tab**: **Start / Resume**. The pipeline runs detached and survives
   closing the browser, the terminal, and reboots.
4. **Output tab**: the exported `<name>-f16.gguf`, its evaluation, and the
   `ggk` command that uses it.

Everything is addressed by filesystem path. Nothing is uploaded anywhere.
