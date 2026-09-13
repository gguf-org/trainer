# GUI guide

The GUI is a single page served by the local backend (`gguf-trainer serve`).
The browser and the server run on the same machine; every request carries
the project's filesystem path, so nothing is uploaded. The page polls the
project every 1.5 s while it is open and the hardware readings every 3 s.

The header shows the server version and the current project. Five tabs:
**Setup**, **Train**, **Hardware**, **Logs**, **Output**.

## Setup tab

### Project

* **+ New project**: pick a parent folder (defaults to
  `~/gguf-trainer/projects`), a name and a **trainer pack**. The name becomes
  the folder name and the export name; a `pig_` prefix is added if missing
  (`llada_adapter` → `pig_llada_adapter-f16.gguf`).
* **Open**: open an existing project folder through the file browser.
  Folders containing a `project.json` are marked as projects.
* The last opened project is remembered in `~/.gguf-trainer/settings.json`
  and reopened on the next launch.
* **Save settings** writes every field on the Setup tab into `project.json`.
  Starting the pipeline saves unsaved changes first.

### Materials

One row per material the pack needs (see [Materials](materials.md)). Each row
shows its status: `ready`, `missing`, `partial`, or `downloading` with bytes
done / total and a rate.

* **Browse**: choose a local file (the student `pig_clip*.gguf`) or override
  the location of a snapshot.
* **Download missing**: starts one detached download per absent Hugging Face
  material. It is disabled while nothing is missing, so a second click never
  starts a duplicate. Each row also has its own download button.
* **Refresh**: re-scans nearby folders and links any complete copy it finds
  (a notice lists what was linked).
* **HF token**: stored in `project.json` as `hf_token`, used for gated repos
  and for the datasets presets.

### Corpus

* **Mode**: public prompt datasets (+ local files), or ready-made
  `train.txt` / `val.txt` files.
* Preset checkboxes (Stable Diffusion prompts, Midjourney prompts,
  DiffusionDB, VidProM). **+ Add file** adds `.txt` (one prompt per line) or
  `.jsonl` (a `text` field) files.
* Train prompts / Val prompts / Empty fraction / Min chars / Max chars /
  Seed.
* Image packs replace the prompt counts with **Image samples (train)**,
  **Two-image samples**, **Text-only samples (train / val)** and **Val image
  samples**, plus image dataset checkboxes and **+ Add folder** for local
  image folders. Ticking an image preset immediately saves the project so
  the Materials list shows its download.
* **Prepare corpus now** runs only the corpus stage (detached) so you can
  inspect `data/` before precomputing.

### Precompute

Device, **Teacher placement** (`sequential` / `balanced`, LLaDA), **Teacher
mode** (`auto` / `gpu` / `offload`, MageFlow), shard sizes, batch and token
budgets (0 = auto for vision packs), teacher GPU / CPU memory budgets in GiB
(0 = auto), student dtype, and the **Dev smoke test: mock teacher** checkbox.
Field meanings are in [Projects and configuration](project.md#precompute).

### Training

Width, depth, steps, batch size, learning rate, warmup, weight decay, grad
clip, cosine weight, sigma floor, grad checkpoint, log / validate /
checkpoint intervals, seed, device. See [Adapters](adapters.md) for what the
recipe does with them.

### Output

**Adapter name**, output folder (default: the folder containing the project
folder), an optional **copy to** folder (your ggk model directory), whether to
export the SigVQ encoder (LLaDA), and whether to run the evaluation.

**Start pipeline** at the bottom starts the full pipeline. If a required
material is not ready you are warned and may start anyway (the run fails at
the first stage that needs the missing file, which is fine when you only
want the corpus built).

## Train tab

* **Start / Resume** and **Stop**. Stop writes the `STOP` file and sends
  SIGTERM; the pipeline saves its checkpoint or finishes the current shard,
  then exits. Resume continues from the saved position.
* **Reset…** menu (only when the pipeline is not running):
  * *training (checkpoints)*: deletes `checkpoints/` and `eval.json`.
  * *precomputed shards + training*: deletes `shards/` and `checkpoints/`.
  * *corpus + everything after*: deletes `data/`, `shards/`, `checkpoints/`.
  * *status only (keep files)*: clears `state.json`; progress is re-detected
    from disk on the next run.
  Downloaded materials and exported GGUFs are never touched by these.
* **Stage strip**: corpus → precompute (val) → precompute (train) → train →
  export → eval, with per-stage progress, throughput and ETA. Once
  `best.pt` exists and nothing is running, the **Export GGUF** and
  **Evaluate** boxes are clickable and regenerate the GGUF / `eval.json`
  from the checkpoint (a forced `--only export eval` run).
* **Training metrics**: loss, train cosine and validation cosine over steps
  (from `checkpoints/log.csv`, downsampled to ~400 points), plus the current
  numbers.
* **Live hardware**: GPU utilisation and memory, CPU, RAM of the machine and
  the pipeline process.

Status labels: Idle, Running, Interrupted (state says running but no live
process, for example after a reboot), Stopped, Failed, Done.

## Hardware tab

GPUs from `nvidia-smi` and from torch (name, VRAM total / used, utilisation,
temperature, power), RAM, free disk at the project path, and the
**Environment**: Python executable and the versions of torch, transformers,
gguf-connector, huggingface_hub, datasets, safetensors, accelerate, pillow,
psutil, plus the CUDA version.

## Logs tab

Tails `<project>/pipeline.log` incrementally (a first read of a large log
starts from its last 256 KB). **Clear view** clears the panel only, not the
file. Download jobs log separately to
`<project>/materials/.download-<id>.log`.

## Output tab

* **Exported files**: every `.gguf` in the output folder carrying this
  project's export prefix (`pig_llada_*` for a project named
  `pig_llada_adapter`), with sizes and timestamps.
* **Export GGUF / Re-export GGUF**: rebuilds the f16 GGUF from `best.pt` and
  re-evaluates it. Training is not touched.
* **Evaluation**: the contents of `eval.json`, the pack's headline keys first
  (see [Evaluation](evaluation.md)).
* **Use it in ggk**: the `ggk diffuser engine` command with the student,
  adapter and (LLaDA) SigVQ paths filled in. **Copy** copies it.
