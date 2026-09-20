# trainer

A trainer GUI for **pig_clip adapters**: small bridge networks that let
`pig_clip` (native train/fine-tune shipped as a GGUF) stand in for a diffusion
model's original text encoder in the **ggk** engine. Four trainer packs ship:

* **LLaDA-Image-Turbo** — the 16B LLaDA2-MoE text stack is replaced by
  `pig_clip` + a 256-query resampler adapter (the trainer8 recipe); the
  adapter is text-only, so it pairs unchanged with the model's SigVQ vision
  encoder for image editing.
* **MageFlow-Edit** — the Qwen3-VL-4B-Instruct text stack is replaced by
  `pig_clip` + a token-aligned adapter with a vision extension (the trainer5
  recipe); the adapter pairs with the unchanged
  `mmproj-qwen3vl-4b-it-f16.gguf` vision encoder for editing and text-to-image.
* **Qwen-Image 2.1** — the Qwen3-VL-8B-Instruct text stack is replaced by
  `pig_clip` + the same kind of adapter, distilled from the *full* Hugging
  Face model (deepstack, M-RoPE image positions, pre-norm tap); text-to-image
  needs only the adapter, editing pairs it with the unchanged
  `mmproj-qwen3vl-8b-it-f16.gguf` vision encoder.
* **PixArt** — the 4.7B T5-XXL v1.1 encoder (`--t5xxl`) is replaced by
  `pig_clip` + a 120-query resampler seeded with the T5 token ids (the
  original `trainer` recipe); the T5 *tokenizer* stays at inference, the
  encoder is never loaded.

Both packs train against `pig_clip-f16.gguf` (get it [here](https://huggingface.co/gguf-org/trainer/blob/main/pig_clip-f16.gguf)) as the student.

```bash
pip install gguf-trainer
gguf-trainer                # opens http://127.0.0.1:8655/ in the browser
```

Full documentation (GUI guide, CLI, configuration reference, pipeline,
packs, API): [`docs/`](docs/README.md).

The GUI runs in your browser against a local backend, in the style of the
ggk diffuser GUI. Nothing is uploaded: models, datasets and outputs are
addressed by filesystem path through the built-in file browser.

![screenshot](https://raw.githubusercontent.com/gguf-org/ggk/master/media/trainer1.png)

## What the GUI does

**Setup tab** — everything a run needs, in one project folder:

* **Project**: create or open a project, picking its **trainer pack**. A
  project is a directory holding the downloaded materials, corpus,
  precomputed shards, checkpoints and `eval.json`, plus `project.json`
  (settings) and `state.json` (progress). The exported GGUFs land **next to
  the project folder** by default (`test-trainer/pig_llada_adapter-f16.gguf`
  for the project `test-trainer/llada_adapter/`), so every adapter trained
  under one folder ends up side by side; the Output section can point them
  elsewhere. The evaluation lives inside the project folder, so several
  projects sharing one output folder never overwrite each other's
  `eval.json`.
* **Materials**: the pack lists what it needs — the teacher from Hugging
  Face (LLaDA: `inclusionAI/LLaDA-Image-Turbo` text encoder, QueryFormer,
  text_projection, tokenizer, ~33 GB; MageFlow: `Qwen/Qwen3-VL-4B-Instruct`,
  ~8.3 GB), the student tokenizer/config (`callgg/pig-clip-tokenizer`,
  ~11 MB), pack extras (LLaDA: the optional SigVQ vision encoder, ~2.4 GB;
  MageFlow: the image dataset(s) selected under Corpus), and your local
  `pig_clip-f16.gguf`. One **Download missing**
  button fetches everything that is not on disk yet; it is disabled (and so
  is each material's own button) as soon as the files are present, so a
  second click can never start a duplicate download. Files you already have
  are found automatically — in the project, in the directory the GUI was
  started from, in another project's `materials/`, or in a folder named by
  `GGUF_TRAINER_MATERIALS` — and linked instead of downloaded again (a
  `pig_clip*.gguf` next to those is picked as the student). Downloads run as
  detached processes (they survive closing the GUI) and resume where they
  stopped after an interruption or a reboot. Headless:
  `gguf-trainer download --project DIR` does the same in the foreground.
* **Corpus**: pick public prompt datasets (Stable Diffusion prompts,
  Midjourney prompts, DiffusionDB, VidProM) and/or your own `.txt`/`.jsonl`
  files, or point at ready-made `train.txt`/`val.txt`. ~1% empty prompts are
  injected so the adapter learns the empty CFG prompt. Image packs add
  **image datasets** (Flickr30k, COCO captions — downloaded as materials)
  and/or local image folders (optional `<name>.txt` caption next to each
  image); edit instructions are synthesized from the captions, and the
  prompt datasets supply the text-only share on the text-to-image template.
* **Precompute / Training / Output**: the pack's reference hyper-parameters,
  editable (LLaDA: width 1024, depth 6, 20k steps, batch 32, lr 2e-4 …;
  MageFlow: width 1024, depth 4, 20k steps, batch 32), the device, memory
  budgets for the teacher, the adapter name, output folder and an optional
  copy destination (your ggk model folder).

**Train tab** — start/stop/resume the pipeline, a stage strip
(corpus → precompute val → precompute train → train → export → eval) with
per-stage progress and ETA, live loss / cosine / val-cosine chart, current
metrics, and live GPU/CPU/RAM readings of the machine and the pipeline
process.

**Hardware tab** — GPUs (nvidia-smi + torch), RAM, disk, Python/torch/
transformers versions. **Logs tab** — the pipeline log, following.
**Output tab** — exported files, the evaluation of the exported GGUF, and
the ggk command that uses it (Copy).

## Resuming after a reboot

The pipeline runs as a detached process (setsid / detached process group)
and every stage is idempotent and checkpointed:

* corpus files and each shard are written atomically and skipped when present;
* training saves `last.pt` every N steps (and on Stop / SIGTERM), including the
  optimizer, RNG and the exact position in the shard stream;
* export/eval rerun only when the checkpoint is newer than the GGUF.
  The **Export GGUF** button (Output tab) or a click on the Export /
  Evaluate stage boxes regenerates them from `best.pt` on demand, e.g.
  after the GGUF was deleted (`gguf-trainer start --project DIR --only export eval --force`).

Open the project (or start the GUI with `gguf-trainer --auto-resume`, which
relaunches the last project if its process died while running) and press
**Start / Resume**. The same works headless:

```bash
gguf-trainer run    --project ~/gguf-trainer/projects/llada_adapter   # foreground
gguf-trainer start  --project ~/gguf-trainer/projects/llada_adapter   # detached
gguf-trainer stop   --project ~/gguf-trainer/projects/llada_adapter   # saves, then exits
gguf-trainer status --project ~/gguf-trainer/projects/llada_adapter
gguf-trainer download --project ~/gguf-trainer/projects/llada_adapter   # fetch missing materials
```

![screenshot](https://raw.githubusercontent.com/gguf-org/ggk/master/media/trainer2.png)

## Changing the number of training steps

The planned step count (`train.steps`, default 20 000) is a setting like any
other, not a fixed recipe: the **Steps** field on the Setup tab, the
**planned steps** box on the Train tab (Apply), or the CLI change it, and the
change is honoured at every point of a project's life:

* **before the run** — the usual case (a quick 5k pilot, a long 40k run);
* **while training runs** — the trainer re-reads `project.json` at every log
  interval (`log_every`, default 25 steps). Lower the count to finish *now*:
  the current step is validated and saved, then export and eval run as
  usual. Raise it to keep going past the original plan;
* **after a finished run** — a higher count un-finishes the training stage;
  **Start / Resume** (or `gguf-trainer start`) continues from `last.pt`
  (optimizer, RNG and shard position included) up to the new count, and the
  GGUF is re-exported if `best.pt` improves.

```bash
gguf-trainer set   --project DIR --steps 12000      # also while it runs
gguf-trainer start --project DIR --steps 40k        # set, then launch
```

The cosine learning-rate schedule is computed from the count in force, so
extending a finished run lifts the lr back up mid-schedule (warm restart);
snapshots and the exported GGUF record the planned count that was in force
(`adapter.planned_steps`).

## Snapshots: a usable GGUF at any step

You do not have to wait for the last of the 20 000 steps. **Stop** pauses the
run after the current step (Start / Resume continues it from `last.pt`), and
**Export snapshot** (Train tab, Output tab, or the CLI) writes the adapter as
it is *right now*:

```bash
gguf-trainer snapshot --project DIR                     # newest step, while training runs or after a stop
gguf-trainer snapshot --project DIR --checkpoint best   # best validation score so far
gguf-trainer snapshot --project DIR --eval              # + evaluate the GGUF on the val shards
```

* While the train stage is running the job drops a `SNAPSHOT` file; the
  trainer validates and saves `last.pt` at the current step, removes the file
  and keeps training — nothing is interrupted (a few seconds).
  `--no-fresh` takes the checkpoint already on disk instead.
* The file is step-tagged, `<name>-step<N>-f16.gguf`, next to the final
  `<name>-f16.gguf`, so several points of one run can be kept and swapped
  into the same engine command. The final export is not affected.
* `<project>/snapshots.json` (Output tab › Snapshots) lists every snapshot
  with its step, the planned steps, the validation cosine measured when the
  checkpoint was saved and the optional eval — the steps-vs-quality trade-off
  is visible instead of hidden behind the final export.
* Provenance lives in the GGUF itself: `adapter.trained_steps`,
  `adapter.planned_steps`, `adapter.checkpoint` (`last` / `best`),
  `adapter.snapshot`, `adapter.val_cos`, `adapter.best_val_cos`.
* The eval of a snapshot taken while training runs is done on the CPU
  (the training process owns the GPU); `--device` overrides.
* Resetting the training stage also clears the manifest; the snapshot files
  are removed with the other exports by the "output" reset.

![screenshot](https://raw.githubusercontent.com/gguf-org/ggk/master/media/trainer3.png)

## The LLaDA-Image pack

Teacher target per prompt = the 256 QueryFormer rows of `cap_feats`
(`[256, 2560]`): LLaDA2-MoE over `[tokens ; 256 queries]` with the text
masked from seeing the queries, then the 6-layer text_projection. QueryFormer
and text_projection are re-implemented in plain torch (bit-exact against the
diffusers originals) so no diffusers install or reference checkout is
needed; the MoE backbone loads through `trust_remote_code` from the snapshot
and is placed *sequentially*: the chosen GPU up to its budget, then the
other CUDA devices, then CPU RAM (the Precompute tab can switch to
accelerate's balanced split, which caps the biggest card at an even share of
the model and offloads the rest — roughly half the throughput).

Student = `pig_clip` final-norm hidden states over the engine's exact template
(`<role>HUMAN</role> Generate an image: {text}\n<role>ASSISTANT</role>\n<IMAGE1>`),
Qwen BPE without special tokens. The adapter is a seedless Perceiver
resampler (self-attn + cross-attn + GELU MLP, head_dim 64) trained with
whitened MSE + cosine on per-dim standardized targets; the export folds the
standardization into `out_proj` and writes f16 weights / f32 norms, biases and
query, exactly the layout `pig_llada_adapter-f16.gguf` shipped with.

Use it in ggk (≥ 0.5.7):

```bash
ggk diffuser engine -- --diffusion-model LLaDA-image-turbo-nvfp4.gguf \
    --vae pig_flux2_vae_fp32-f16.gguf \
    --llm pig_clip-q8_0.gguf --llm-adapter pig_llada_adapter-f16.gguf \
    --llm_vision pig_llada_sigvq-f16.gguf \
    --ref-image sheep.png -p "a sheep in sunglasses" --cfg-scale 1.0 \
    --steps 4 --sampling-method euler --diffusion-fa -o out.png
```

Text-to-image works with any student quantization; editing wants `pig_clip`
at q8_0 or better. Judge a run by **val centred cosine / rel_mse** (0.965 /
0.0024 on the reference 5090 run, ~55 min of training); plain cosine on these
rows is ~0.99 even for a zero prediction.

**transformers 5 and the teacher's rotary tables.** transformers 5 builds
models on the meta device and does not re-initialize the non-persistent
buffers of remote (`trust_remote_code`) models, so the LLaDA2-MoE backbone
came up with an uninitialized `inv_freq` RoPE table. The teacher still
emitted plausible rows (same mean and scale, plain cosine 0.999 to the real
ones) but with the positional signal scrambled, and adapters trained on those
targets ignore the prompt in the engine (0.1–0.2 lower centred cosine against
the true teacher; edits return the reference image). Since 0.0.3 the teacher
repairs the tables after loading and refuses to run with a bad one, and every
shard directory carries a `CONTRACT` marker: shards written before the fix
(contract `llada_image/1` or none) are discarded on the next run, the
checkpoints trained on them are moved to `checkpoints.stale-<time>/`, and
precompute + training start over. A quick health check of any adapter is its
centred cosine against a teacher-conditioned engine context
(`trainer8/dumps/m1_gpu_full/context.bin`): ≥ 0.96 is healthy, ~0.87 is the
broken-RoPE signature.

## The MageFlow-Edit pack

MageFlow-Edit conditions its DiT on **Qwen3-VL-4B-Instruct** final-norm
hidden states with the reference image spliced in as mmproj vision tokens.
The pack distills that conditioning into `pig_clip` + a **token-aligned
adapter with a vision extension** (trainer5): position i of the student maps
to position i of the teacher, and because ggk splices the 2560-d mmproj
embeds into the LLM input while the student embeds at 1024, the adapter
owns the bridge in both directions — a frozen `vision_proj` (2560 → 1024,
ridge least squares over the shared vocabulary, applied by the *engine* to
every mmproj embed before the student) and a trained `vis_in` (2560 → width)
that hands the adapter the *raw* mmproj embeds, so vision fidelity does not
depend on what survives the student. The 4B mmproj stays exactly as the
teacher uses it; nothing else needs converting.

The engine contract is replicated exactly (verified against ggk
`SD_DUMP_COND` dumps): the 64-token edit template / 34-token text-to-image
template, the 6-token `Image N: <|vision_start|>` header, nearest-neighbour
resize to a multiple of 32 with the long side capped at 384, OpenAI-CLIP
normalisation, the vision tower's main merger output only (ggk drops
deepstack), all-equal M-RoPE (= plain rope), final-norm tap. The vision
tower is the one the mmproj was converted from (HF bf16 vs engine f16: cosine
0.998), the text stack matches the engine's q4_k_m teacher at the known
quantisation floor (0.975 on text positions).

Corpus (trainer5 mix): 56k single-image + 3k two-image samples with
instructions synthesized from the captions (plain and truncated captions,
add / remove / replace / restyle / recolor / background patterns, ~1.5%
empty) plus 27k text-only prompts; val = 1024 image + 512 text samples,
val images never in train. Default image source is Flickr30k (4.4 GB, 31k
photos with 5 captions each — several instructions per photo); the COCO
captions preset (17 GB, 113k photos) is the trainer5 source. Every sample
stores the teacher and student states of every token plus the raw vision
embeds (~2.4 MB per image sample, ~200 GB for the full mix) — pick the
corpus size for your disk.

Precompute runs the vision tower, the Qwen3-VL text stack and `pig_clip`
once per shard. The text stack (~7.5 GB bf16) stays resident when the GPU
budget allows, otherwise it is streamed through the GPU with accelerate's
`cpu_offload` (≈ 1–2 samples/s on a 6 GB laptop card, ~44 on an RTX 5090
resident); batch/token budgets at 0 are chosen per card. Training uses the
trainer5 recipe: width 1024 (its width gate winner), depth 4, 20k steps,
batch 32, lr 2e-4, warmup 1k, whitened MSE + 0.5·(1−cos) masked to real
tokens, `out_proj` zero-initialised, `vision_proj` frozen. The export writes
the same layout as the shipped `pig_qwen3vl_4b_adapter-f16.gguf` (f16
weights, f32 norms/biases/`vision_proj`).

Use it in ggk:

```bash
ggk diffuser engine -- --diffusion-model mageflow-edit-turbo-nvfp4.gguf \
    --vae pig_mageflow_vae_fp32-f16.gguf \
    --llm pig_clip-q8_0.gguf --llm-adapter pig_qwen3vl_4b_adapter-f16.gguf \
    --llm_vision mmproj-qwen3vl-4b-it-f16.gguf \
    --ref-image sheep.png -p "a sheep in sunglasses" --cfg-scale 1.0 \
    --steps 4 --sampling-method euler --diffusion-fa -o out.png
```

Judge a run by **cos_slice** (cosine over the positions the DiT consumes)
and the **cos_vis / cos_txt** split: the trainer5 reference reached val cos
0.915 (vision positions 0.80, text 0.98) and its A/B edits were
near-identical to the teacher's — the 4-step DiT forgives far more than the
cosine suggests. A lagging vision cosine means the `vis_in` path or the
width is the limiter, not the student.

## The Qwen-Image 2.1 pack

Qwen-Image 2.1 conditions its DiT on **Qwen3-VL-8B-Instruct**: the last
decoder layer *before* the final RMSNorm, on one template for both modes
(`<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n`
— 14 tokens, dropped — then the user turn, with
`<imageN><|vision_start|>…<|vision_end|>` blocks in front of the instruction
when editing). The pack (`qwen3vl_qwenimage`) is the 8B sibling of the
MageFlow pack — same adapter kind, same student, the same frozen
`vision_proj` / trained `vis_in` bridge, here 4096 ↔ 1024 — and exports
`pig_qwen3vl_8b_adapter-f16.gguf`:

```
--llm pig_clip-<quant>.gguf --llm-adapter pig_qwen3vl_8b_adapter-f16.gguf
[--llm_vision mmproj-qwen3vl-8b-it-f16.gguf -r ref.png]
```

Three things differ from MageFlow, all of them the engine contract
(`vision_data.QwenImage21Contract`) rather than the model size:

* **The teacher is the unreduced model.** For MageFlow the teacher replicates
  what ggk runs (no deepstack, plain 1-D rope, final norm). Here it is
  Hugging Face's own forward — deepstack features added into the first
  decoder layers, real (t, h, w) M-RoPE positions for the image tokens, and
  the pre-norm tap (`qwen3vl_teacher.Qwen3VLFullTeacher`; it reproduces
  `Qwen3VLModel(pixel_values=…)` to 2e-7 on a padded multi-image batch, the
  rest being the bf16 the shards are stored in). The adapter still only ever
  receives the mmproj's main output, so what deepstack told the teacher is
  learned from that — which is also why the adapter route needs no deepstack
  support in the engine.
* **Only the rows the DiT reads are trained.** The 2.1 DiT substitutes the
  reference latents for the context rows under the image slots (4 latent
  tokens per slot) and drops the system turn, so the loss, the validation
  scores and the target statistics cover the text rows from position 14 on.
  Those rows sit after the image in a causal LM, so they are where the
  image's influence on the conditioning lives. Taking the statistics over
  the same rows matters with a pre-norm tap: its attention-sink rows would
  otherwise own the whitening. Teacher rows under the slots are not stored.
* **The vision token count follows the render size.** The engine resizes a
  reference to the render area (nearest, both sides rounded to 32, alpha over
  white, `2x − 1`), one slot per 32×32 px: 144 tokens at 384² up to 1024 at
  1024². Each image sample draws its area from `corpus.ref_areas`
  (`[[area_px, weight], …]`, default 384²/512²/768²/1024² at
  0.35/0.35/0.20/0.10, deterministic per sample); two-image samples stay at or
  below `corpus.two_image_max_area` each (default 512², for cost). The engine
  gives *every* reference the full render area, so two-reference edits above
  that size are outside what the adapter saw unless you raise it. Shards run ~0.6 MB per text sample and
  3–11 MB per image sample — the default corpus (40k text, 30k image, 2k
  two-image) is about 170 GB.

The 17.5 GB teacher is resident on a 24 GB+ card; below that it is streamed
through the GPU (`teacher_mode: offload`, RAM must hold it), which is slow at
1024-token references — lower the weights of the large areas, or the image
counts, for a first run on a small card.

Judge a run by **cos_slice** (exactly the rows the DiT consumes), split into
**cos_t2i** (text-only samples) and **cos_edit** (the text rows of image
samples). A pre-norm tap has a few very large dimensions, so plain cosine
runs high; compare runs by `rel_mse` as well. No reference run exists yet.

On the ggk side this needs 0.6.6+: Qwen-Image 2.1 support, and for editing
the 2.1 conditioner path that takes the references through the mmproj when an
adapter with a vision extension is loaded (without one ggk refuses 2.1 edits,
because it has no deepstack path). The engine and this pack were checked
against each other: reference resize (pixel-index exact), template tokens and
image slot positions for one and two references.

## The PixArt / T5-XXL pack

PixArt's DiT cross-attends to **T5-XXL v1.1** encoder output over a 120-slot
caption window (4096-dim rows; pads carry a -10000 attention bias, so only
the real tokens matter). The pack distills that conditioning into
`pig_clip` + a **seeded resampler**: 120 learned queries, each seeded with
the T5 sentencepiece embedding of its slot (`adapter.t5_embed.weight`),
cross-attend over the student's final-norm states and are projected to
4096. The seed is what lets a fixed query window follow T5's own
segmentation over Qwen BPE states — without it the adapter plateaus barely
above the per-position mean baseline (0.515 vs 0.455 measured).

Contract (verified against the engine at cosine 0.999999 in `trainer/`):

* student ids = the raw prompt through the Qwen BPE, no special tokens, an
  empty prompt = the single pad token (`151643`);
* seed ids = T5 sentencepiece + EOS, pad-0 to 120 — exactly the sequence
  ggk's `PixArtT5Embedder` builds for its mask, so the ids are free at
  inference; `query_i = query[i] + t5_embed[id_i]`;
* target = `T5EncoderModel.last_hidden_state` at the real slots; the loss is
  masked to them: MSE whitened by the per-dim sigma (floored at 0.03 — T5-XXL
  has ~120 near-constant dims that otherwise carry half the loss as bf16
  noise) + 0.5·(1−cos), on RAW targets (no standardization fold at export).

The teacher is one dense 9.5 GB bf16 encoder: `teacher_mode` gpu keeps it
resident (needs a ~12 GiB budget), offload streams the 24 layers from RAM
through the GPU (a 6 GB card runs it at ~0.25 GiB of VRAM), and the batched
teacher matches the original per-prompt full-window path bit for bit. Only
the real slots are stored (~40 rows × 4096 per prompt, ~0.4 MB).

Export: `pig_t5_adapter-f16.gguf` with the layout of the shipped file
(`adapter.query` f32, `adapter.t5_embed.weight` f16 `[32128, width]`,
explicit q/k/v/o linears, norms and biases f32, KV `adapter.t5_vocab`).

```bash
ggk diffuser engine -- --diffusion-model pixart-nvfp4.gguf \
    --vae pig_pixart_vae_fp16-f16.gguf \
    --llm pig_clip-q8_0.gguf --llm-adapter pig_t5_adapter-f16.gguf \
    -p "close-up portrait of a young lady" --diffusion-fa -s 42 -o out.png
```

Judge a run by **cos** over the real slots and **rel_mse**; `cos_eos`
isolates the EOS slot (the hardest one), `cos_rms` is the per-row-normalized
view. The reference run reached val cos 0.827 at 12.5k steps on 117k
prompts; both reference runs flattened around 0.80–0.83 — the 0.6B student's
representation is the ceiling, not data. Use `pig_clip` at f16 or q8_0 in the
engine: the 4-bit student loses ~0.09 cosine before the adapter even runs.

## Requirements

Python ≥ 3.10, PyTorch (CUDA strongly recommended), transformers,
accelerate, safetensors, huggingface_hub, datasets, pyarrow, pillow,
gguf-connector, psutil. Running the LLaDA teacher needs ~34 GB of combined
GPU + CPU memory; the reference run used an RTX 5090 with CPU offload
(4.5 prompts/s, ~3.7 h for 60k prompts). The Qwen3-VL teacher needs ~9 GB
of GPU memory resident, or ~9 GB of RAM plus any CUDA card when streamed.
Training the adapter itself fits in a few GB of VRAM.

Set `GGUF_TRAINER_MOCK_TEACHER=1` (or tick the checkbox under Precompute) to
run the whole pipeline with synthetic targets — a smoke test of the
machinery, never a usable adapter.

## Adding a pack

Subclass `gguf_trainer.packs.base.TrainerPack`: declare the adapter kind
(`resampler` or `token_aligned_vision`), the materials (model or dataset
snapshots, local files), the pack's config defaults and GUI hints, the
prompt template / `build_teacher()` / `build_mock_teacher()`, and the
export key/values, then register it in `gguf_trainer/packs/__init__.py`.
`packs/llada_image.py` and `packs/qwen3vl_mageflow.py` are the two
references.

![screenshot](https://raw.githubusercontent.com/gguf-org/gguf-desktop/master/pizza.jpg)

## Reference
[pig engine - the new gguf compute kernels (gk)](https://github.com/gguf-io/gk)
