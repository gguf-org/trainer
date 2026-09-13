# Troubleshooting

Look at the **Logs tab** or `<project>/pipeline.log` first: every failure
logs its traceback prefixed with `!!!`, and `state.json` records the error
under the failing stage. Download problems are in
`<project>/materials/.download-<id>.log`.

## Status and process

**Interrupted.** The state says running but no process is alive: a reboot,
a kill, or a crash before the state was written. Press Start / Resume, or
start the GUI with `--auto-resume`. Nothing is lost beyond the shard in
flight or the steps since the last `last.pt`.

**"pipeline already running (pid N)".** A live pipeline for this project
exists. Stop it first; if the pid is stale but reported alive, the process
name is checked for `gguf_trainer.pipeline`, so a recycled pid is not the
cause. Use the GUI's kill endpoint only for a hung process.

**Stop takes long.** Precompute finishes or discards the current teacher
batch; training finishes the current step. SIGTERM (POSIX) interrupts a long
teacher batch sooner. `gguf-trainer stop --timeout 0` returns immediately.

**Stages show done after clearing the state.** Intended: completion is
derived from the files on disk, not from `state.json`. Use the Reset menu to
delete artifacts.

## Materials

**"downloading needs the huggingface_hub package".** Install it into the
interpreter that runs the server (`<python> -m pip install huggingface_hub`)
and press Refresh.

**Download shows partial after a restart.** Normal. Press Download missing
again; `snapshot_download` resumes the partial blobs.

**A material was linked to the wrong file.** Browse to the right path; the
override is stored in `project.json` under `materials.<id>.path`. Remove the
key to fall back to `<project>/materials/`.

**Gated repositories.** Set the HF token under Materials (`hf_token`), or
export `HF_TOKEN` for the CLI.

**"student pig_clip GGUF not set".** Select the file with Browse, or place
a `pig_clip*.gguf` next to the project or in the directory the GUI was
started from and press Refresh.

**"student GGUF does not match the Qwen3 config".** The selected GGUF is
not a `pig_clip` build (or the tokenizer snapshot is wrong). The loader
expects HF tensor names under `model.` in the Qwen3 layout described by
the `callgg/pig-clip-tokenizer` config; `pig_clip` is its own train /
fine-tune, not a stock Qwen3 checkpoint, but it shares that layout and
tokenizer.

**"gguf-connector cannot dequantize it".** The GGUF uses a quantization the
installed `gguf-connector` lacks the quant module for. Use the f16 file, or
upgrade `gguf-connector` (≥ 3.7.1).

## Corpus

**"only N usable prompts; need more than M".** The presets failed to load
(offline, gated, or a dead dataset; see the warnings above it) or the length
filter is too strict. Add a local file or relax `min_chars` / `max_chars`.

**"image preset 'x' is not downloaded".** Tick the preset, then run
Download missing; image datasets are materials.

**"corpus: only N images; need more than ...".** The folder or preset has
fewer images than `n_val_image + 8`. Lower `n_val_image`.

## Precompute

**CUDA out of memory (LLaDA).** Lower `gpu_mem_gib` by 2–3 GiB, or reduce
`teacher_batch` / `tok_budget`. Switching to `balanced` placement also
works but roughly halves throughput.

**CUDA out of memory (MageFlow).** Set `teacher_mode: offload`, or lower the
budgets (`teacher_batch`, `tok_budget`, `vis_tok_budget`,
`student_batch`, `student_tok_budget`) from their auto values.

**"rotary inv_freq buffers are not initialized"**. The teacher refused to
run with scrambled rotary tables and the automatic repair did not apply.
Check the transformers version in the Hardware tab; the repair is written
for 4.51–5.x. Never train on targets from a teacher in this state; see the
[LLaDA pack](packs/llada-image.md#the-transformers-5-rope-incident).

**"shards were written under teacher contract 'x', current is 'y'".** Not an
error: the shards predate a target-changing fix. They are recomputed, and
old checkpoints are archived to `checkpoints.stale-<time>/`.

**"edit prefix tokenizes to N tokens, the engine hardcodes 64"** or
**"teacher/student tokenizer mismatch".** The tokenizer snapshot differs
from the one the engine and the recipe assume. Re-download
`callgg/pig-clip-tokenizer` and the teacher's tokenizer files.

**"sample of N tokens exceeds max_len 1024".** A very large prompt or
image; shorten `max_chars` or use smaller images (a 384-px long side is the
engine cap, 144 vision tokens).

**"vision_proj.pt missing: run the precompute stage first".** Training was
started with `--only train` before any precompute ran. Run precompute.

## Training

**"checkpoints/last.pt was trained with {...} but the project now asks for
{...}".** `width` or `depth` changed after training started. Restore the
settings, or Reset training.

**val cosine stays low (LLaDA, < 0.9).** Widen to 1536 or deepen before
changing anything else. If `cos_centered` in the final eval sits near 0.87,
suspect stale targets (see the RoPE note above).

**Slow steps.** Check `grad_checkpoint`: `auto` enables it on cards under
12 GB, which trades speed for memory. Set it to `off` on a large card.

## Export and eval

**"nothing to export: checkpoints/best.pt does not exist".** Train first;
`best.pt` is written at the first validation pass and at the end.

**Export box not clickable.** It needs `best.pt` and an idle pipeline.

**Low `roundtrip_cos` (< 0.999).** The GGUF disagrees with the checkpoint:
a broken write or an old `gguf-connector`. Re-export; the writer verifies
the tensor count and layout on read-back.

## Windows

**PermissionError on rename.** Atomic renames retry for 15 s; a persistent
failure means something (an editor, an indexer) holds the file open.

**A console window appears for the pipeline.** Should not happen; the child
is spawned with a hidden console. Report the launcher you use (`py`, venv
redirector).

## Mock teacher

Shards written with `GGUF_TRAINER_MOCK_TEACHER=1` carry
`"mock_teacher": true` in their metadata and are useless for a real
adapter. Reset shards + training before a real run in the same project.
