# Pipeline

The pipeline is one process (`python -m gguf_trainer.pipeline --project DIR`)
that runs a project's stages in order. It is launched detached by the GUI and
by `gguf-trainer start`, or in the foreground by `gguf-trainer run`.

## Stages

| Id | Title | Produces | "Done" means |
| --- | --- | --- | --- |
| `corpus` | Corpus | `data/train.txt` + `data/val.txt` (text packs) or `data/train.jsonl` + `data/val.jsonl` + `data/images/` (image packs) | both corpus files exist (for `corpus.mode: files`, the configured files exist) |
| `precompute_val` | Precompute (val) | `shards/val/val-NNNNN.npz` | every expected shard exists under the current contract |
| `precompute_train` | Precompute (train) | `shards/train/train-NNNNN.npz` | same |
| `train` | Train adapter | `checkpoints/last.pt`, `best.pt`, `log.csv` | `last.json` records a step ≥ `train.steps` |
| `export` | Export GGUF | `<output_dir>/<name>-f16.gguf` (+ `<prefix>_sigvq-f16.gguf` for LLaDA) | the GGUF exists and is newer than `best.pt` |
| `eval` | Evaluate | `<project>/eval.json` | `eval.json` exists and is newer than the GGUF |

The number of shards is `ceil(lines / shard_size)` per split, computed from
the corpus files. The student and teacher are loaded once and shared by the
two precompute stages; they are freed before training starts.

## Idempotence and resume

Every stage checks its artifacts before doing work, so re-running a project
is always safe:

* corpus files and each shard are written to a temp file, fsynced and
  renamed; a shard that exists is skipped, so a reboot costs at most the
  shard in flight;
* training saves `last.pt` every `save_every` steps and on Stop / SIGTERM,
  including the optimizer state, the torch RNG state, the best validation
  cosine and the exact position in the shard stream (epoch, shard index,
  batch index), so a resume replays the same batches;
* export and eval rerun only when their inputs are newer than their outputs.

"Done" is derived from the files, not from `state.json`. Deleting a shard
or the GGUF makes the corresponding stage run again on the next start.

Skipped stages are recorded in `state.json` with `"skipped": true`. The eval
stage is recorded as `skipped` when `export.run_eval` is false.

## Stopping

`Stop` in the GUI, `gguf-trainer stop`, SIGINT or SIGTERM all do the same:
set a flag (the `STOP` file in the project, or the signal handler), after
which

* precompute stops before the next shard, or discards the shard in flight if
  a teacher batch was running (it is recomputed on resume);
* training finishes the current step, saves `last.pt`, and returns;
* the pipeline writes `status: stopped` and exits with code 0.

The GUI's `/api/pipeline/kill` endpoint force-kills the process (SIGKILL or
`taskkill /F /T`) and marks the state `stopped`. Use it only when the process
is unresponsive; at most `save_every` steps or one shard are lost.

## `--only` and `--force`

`--only STAGE ...` restricts the run to the named stages; the others are
skipped without being marked. `--force` makes the **export** and **eval**
stages run even when their outputs are current. The GUI's **Export GGUF**
button and the clickable Export / Evaluate stage boxes are exactly
`--only export eval --force` (the server refuses when `checkpoints/best.pt`
does not exist).

## Reset

The Train tab's **Reset…** menu (or `POST /api/project/reset`) deletes
artifacts so stages run again. Targets: `train` (checkpoints + eval.json),
`shards` (shards + checkpoints), `shards_val` (validation shards only, API
only), `corpus` (data + shards + checkpoints), `output` (this project's
exported GGUFs only, API only), `state` (clear `state.json`, delete nothing).
Resets are refused while the pipeline is running.

## The shard contract

Each pack declares a `shard_contract` string (`llada_image/2`,
`qwen3vl_mageflow/1`) that is bumped whenever the meaning of the teacher
targets changes. Every shard directory carries a `CONTRACT` file, and each
shard's metadata records the contract it was written under.

At the start of each precompute split the pipeline compares the directory's
contract with the pack's. If they differ (or the file is missing), every
shard in that split is deleted, `checkpoints/` is renamed to
`checkpoints.stale-<timestamp>/`, and precompute plus training start over.
The reason this exists is documented under the
[LLaDA pack](packs/llada-image.md#the-transformers-5-rope-incident): shards
computed with the broken rotary tables looked plausible but trained adapters
that ignored the prompt.

## Auto-resume after a reboot

`gguf-trainer serve --auto-resume` looks up `last_project` in
`~/.gguf-trainer/settings.json`. If that project exists, its `auto_resume`
setting is true, and its runtime status is `interrupted` (state says running,
no live process), the pipeline is relaunched detached. Put this in a login
item or systemd user service if you want unattended recovery.

## Logging

The pipeline prints timestamped lines to stdout; when detached these go to
`<project>/pipeline.log`, appended, with a `===== launch <time>: <cmd>` banner
per launch. Failures log the full traceback prefixed `!!!` and set
`stages.<stage>.error` and the top-level `error` in `state.json`.
