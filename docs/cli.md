# CLI reference

The console script is `gguf-trainer` (entry point `gguf_trainer.__main__:main`).
With no subcommand, or with an unrecognised first argument, it behaves as
`serve`.

```
gguf-trainer [--version] [-h]
gguf-trainer serve    [--host H] [--port P] [--no-browser] [--auto-resume]
gguf-trainer run      --project DIR [--only STAGE ...] [--force]
gguf-trainer start    --project DIR [--only STAGE ...] [--force]
gguf-trainer stop     --project DIR [--timeout S]
gguf-trainer status   --project DIR
gguf-trainer download --project DIR [--id ID ...] [--required-only]
```

## serve

Launches the GUI backend and opens the browser.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Interface to bind. There is no authentication; keep it local unless you trust the network. |
| `--port` | `8655` | Port to listen on. `0` picks a free port. |
| `--no-browser` | off | Do not call `webbrowser.open`. |
| `--auto-resume` | off | Before serving, relaunch the last opened project if its state says "running" but no process is alive (a reboot or kill), and the project's `auto_resume` setting is true. |

Ctrl-C stops the server only. A running pipeline keeps going.

## run

Runs the project's pipeline in the foreground of the current terminal
(`python -m gguf_trainer.pipeline`). Output goes to stdout, and the same
`state.json` / `pipeline.log` bookkeeping applies as in the detached mode.

| Flag | Meaning |
| --- | --- |
| `--project DIR` | Project folder (must contain `project.json`). |
| `--only STAGE ...` | Restrict to these stages. Stage ids: `corpus`, `precompute_val`, `precompute_train`, `train`, `export`, `eval`. |
| `--force` | Re-run `export` and `eval` even if the GGUF and `eval.json` are up to date. Only these two stages honour it. |

Exit codes: `0` when the pipeline finished or was stopped cleanly, `1` when a
stage failed, `2` when there is no project at the path.

## start

Same as `run` but detached: the pipeline gets its own session (`setsid` on
POSIX, a hidden console and new process group on Windows), stdout and stderr
are appended to `<project>/pipeline.log`, and the pid goes to
`<project>/pipeline.pid`. It survives the terminal and the GUI closing.
Refuses to start if a pipeline for the project is already alive.

```bash
gguf-trainer start --project DIR --only export eval --force   # regenerate the GGUF + eval from best.pt
```

## stop

Asks a running pipeline to save and exit: writes `<project>/STOP` and sends
SIGTERM (POSIX). The pipeline finishes its current training step or discards
the shard in flight, saves `last.pt`, and exits.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--timeout` | `30` | Seconds to wait for the process to disappear before returning. |

Prints `stopped` when the process is gone, otherwise `stop requested (pid N
still finishing its step)`. There is no CLI kill; the GUI's `/api/pipeline/kill`
endpoint does that.

## status

Prints the runtime status (`idle`, `running`, `interrupted`, `stopped`,
`failed`, `done`), the pid, and one line per stage with `done` derived from
the files on disk plus the last recorded fields (`step`, `steps`,
`done_shards`, `total_shards`, `val_cos`, `best_val_cos`, `error`).

## download

Foreground, resumable fetch of the materials a project still lacks. Copies
found nearby are linked first (see [Materials](materials.md)); everything
else is downloaded with `huggingface_hub.snapshot_download`. Ctrl-C and rerun
to continue.

| Flag | Meaning |
| --- | --- |
| `--id ID ...` | Restrict to these material ids (for example `teacher`, `sigvq`, `student_tokenizer`, `images_flickr30k`). |
| `--required-only` | Skip optional materials such as the LLaDA SigVQ vision encoder. |

Uses `hf_token` from `project.json`, else the `HF_TOKEN` environment
variable. Local-file materials (the student GGUF) are only reported; select
them in the GUI. Exit codes: `0` ok, `1` at least one failure, `2` no project,
`130` interrupted.

## Module entry points

These are what the GUI and `start` spawn; they are usable directly.

```bash
python -m gguf_trainer.pipeline  --project DIR [--only ...] [--force]
python -m gguf_trainer.materials --project DIR --id MATERIAL_ID        # one detached download job
```

The detached child processes are identified by their command line
(`gguf_trainer.pipeline`, `gguf_trainer.materials`), so a recycled pid that
belongs to another program is not mistaken for a live job.

## Environment variables

| Variable | Effect |
| --- | --- |
| `GGUF_TRAINER_MOCK_TEACHER=1` | Replace the teacher by a deterministic synthetic one. Smoke tests only. |
| `GGUF_TRAINER_MATERIALS=DIR` | An extra directory searched first for pre-downloaded materials. |
| `HF_TOKEN` | Hugging Face token when `project.json` has no `hf_token`. |
| `HF_HOME` | Where `datasets` caches the prompt presets (default `~/.cache/huggingface`). |
| `PYTHONUNBUFFERED` | Set to `1` for spawned children so `pipeline.log` streams. |
