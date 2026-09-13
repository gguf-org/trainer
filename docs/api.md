# HTTP API

The GUI talks to a stdlib `ThreadingHTTPServer` (`server.py`). All responses
are JSON with `Cache-Control: no-store`; errors are `{"error": "..."}` with a
4xx / 5xx status. NaN and infinities are serialised as `null`. POST bodies
are JSON objects. Every project-scoped call takes the project folder as
`path`.

There is no authentication. The server binds `127.0.0.1` by default.

## GET

| Endpoint | Query | Returns |
| --- | --- | --- |
| `/` , `/index.html`, `/<file>` | | the static GUI (`static/`) |
| `/api/status` | | `version`, `home`, `projects_root`, `last_project`, `packs` (each pack's `describe()`), `corpus_presets`, `image_presets`, `stages`, `stage_titles`, `windows`, `mock_teacher` (env var set) |
| `/api/hardware` | `path` (optional) | the hardware snapshot; with a running project, its process stats |
| `/api/project` | `path`, `online` (`1` default / `0`) | the project payload (below). `online=1` also scans for and links existing materials |
| `/api/log` | `path`, `after` (byte offset) | `{text, next, size}`: the log after `after`, capped at 256 KB; a first read of a big log starts at its tail |
| `/api/metrics` | `path` | `{rows: [{step, loss, cos, rel_mse, val_cos, lr}]}` from `log.csv`, downsampled to ~400 points |
| `/api/download` | `path`, `id` | the download job for a material, or 404 |

## POST

| Endpoint | Body | Effect |
| --- | --- | --- |
| `/api/browse` | `path`, `kind` (`any` / `model` / `text` / `dir`) | directory listing `{path, parent, entries: [{name, path, is_dir, is_project, size}]}`; hidden entries skipped; `model` filters `.gguf` / `.safetensors`, `text` filters `.txt` / `.jsonl` / `.csv` |
| `/api/project/create` | `pack`, `name`, `dir` | creates `<dir>/<name>/project.json` with the pack's defaults (`pig_` prefix added to the export name), remembers it as last project, returns the payload. 400 if it exists |
| `/api/project/open` | `path` | remembers and returns the project |
| `/api/project/save` | `path`, `config` | replaces `project.json` (merged over defaults) |
| `/api/materials/download` | `path`, `id` | starts one detached download job |
| `/api/materials/download_all` | `path`, `include_optional` (default true) | links found copies, starts a job per missing HF material; payload plus `started: [ids]` |
| `/api/pipeline/start` | `path`, `only` (list, optional), `force` (bool), `mock_teacher` (bool) | launches the detached pipeline; `{pid}`. 400 if already running or if `force` without `best.pt` |
| `/api/pipeline/stop` | `path`, `timeout` (s) | STOP file + SIGTERM; `{stopped, pid}` |
| `/api/pipeline/kill` | `path` | force-kill; `{killed}` |
| `/api/project/reset` | `path`, `what` (`corpus` / `shards` / `shards_val` / `train` / `output` / `state`) | deletes the target artifacts; 400 while running |

## The project payload

Returned by create / open / save / project / download_all:

| Field | Content |
| --- | --- |
| `path`, `exists` | project folder |
| `config` | the merged `project.json` |
| `state` | `state.json` |
| `runtime_status`, `pid` | `idle` / `running` / `interrupted` / `stopped` / `failed` / `done` |
| `artifacts` | per-stage completion from disk: `corpus {done, train_file, val_file}`, `precompute_* {done, done_shards, total_shards}`, `train {done, has_last, has_best, step, steps}`, `export {done, path}`, `eval {done, path}` |
| `output_dir`, `output_files` | export folder and this project's `.gguf` files there (`name, path, size, mtime`) |
| `eval` | `eval.json` or null |
| `stages`, `stage_titles` | stage ids and titles |
| `python` | interpreter running the server |
| `pack` | the pack description (`id, title, description, adapter_kind, needs_images, dims, default_name, hints, eval_keys, materials, engine_command`) |
| `materials` | one status per material: the material fields plus `path, wanted, linked, status (ready/missing/partial/downloading), bytes_done, bytes_total, n_files, n_present, missing[:8], job` |
| `adopted` | material ids linked during this call |
| `ready` | every required material is ready |
| `missing_downloads`, `downloading` | ids |
| `engine_command` | the ggk command with the student / adapter / extras paths filled in |
| `sample_bytes` | estimated shard bytes per corpus sample |
| `log_size` | bytes of `pipeline.log` |

## Example

```bash
curl -s -X POST localhost:8655/api/project/create \
  -H 'content-type: application/json' \
  -d '{"pack":"llada_image","name":"llada_adapter","dir":"/data/trainer"}'
curl -s -X POST localhost:8655/api/pipeline/start \
  -H 'content-type: application/json' -d '{"path":"/data/trainer/llada_adapter"}'
curl -s "localhost:8655/api/project?path=/data/trainer/llada_adapter&online=0" | jq .runtime_status
```
