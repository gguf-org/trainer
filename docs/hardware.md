# Hardware

## Requirements by stage

| Stage | LLaDA-Image pack | MageFlow-Edit pack |
| --- | --- | --- |
| Corpus | CPU, network for the datasets | CPU, disk for the copied images |
| Precompute | the 33 GB bf16 LLaDA2-MoE backbone needs ~34 GB of **combined GPU + CPU** memory; the student adds ~1.2 GB | ~9 GB of GPU memory to keep the Qwen3-VL text stack resident, or ~9 GB of RAM plus any CUDA card when streamed; the vision tower (~1 GB) is always on the GPU |
| Train | a few GB of VRAM (activation checkpointing switches on automatically under 12 GB) | same |
| Export / eval | CPU or GPU, seconds to a minute | same |

Reference throughput:

| Setup | Precompute | Train |
| --- | --- | --- |
| RTX 5090, LLaDA, sequential placement with CPU offload | 4.5 prompts/s, ~3.7 h for 60k prompts | ~55 min for 20k steps |
| RTX 5090, MageFlow, text stack resident | ~44 samples/s | comparable |
| 6 GB laptop GPU, MageFlow, offload | ~1–3.5 samples/s | slower, checkpointed |

## Device selection

`precompute.device` and `train.device` accept any torch device string
(`cuda:1`, `cpu`). `auto` picks the CUDA device with the most VRAM (torch
orders devices fastest-first, so an index is never hardcoded), else CPU.

## Memory budgets (LLaDA teacher)

`precompute.gpu_mem_gib` and `cpu_mem_gib` become accelerate `max_memory`
entries. `0` means: GPU = total VRAM of the chosen device − 1.5 GiB; CPU =
available RAM − 6 GiB (at least 4 GiB, 40 GiB if psutil is missing). Every
other CUDA device is added automatically with its total − 1.5 GiB when that
leaves more than 2 GiB, and it is filled before the CPU, because a
CPU-resident MoE layer runs its 256 experts eagerly and halves throughput.

`placement: sequential` fills the chosen GPU to its budget, then the other
GPUs, then RAM. `balanced` is transformers' even split, which caps the big
card at an even share and offloads the rest (about half the speed). Out of
VRAM with `sequential`? Lower the GPU budget by 2–3 GiB.

transformers 5 keeps CPU-offloaded layers on the meta device and streams
them from the memory-mapped safetensors at forward time, so the CPU budget
is a ceiling, not an allocation.

## Teacher mode (Qwen3-VL teacher)

`teacher_mode: auto` keeps the text stack on the GPU when the GPU budget is
at least the bf16 text stack size + 3 GiB, otherwise streams it with
accelerate `cpu_offload` (weights in RAM, compute on the GPU). Force either
with `gpu` / `offload`. Batch, token and vision-token budgets at `0` are
chosen per card: a card over 24 GB gets the large profile, everything else
the laptop profile (see [Projects](project.md#precompute)).

## Windows notes

The detached pipeline and downloads run in a hidden console of their own
(`CREATE_NO_WINDOW` + `CREATE_NEW_PROCESS_GROUP`), so a venv launcher does
not pop up a visible window. Atomic renames retry for up to 15 s on sharing
violations, since the GUI, an antivirus or an indexer may briefly hold a
file open. `gguf-trainer stop` cannot send SIGTERM on Windows; it relies on
the `STOP` file, which the pipeline checks between steps and shards.
`nvidia-smi` is also looked for under `/usr/lib/wsl/lib` for WSL.

## The Hardware tab

Shows GPUs from `nvidia-smi` (index, name, VRAM total / used, utilisation,
temperature, power) and from torch (name, total memory), RAM total and
available, CPU load, free disk at the project path, and the environment
versions. The Train tab's live readings refresh every 3 s and include the
pipeline process's CPU %, RSS and thread count.
