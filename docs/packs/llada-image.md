# Pack: LLaDA-Image-Turbo

Pack id `llada_image`. Replaces the 16B LLaDA2-MoE text stack of
`inclusionAI/LLaDA-Image-Turbo` with `pig_clip` + a **256-query resampler
adapter** (the trainer8 recipe). The adapter is text-only, so it pairs
unchanged with the model's SigVQ vision encoder for image editing.

| Property | Value |
| --- | --- |
| Adapter kind | `resampler` |
| Student dim (`in_dim`) | 1024 |
| Target dim (`out_dim`) | 2560 |
| Query rows | 256 |
| Student max length | 512 tokens |
| Teacher max length | 2048 tokens |
| Default export name | `pig_llada_adapter` |
| Shard contract | `llada_image/2` |

## The target

Teacher target per prompt = the 256 QueryFormer rows of `cap_feats`
(`[256, 2560]`): the LLaDA2-MoE backbone runs over `[tokens ; 256 learned
queries]` with the text masked from seeing the queries, then the 6-layer
`text_projection`. The DiT conditioned on those 256 rows alone reproduces the
full result (trainer8 M0 ablation), so nothing else is distilled.

QueryFormer and text_projection are re-implemented in plain torch
(`llada_teacher.py`), bit-exact against the diffusers originals, and load the
HF safetensors directly, so no diffusers install or reference checkout is
needed. The MoE backbone loads through `trust_remote_code` from the snapshot.
Batching matches `LLaDAImagePipeline._encode_text`: right padding, key-padding
mask, text-to-query block masked, positions = cumsum of the mask.

## Prompt template

Student and teacher see the engine's exact template:

```
<role>HUMAN</role> Generate an image: {text}\n<role>ASSISTANT</role>\n<IMAGE1>
```

An empty prompt becomes `<role>HUMAN</role> Generate an image.` + suffix. The
student tokenizes it with Qwen BPE and no special tokens (the `<role>` and
`<IMAGE1>` markers are plain text there); the teacher's own tokenizer adds
its special tokens.

## Teacher placement

The backbone is placed by accelerate's `device_map` with explicit budgets:

* `sequential` (default): the chosen GPU up to `gpu_mem_gib`, then every
  other CUDA device (its total − 1.5 GiB, if that leaves more than 2 GiB),
  then CPU RAM up to `cpu_mem_gib`. On a 5090 + 4050 the whole 30 GB
  backbone stays on the GPUs.
* `balanced`: transformers' even split. It caps the biggest card at an even
  share of the model and offloads the rest, where each MoE layer runs its
  256 experts eagerly on the CPU: roughly 2.4 vs 4+ prompts/s.

If `sequential` runs out of VRAM, lower the GPU budget by 2–3 GiB rather
than switching. The token budget caps `batch × (text tokens + 256)`.
Reference: RTX 5090 with CPU offload, 4.5 prompts/s, ~3.7 h for 60k prompts.

## Adapter and training

A seedless Perceiver resampler: 256 learned query rows, `depth` blocks of
pre-LN self-attention over the queries, cross-attention to the projected
student states, and an exact-GELU MLP (head_dim 64). Recipe: width 1024,
depth 6, 20k steps, batch 32, lr 2e-4, warmup 1k, cosine to 10 %.

Loss = whitened MSE + 0.5·(1 − cos) on the **per-dimension standardized**
target `(target − mu) / sigma`, because the teacher rows share a large
per-dim offset that an O(1) head cannot reach and that makes plain cosine
meaningless (0.99 for a zero prediction). The export folds `mu` / `sigma`
back into `out_proj`. See [Adapters](../adapters.md).

Judge a run by **val centred cosine / rel_mse**: 0.965 / 0.0024 on the
reference 5090 run (~55 min of training). Below ~0.9, widen (1536) or deepen
before touching the rest of the recipe.

## Exports

* `<prefix>_adapter-f16.gguf` (or `<name>-f16.gguf`): the adapter, with
  metadata `adapter.qwen_hidden_source = final_norm`,
  `adapter.target = "llada_image_turbo text_projection query rows"`,
  `adapter.teacher = inclusionAI/LLaDA-Image-Turbo`.
* `<prefix>_sigvq-f16.gguf` when `export.export_sigvq` is true and the
  `sigvq` material is present: the SigVQ safetensors cast to f16 with the HF
  tensor names verbatim (491 tensors), what ggk loads via `--llm_vision`. It
  is rebuilt only when the safetensors are newer than the GGUF.

## Use it in ggk (≥ 0.5.7)

```bash
ggk diffuser engine -- --diffusion-model LLaDA-image-turbo-nvfp4.gguf \
    --vae pig_flux2_vae_fp32-f16.gguf \
    --llm pig_clip-q8_0.gguf --llm-adapter pig_llada_adapter-f16.gguf \
    --llm_vision pig_llada_sigvq-f16.gguf \
    --ref-image sheep.png -p "a sheep in sunglasses" --cfg-scale 1.0 \
    --steps 4 --sampling-method euler --diffusion-fa -o out.png
```

Text-to-image works with any student quantization; editing wants `pig_clip`
at q8_0 or better and prefers the f16 DiT (trainer8 A/B).

## The transformers 5 RoPE incident

transformers 5 builds models on the meta device and does not re-initialise
the non-persistent buffers of remote (`trust_remote_code`) models. The
LLaDA2-MoE rotary embedding computes `inv_freq` in `__init__` and keeps it as
a non-persistent buffer, so the loaded teacher ran with garbage rotary
frequencies. It still emitted plausible rows (same mean and scale, plain
cosine 0.999 to the real ones) with the positional signal scrambled, and
adapters trained on those targets ignored the prompt in the engine (0.1–0.2
lower centred cosine against the true teacher; edits returned the reference
image).

Since 0.0.3 (`llada_compat.py`):

* the 4.x `"default"` rope-init function is re-registered (transformers 5
  dropped it and the checkpoint's `rope_scaling` names it);
* every rotary `inv_freq` buffer is recomputed after loading and the teacher
  refuses to run unless the tables are the plain decreasing `1/theta^(2i/d)`
  ladder;
* the shard contract was bumped to `llada_image/2`. Shards written under
  contract 1 or none are discarded on the next run, checkpoints trained on
  them move to `checkpoints.stale-<time>/`, and precompute + training start
  over.

A quick health check of any adapter is its centred cosine against a
teacher-conditioned engine context: ≥ 0.96 is healthy, ~0.87 is the
broken-RoPE signature. With the repair the teacher matches the trainer8
engine dump at centred cosine 0.998 under transformers 5.15 and 4.57 alike.
