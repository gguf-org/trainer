# Adapters and export

Two adapter architectures ship, both under the engine's `adapter.` tensor
prefix (ggk `llm_adapter.hpp`), head_dim 64, LayerNorm eps 1e-5, exact GELU.
The engine tells them apart by tensor names alone.

## Resampler (`resampler`, LLaDA-Image)

Selected by the **presence** of `adapter.query`.

```
kv  = in_proj(h)                        # student final-norm states, in_dim -> width
q   = query                             # num_queries learned rows, no token-identity seed
q   = Block(q, kv) × depth              # pre-LN self-attn over q, cross-attn to kv, exact-GELU MLP
out = out_proj(ln_out(q))               # width -> out_dim
```

Each `Block` holds `ln_self`, `self_attn` (q/k/v/o linears), `ln_q`,
`ln_kv`, `cross_attn`, `ln_mlp`, and `mlp` (width → 4·width → width). The
student's padding mask is applied as a key mask in the cross-attention.

Parameters at the reference size (width 1024, depth 6): the 256 × 1024
query, the projections, and six blocks, about 90M parameters.

## Token-aligned + vision extension (`token_aligned_vision`, MageFlow-Edit)

Selected by the **absence** of `adapter.query`; the vision extension by the
presence of `adapter.vision_proj.weight`.

```
x   = in_proj(h) + vis_in(v)            # 1024 -> width, 2560 -> width (v = 0 at text positions)
x   = TokenBlock(x) × depth             # pre-LN bidirectional self-attn + exact-GELU MLP
out = out_proj(ln_out(x)) + skip(h)     # width -> out_dim, plus a direct linear path
```

`v` is the raw mmproj embed at vision positions. `vis_in` has no bias (zero
input → zero contribution); `out_proj` is zero-initialised so training
starts from the best linear map of the student states. The module also
carries the frozen `vision_proj` (`[in_dim, vis_dim]`), which the engine
applies to every mmproj embed before the student LLM.

## Training losses

Implemented in `train.py`.

**Resampler.** The teacher rows carry a large shared per-dimension offset an
O(1) head cannot reach, so the loss works on the standardized target
`t_std = (t − mu) / sigma`, where `mu` and `sigma` are the per-dimension
moments accumulated over the training shards and `sigma` is floored at
`train.sigma_floor`:

```
loss = mean((pred − t_std)²) + cos_weight · (1 − cos(pred, t_std))
```

Readouts: `rel_mse` in raw space (after un-standardizing the prediction),
`cos` in standardized (equivalently, centred) space. `best.pt` tracks the
best validation cosine.

**Token-aligned.** Per-token loss masked to real tokens on the raw target:

```
loss = mean_over_tokens_and_dims(((pred − t) / sigma)²) + cos_weight · (1 − mean_token_cos(pred, t))
```

Validation additionally splits the cosine into vision and text positions.

Common to both: AdamW (betas 0.9 / 0.95), linear warmup then cosine decay
from `lr` to 10 % of it, gradient-norm clipping, bf16 autocast on CUDA,
optional activation checkpointing (automatic on cards under 12 GB). The
shard stream is an infinite, seeded permutation of shards and rows whose
position is part of the checkpoint, so a resume replays the exact sequence.

## Checkpoints

`checkpoints/last.pt` and `best.pt` are torch pickles with:

| Key | Content |
| --- | --- |
| `config` | `AdapterConfig` dict: `in_dim`, `out_dim`, `width`, `depth`, `num_queries`, `mlp_ratio`, `kind`, `vis_dim` |
| `model` | adapter state dict |
| `opt` | AdamW state |
| `step` | steps completed |
| `train_config` | the `train` section at save time |
| `best_val_cos` | best validation cosine so far |
| `stream_state` | `{epoch, shard_i, batch_i}` |
| `torch_rng` | torch RNG state |
| `mu`, `sigma` | per-dimension target moments (sigma floored) |
| `standardized` | true for resampler checkpoints |
| `pack` | pack id |

`last.json` / `best.json` next to them hold `step`, `best_val_cos`, `steps`,
`time` for cheap reads. A resume with a different `width` / `depth` is
refused.

## Export to GGUF

`export.py` rebuilds the model from `best.pt`, and for a resampler folds the
standardization into the output layer so the engine emits teacher-scale
rows:

```
W' = diag(sigma) · W        b' = sigma ⊙ b + mu
```

Then it writes `<name>-f16.gguf` with `gguf_connector`'s writer
(architecture string `pig`):

* **Tensors** under `adapter.<state_dict name>`. 1-D tensors (biases, norm
  weights), `query`, `vision_proj.weight` and every LayerNorm weight stay
  **f32**; all other 2-D weights go **f16**.
* **Metadata**:

| Key | Type | Value |
| --- | --- | --- |
| `general.name` | string | the project's `name` |
| `adapter.in_dim` | u32 | 1024 |
| `adapter.out_dim` | u32 | 2560 |
| `adapter.vis_dim` | u32 | vision packs only |
| `adapter.width`, `adapter.depth`, `adapter.heads` | u32 | geometry (`heads = width / 64`) |
| `adapter.num_queries` | u32 | resampler only |
| `adapter.trained_steps` | u32 | step of `best.pt` |
| `adapter.pack` | string | pack id |
| `adapter.trainer` | string | `gguf-trainer` |
| pack extras | mixed | `adapter.qwen_hidden_source`, `adapter.teacher`, `adapter.target` / `adapter.teacher_tap` |

The file is written to a temp name, read back to verify the tensor count and
the layout (query present for a resampler; `skip.weight`,
`vision_proj.weight`, `vis_in.weight` present and no query for the vision
kind), then renamed into place. If `export.copy_to` is set the file is also
copied there.

`gguf_adapter_model()` rebuilds the adapter from a GGUF exactly the way the
engine reads it; the evaluation stage uses it so the folded weights and the
f16 cast are part of the reported numbers.

## Companion export: SigVQ

For the LLaDA pack, `sigvq_export.py` casts `sigvq/diffusion_pytorch_model.safetensors`
to f16 and writes the tensors under their HF names (no `sigvq.` prefix; the
engine adds its own and reads no metadata) as `<prefix>_sigvq-f16.gguf`.
