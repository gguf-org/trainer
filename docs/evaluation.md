# Evaluation

The eval stage measures what the DiT actually sees: it runs the **exported
GGUF** (rebuilt the way the engine reads it, so folded weights and the f16
cast count) over the validation shards, and compares it with the checkpoint
it came from. Results go to `<project>/eval.json`.

## Resampler packs (LLaDA-Image)

| Key | Meaning | Judge? |
| --- | --- | --- |
| `cos_centered` | cosine after subtracting the per-dimension target mean `mu` | **yes**: ≥ 0.96 healthy, ~0.87 is the broken-RoPE signature, below ~0.9 widen or deepen |
| `rel_mse` | `Σ(pred − t)² / Σt²` over all rows (scale-aware) | **yes**: 0.0024 on the reference run |
| `cos_rms` | cosine after per-row RMSNorm, which is what the DiT consumes | informative |
| `worst_row_cos_rms` | minimum per-row `cos_rms` over the val set | informative |
| `cos_raw` | plain per-row cosine; inflated by the rows' shared offset (~0.99 even for a zero prediction) | no |
| `roundtrip_cos` | centred cosine between the GGUF's output and the checkpoint's | expect ≥ 0.999 |
| `trained_steps` | from the GGUF metadata | |
| `val_batches` | number of validation batches averaged | |

## Token-aligned vision packs (MageFlow-Edit)

| Key | Meaning | Judge? |
| --- | --- | --- |
| `cos_slice` | per-token cosine averaged per sample over positions ≥ the template start (64 for edit, 34 for text-to-image), which is the conditioning the DiT consumes | **yes** |
| `cos_vis` | per-token cosine over vision positions | **yes**: reference 0.80; lagging means the `vis_in` path or width limits |
| `cos_txt` | per-token cosine over text positions | **yes**: reference 0.98 |
| `cos` | per-token cosine over every real token | informative (reference 0.915) |
| `rel_mse` | relative MSE over real tokens | informative |
| `worst_sample_cos_slice` | minimum `cos_slice` over the val samples | informative |
| `roundtrip_cos` | per-token cosine between the GGUF's output and the checkpoint's | expect ≥ 0.999 |
| `trained_steps`, `val_samples`, `val_batches` | bookkeeping | |

## Common fields

`gguf` (path evaluated), `kind` (`resampler` / `token_aligned_vision`),
`time`, `config` (the adapter geometry read from the GGUF).

## Where the numbers show up

* **Output tab**: the pack's headline keys first (`eval_keys`), the rest
  after.
* `gguf-trainer status`: nothing from `eval.json`; read the file.
* During training, `state.json` and the Train tab chart show `val_cos`
  (standardized / centred cosine for the resampler, per-token cosine for the
  vision kind) plus `val_cos_vis` / `val_cos_txt` for vision packs. These are
  computed from the checkpoint, not the GGUF.

## Re-running

The eval stage reruns automatically when `eval.json` is older than the
GGUF. To force it: click the Evaluate stage box, or

```bash
gguf-trainer start --project DIR --only eval --force
```
