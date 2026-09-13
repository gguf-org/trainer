# Data formats

## Corpus files

* **Text packs**: `data/train.txt`, `data/val.txt`. One prompt per line,
  UTF-8, newlines inside prompts replaced by spaces. An empty line is the
  empty prompt.
* **Image packs**: `data/train.jsonl`, `data/val.jsonl`. One object per
  line: `{"text": "make it night", "images": ["images/flickr30k_123.jpg"]}`.
  `images` has 0, 1 or 2 relative paths under `data/`. Absolute paths are
  accepted too.

Both are written atomically (`.tmp` + rename).

## Shards

`shards/<split>/<split>-NNNNN.npz` (numpy, `allow_pickle=False`), one per
`shard_size` samples, samples **sorted by student token length** inside a
shard. bf16 values are stored as their raw bits in `int16` arrays. A `.tmp.npz`
is a shard in flight and is ignored.

Common arrays:

| Array | Shape / dtype | Meaning |
| --- | --- | --- |
| `prompts` | `[S]` str | the sample texts in stored order |
| `q_pack` | `[T, in_dim]` int16 (bf16 bits) | student final-norm states of every real token, samples concatenated |
| `len` | `[S]` int32 | student tokens per sample; offsets are its cumsum |
| `ids` | `[T]` int32 | student token ids |
| `stat_s`, `stat_s2` | `[out_dim]` float64 | per-dimension sum and sum of squares of the targets |
| `stat_n` | `[1]` | number of target rows in the shard |
| `num_queries` | `[1]` | 256 for resampler shards, 0 for token-aligned |
| `meta` | `[1]` str | JSON: `pack`, `student` file name, `contract`, `mock_teacher`, `storage`, `shard`, `time` (+ `teacher_mode`, `tap` for vision packs) |

Resampler packs add:

| Array | Shape | Meaning |
| --- | --- | --- |
| `t_pack` | `[S × 256, out_dim]` int16 | the 256 teacher rows per sample |

Token-aligned vision packs add:

| Array | Shape | Meaning |
| --- | --- | --- |
| `t_pack` | `[T, out_dim]` int16 | one teacher row per real token, packed like `q_pack` |
| `n_images` | `[S]` int32 | images per sample (selects the template start index at eval) |
| `v_pack` | `[Tv, vis_dim]` int16 | raw mmproj vision embeds, all images concatenated |
| `vis_sample`, `vis_start`, `vis_len` | `[n_segments]` int32 | segment table: sample index, position of the `<\|image_pad\|>` run, its length; rows in `v_pack` follow the same order |

`shards.load_stats()` sums the moment arrays over every training shard to
get `mu`, `sigma` and `1/sigma` for whitening.

A `CONTRACT` text file next to the shards holds the pack's shard contract
string; see [Pipeline](pipeline.md#the-shard-contract).

## Checkpoints and training log

`checkpoints/last.pt`, `best.pt`, `last.json`, `best.json`: see
[Adapters](adapters.md#checkpoints).

`checkpoints/log.csv` gets a row every `log_every` steps:

```
step,loss,rel_mse,cos,val_rel_mse,val_cos,val_cos_vis,val_cos_txt,lr,prompts_per_s,time
```

`val_*` repeat the last validation result (NaN before the first one);
`val_cos_vis` / `val_cos_txt` are empty for resampler packs. The GUI chart
reads this file.

## vision_proj.pt

Vision packs only. A torch pickle `{"weight": [1024, 2560] float32,
"ridge": 1e-4, "fit_cos_mean": float, "pack": id, "vocab": 151936}`. Fitted
once per project during precompute; training refuses to start without it.
Delete it to refit (the shards must then be recomputed too, since the
student states bake the map in).

## eval.json

See [Evaluation](evaluation.md).

## Downloads bookkeeping

`materials/.download-<id>.pid` (pid), `.log` (child output), `.status` (JSON
`{"status": running|done|failed, "started", "finished", "bytes_at_start",
"error"}`).
