# Corpus

The corpus stage turns prompt datasets, local files and (for image packs)
image datasets into the files the precompute stage reads. It is
deterministic for a given `corpus.seed`.

## Text corpus (LLaDA-Image pack)

Output: `data/train.txt` and `data/val.txt`, one prompt per line. Empty lines
are the empty prompt.

Sources, in `corpus.mode: presets`:

| Preset id | Dataset | Column | Size |
| --- | --- | --- | --- |
| `gustavosta` | `Gustavosta/Stable-Diffusion-Prompts` | `Prompt` | ~80k |
| `midjourney` | `succinctly/midjourney-prompts` | `text` | ~250k |
| `diffusiondb` | `poloclub/diffusiondb` (`metadata.parquet`) | `prompt` | ~1.5M unique |
| `vidprom` | `WenhaoWang/VidProM` | `prompt` | ~1.6M |

Presets load through `datasets.load_dataset` (or `hf_hub_download` +
pyarrow for the parquet one) and are cached under `HF_HOME`. A preset that
fails to load is logged as a warning and skipped, so a dead dataset does not
kill the run. Local files from `corpus.extra_files` are added: `.txt` (one
prompt per line) or `.jsonl` (a `text` field per line).

Processing:

1. whitespace is normalised, prompts outside `[min_chars, max_chars]` are
   dropped, duplicates (case-insensitive) are removed;
2. the pool is shuffled; the first `n_val` become validation, the next
   `n_train` training. At least `n_val + 100` prompts are required;
3. `empty_frac` of the training set (and at least one validation prompt) is
   replaced by empty prompts, so the adapter learns the checkpoint's empty
   classifier-free-guidance prompt.

In `corpus.mode: files`, `train_file` and `val_file` are used as they are and
the corpus stage is considered done when both exist.

## Image corpus (MageFlow-Edit pack)

Output: `data/train.jsonl` and `data/val.jsonl`, one JSON object per line
`{"text": str, "images": ["images/<file>", ...]}`, plus the referenced files
copied into `data/images/` (raw bytes, no re-encoding). The stage is skipped
when both jsonl files exist.

Image sources (`corpus.image_presets`, downloaded as materials, and
`corpus.image_folders`):

| Id | Dataset | Format | Notes |
| --- | --- | --- | --- |
| `flickr30k` | `nlphuji/flickr30k` | one zip + annotations CSV (`raw` column holds 5 captions) | 31k photos, 4.4 GB |
| `coco_captions` | `Multimodal-Fatima/COCO_captions_train` | 38 parquet shards with embedded JPEGs, `sentences_raw` captions | 113k photos, 17 GB; the trainer5 reference corpus |
| local folder | any folder | `.jpg .jpeg .png .webp .bmp`, recursive; optional `<name>.txt` next to each image, one caption per line | uncaptioned images get generic instructions |

Sample mix (the trainer5 recipe):

* `n_val_image` images are held out first; **val images never appear in
  train**. Then `n_image` single-image samples are drawn round-robin over
  the remaining images (several instructions per photo when `n_image`
  exceeds the image count), and `n_two` two-image samples pair random
  images.
* `n_text` / `n_val_text` text-only samples come from the prompt presets and
  extra files (same pool as the text corpus) and are cleaned of `()` and
  `[]`, which the engine parses as attention syntax. Set both to 0 to skip
  text-only samples.
* Everything is shuffled together.

### Instruction synthesis

Each image sample's `text` is generated from its captions. The **subject** is
the head noun of a caption after skipping determiners and common adjectives
("A young man in a blue shirt" → "man"). Draws, by probability:

| Share | Instruction |
| --- | --- |
| 1.5 % | empty prompt |
| 28.5 % | a plain caption |
| 12 % | `add <thing>` / `add <thing> to the image` / `the <subject> with <thing>` |
| 8 % | `remove the <subject>` (…`from the image`) |
| 6 % | `replace the <subject> with <thing>` |
| 12 % | `make it <style>` / `turn this into <style>` / `restyle the image as <style>` |
| 8 % | `change the background to <background>` |
| 6 % | `make the <subject> <color>` |
| 8 % | `<caption>, <style>` |
| 4 % | the first 2–5 words of a caption |
| rest | a caption, or `make it <style>` when there is none |

Two-image samples use patterns such as "combine the two images into one
scene", "put the {subject} from Image 1 into Image 2", "apply the style of
Image 2 to Image 1", or a caption of the second image.

The lists of things, styles, backgrounds and colours are in
`vision_data.py`.

## Sizing the image corpus

Every precomputed image sample stores the teacher and student states of
every token plus the raw vision embeds: roughly 2.4 MB per image sample and
~0.4 MB per text sample, about 200 GB for the full reference mix. The GUI
shows a disk estimate from `pack.sample_bytes()`; reduce `n_image` / `n_two`
to fit. Samples longer than 1024 student tokens (very large images or
prompts) make precompute fail with a clear message.
