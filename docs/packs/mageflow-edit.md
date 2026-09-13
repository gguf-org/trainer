# Pack: MageFlow-Edit

Pack id `qwen3vl_mageflow`. MageFlow-Edit conditions its DiT on
**Qwen3-VL-4B-Instruct** final-norm hidden states with the reference image
spliced in as mmproj vision tokens. This pack distills that conditioning
into `pig_clip` + a **token-aligned adapter with a vision extension** (the
trainer5 recipe), so the same generation runs with

```
--llm pig_clip-<quant>.gguf --llm-adapter pig_qwen3vl_4b_adapter-f16.gguf \
--llm_vision mmproj-qwen3vl-4b-it-f16.gguf
```

The 4B mmproj stays exactly as the teacher uses it; nothing else needs
converting.

| Property | Value |
| --- | --- |
| Adapter kind | `token_aligned_vision` |
| Student dim (`in_dim`) | 1024 |
| Target dim (`out_dim`) | 2560 |
| Raw vision embed dim (`vis_dim`) | 2560 |
| Student max length | 1024 tokens |
| Default export name | `qwen3vl_4b_adapter` |
| Shard contract | `qwen3vl_mageflow/1` |
| Needs images | yes |

## The vision bridge

ggk splices the 2560-d mmproj embeds into the LLM input, while the
student embeds at 1024. The adapter owns the bridge in both directions:

* **`vision_proj`** (2560 → 1024, frozen): applied by the *engine* to every
  mmproj embed before the student. It is fitted once per project by ridge
  least squares over the shared vocabulary (both models tokenize
  identically, so row i of both embedding tables is the same token), stored
  in `<project>/vision_proj.pt`, and exported inside the adapter GGUF. The
  precomputed student states bake in the same map bit-for-bit.
* **`vis_in`** (2560 → width, trained, bias-free): hands the adapter the
  *raw* mmproj embeds at vision positions, so vision fidelity does not
  depend on what survives the student.

## The engine contract

Replicated exactly (verified against ggk `SD_DUMP_COND` dumps):

* **Edit template** (reference images present), 64 tokens before the
  first image, `prompt_template_encode_start_idx = 64`:
  ```
  <|im_start|>system\n{EDIT_SYSTEM}<|im_end|>\n<|im_start|>user\n
  per image i:  "Image {i+1}: <|vision_start|>" (6 tokens) + "<|image_pad|>" × n + "<|vision_end|>"
  {text}<|im_end|>\n<|im_start|>assistant\n
  ```
* **Text-to-image template** (no image), 34 tokens of prefix.
* Raw BPE with no special tokens, no padding. The tokenizer is asserted to
  produce exactly 64 / 34 prefix tokens and 6-token image headers, and
  the `<|vision_start|>` / `<|image_pad|>` / `<|vision_end|>` ids are checked
  to sit where the engine expects them (first vision index 64 + 6, advancing
  by `1 + n + 6` per image).
* **Image preprocessing**: sides rounded to a multiple of 32; if the long
  side exceeds 384 the image is scaled by `384 / max_side` with floor
  rounding. Then ggk's `clip_preprocess`: aspect-preserving **nearest**
  resize with integer index math, centre crop, clamp to 0..1, OpenAI-CLIP
  mean / std. Do not "improve" the resize; an antialiased resample changes
  every vision token the teacher saw. Vision tokens per image =
  `(w/32) × (h/32)`, so at most 144 for a 384 × 384 image.
* **Vision tower**: the main merger output only. ggk drops deepstack.
* **Text stack**: vision embeds spliced into `inputs_embeds` at the
  `<|image_pad|>` runs, causal mask, plain 1-D rope (transformers' all-equal
  M-RoPE expansion equals ggk's), **final-norm** `last_hidden_state`
  (`out_layers = {}`).
* Fidelity: HF bf16 vision tower vs engine f16 mmproj cosine 0.998; the text
  stack matches the engine's q4_k_m teacher at 0.975 on text positions.

## Teacher placement

The text stack is ~7.5–8 GB in bf16, the vision tower ~1 GB and always
GPU-resident. `precompute.teacher_mode`:

* `gpu`: text stack resident. Chosen by `auto` when the GPU budget is at
  least the text stack size + 3 GiB. ~44 samples/s on an RTX 5090.
* `offload`: bf16 weights stay in RAM and are streamed through the GPU with
  accelerate's `cpu_offload`; all compute on the GPU. About 1–3.5 samples/s
  on a 6 GB laptop card.
* On a CPU device the mode is `cpu`.

Batch and token budgets at 0 are picked per card (see
[Projects](../project.md#precompute)). The teacher and student tokenizers are
checked to agree on a few templated prompts before any shard is written.

## Corpus

The trainer5 mix: 56k single-image + 3k two-image samples with instructions
synthesized from captions, plus 27k text-only prompts on the text-to-image
template; validation = 1024 image + 512 text samples, val images never in
train. Default image source is Flickr30k (4.4 GB, 31k photos with 5 captions
each, so several instructions per photo); the COCO captions preset (17 GB,
113k photos) is the trainer5 source. Details in [Corpus](../corpus.md).

Each image sample stores ~2.4 MB of shard data; the full mix is ~200 GB.

## Adapter and training

Position i of the student maps to position i of the teacher. Each block is
pre-LN bidirectional self-attention + exact-GELU MLP; the output is
`out_proj(ln_out(x)) + skip(h)`, a direct linear path from the student
states. `out_proj` starts at zero so training begins from the best linear
map. Recipe: width 1024 (its width-gate winner), depth 4, 20k steps, batch
32, lr 2e-4, warmup 1k, whitened MSE + 0.5·(1 − cos) masked to real tokens,
`vision_proj` frozen. Targets are **raw** (final-norm rows are tame); only
the MSE is whitened by `1/sigma`.

Judge a run by **cos_slice** (cosine over the positions the DiT consumes)
and the **cos_vis / cos_txt** split. The trainer5 reference reached val cos
0.915 (vision positions 0.80, text 0.98) and its A/B edits were
near-identical to the teacher's; the 4-step DiT forgives far more than the
cosine suggests. A lagging vision cosine means the `vis_in` path or the
width is the limiter, not the student.

## Exports

`<name>-f16.gguf` with f16 weights and f32 norms, biases, `vision_proj`;
metadata `adapter.vis_dim = 2560`, `adapter.qwen_hidden_source =
final_norm`, `adapter.teacher = Qwen/Qwen3-VL-4B-Instruct`,
`adapter.teacher_tap = final_norm`. The mmproj file is the teacher's own and
is not exported here.

## Use it in ggk

```bash
ggk diffuser engine -- --diffusion-model mageflow-edit-turbo-nvfp4.gguf \
    --vae pig_mageflow_vae_fp32-f16.gguf \
    --llm pig_clip-q8_0.gguf --llm-adapter pig_qwen3vl_4b_adapter-f16.gguf \
    --llm_vision mmproj-qwen3vl-4b-it-f16.gguf \
    --ref-image sheep.png -p "a sheep in sunglasses" --cfg-scale 1.0 \
    --steps 4 --sampling-method euler --diffusion-fa -o out.png
```

Editing wants `pig_clip` at q8_0 or better; text-to-image works with every
quantization.
