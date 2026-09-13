"""Shims so the LLaDA2-MoE remote modeling code (written for transformers
4.51-4.57) also loads AND RUNS CORRECTLY under transformers 5.x.

1. transformers 5 dropped the "default" entry from ROPE_INIT_FUNCTIONS (the
   plain RoPE is computed inline now); the checkpoint's rope_scaling says
   rope_type "default", so the registry lookup raises KeyError.  Re-register
   the 4.x default (inv_freq over head_dim * partial_rotary_factor, scaling 1).

2. transformers 5 builds every model on the meta device and materializes it
   while loading the checkpoint.  Non-persistent buffers are NOT in the
   checkpoint, so they come out as uninitialized memory unless the model's
   `_init_weights` recomputes them — the native models do, remote code
   written for 4.x does not.  The LLaDA2-MoE rotary embedding computes
   `inv_freq` in `__init__` and keeps it as a non-persistent buffer, so the
   loaded teacher ran with garbage rotary frequencies (1e27, 0, ...): the
   backbone still produced plausible rows (same mean/scale as the real
   ones, cos_raw 0.999) but with the positional signal scrambled, and an
   adapter trained on those targets ignores the prompt in the engine.
   `repair_rope_buffers` recomputes every such buffer from the module's own
   `rope_init_fn`, and `check_rope_buffers` refuses a model whose rotary
   tables are still not the plain decreasing 1/theta^(2i/d) ladder.
"""

from __future__ import annotations


def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
    import torch

    base = config.rope_theta
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0) or 1.0
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def ensure_transformers_compat() -> None:
    from transformers import modeling_rope_utils as m

    if "default" not in m.ROPE_INIT_FUNCTIONS:
        m.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _rotary_modules(model):
    for name, mod in model.named_modules():
        if hasattr(mod, "rope_init_fn") and hasattr(mod, "config") and "inv_freq" in getattr(mod, "_buffers", {}):
            yield name, mod


def rope_buffer_is_sane(inv_freq) -> bool:
    """A plain RoPE table starts at 1.0 and decreases monotonically to a
    small positive number; uninitialized memory does not."""
    import torch

    if inv_freq is None or inv_freq.numel() == 0 or inv_freq.device.type == "meta":
        return False
    f = inv_freq.detach().float().cpu()
    if not torch.isfinite(f).all() or abs(float(f[0]) - 1.0) > 1e-4:
        return False
    return bool((f[1:] <= f[:-1]).all() and (f > 0).all())


def repair_rope_buffers(model, log=None) -> int:
    """Recompute non-persistent rotary `inv_freq` buffers left uninitialized
    by a meta-device load.  Returns the number of modules touched (0 = the
    tables were already correct)."""
    import torch

    n = 0
    for name, mod in _rotary_modules(model):
        buf = mod.inv_freq
        if rope_buffer_is_sane(buf):
            continue
        device = buf.device if buf.device.type != "meta" else None
        inv_freq, scaling = mod.rope_init_fn(mod.config, device)
        with torch.no_grad():
            if buf.device.type == "meta" or buf.shape != inv_freq.shape:
                mod.inv_freq = inv_freq.to(device=device)
            else:
                buf.copy_(inv_freq.to(device=buf.device, dtype=buf.dtype))
            mod.original_inv_freq = mod.inv_freq
        mod.attention_scaling = scaling
        if log:
            log(f"rope: recomputed {name}.inv_freq ({int(mod.inv_freq.numel())} freqs; the meta-device "
                f"load had left it uninitialized)")
        n += 1
    return n


def check_rope_buffers(model) -> None:
    bad = [name for name, mod in _rotary_modules(model) if not rope_buffer_is_sane(mod.inv_freq)]
    if bad:
        raise RuntimeError(f"rotary inv_freq buffers are not initialized in {bad}; refusing to run a "
                           f"teacher with scrambled positions (transformers {_tf_version()})")


def _tf_version() -> str:
    try:
        import transformers

        return transformers.__version__
    except Exception:  # pragma: no cover
        return "?"
