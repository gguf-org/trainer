"""Train stage: adapter-only distillation from precomputed shards.

resampler packs (trainer8): loss over all query rows = whitened MSE +
cos_weight * (1 - cosine) on the per-dim STANDARDIZED target
(target - mu) / sigma — the teacher rows carry a large shared per-dim offset
an O(1) head cannot reach, and it makes plain cosine meaningless (0.99 for a
zero prediction).  export.py folds mu/sigma back into out_proj.  Readouts:
rel_mse in raw space, cos in standardized (== centred) space.

token-aligned vision packs (trainer5): per-token loss masked to real tokens =
MSE whitened by 1/sigma per dim + cos_weight * (1 - cosine) on the RAW
target (final-norm rows are tame), out_proj zero-initialized, vision_proj
frozen; the val report splits cosine into vision vs text positions.

Stop-at-any-time: the STOP file or SIGINT/SIGTERM saves last.pt atomically
after the current step; the next run resumes step, optimizer, best-val
tracking and the exact stream position.

Snapshot-at-any-time: the SNAPSHOT file (written by snapshot.py while the
pipeline runs) makes the loop validate and save last.pt (and best.pt when
the score improved) after the current step, then delete the file and keep
training — so a GGUF of the adapter at exactly that step can be exported
and tried in the engine without ending the run.  Every checkpoint's .json
sidecar carries the step and its validation cosine for the manifest.
"""

from __future__ import annotations

import csv
import math
import os
import time

import torch
import torch.nn.functional as F

from .adapter import KIND_TOKEN_VISION, AdapterConfig, adapter_config_for, build_adapter
from .devices import device_index, pick_device
from .shards import ShardStream, ValSet, load_stats
from .util import replace_atomic, write_json_atomic


# ---------------------------------------------------------------- resampler

def loss_fn(pred_std, target, cos_weight, mu, sigma):
    pred_std = pred_std.float()
    target = target.float()
    target_std = (target - mu) / sigma
    mse = ((pred_std - target_std) ** 2).mean()
    cos = F.cosine_similarity(pred_std, target_std, dim=-1).mean()
    pred_raw = pred_std * sigma + mu
    rel_mse = ((pred_raw - target) ** 2).sum() / (target ** 2).sum().clamp_min(1e-8)
    return mse + cos_weight * (1.0 - cos), rel_mse.detach(), cos.detach()


def run_batch(adapter, batch, device, cos_weight, stats):
    mu, sigma = stats
    hidden = batch["qwen_hidden"].to(device).float()
    keep = batch["keep"].to(device)
    target = batch["target"].to(device).float()
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = adapter(hidden, keep)
    else:
        pred = adapter(hidden, keep)
    return loss_fn(pred, target, cos_weight, mu, sigma)


@torch.no_grad()
def validate(adapter, val_set, device, cos_weight, stats):
    adapter.eval()
    tot_mse = tot_cos = 0.0
    for batch in val_set:
        _, rel_mse, cos = run_batch(adapter, batch, device, cos_weight, stats)
        tot_mse += rel_mse.item()
        tot_cos += cos.item()
    adapter.train()
    n = max(1, len(val_set))
    return {"val_rel_mse": tot_mse / n, "val_cos": tot_cos / n}


# ---------------------------------------------------------------- token-aligned (+ vision)

def masked_loss(pred, target, mask, cos_weight, inv_sigma):
    pred = pred.float()
    target = target.float()
    m = mask.unsqueeze(-1)
    err = ((pred - target) ** 2 * m).sum()
    ref = (target ** 2 * m).sum().clamp_min(1e-8)
    rel_mse = err / ref
    d = (pred - target) * inv_sigma
    opt_mse = (d ** 2 * m).sum() / (m.sum().clamp_min(1.0) * pred.shape[-1])
    cos_tok = F.cosine_similarity(pred, target, dim=-1)           # [B, L]
    cos = (cos_tok * mask).sum() / mask.sum().clamp_min(1.0)
    return opt_mse + cos_weight * (1.0 - cos), rel_mse.detach(), cos.detach(), cos_tok.detach()


def run_batch_ta(adapter, batch, device, cos_weight, inv_sigma):
    hidden = batch["qwen_hidden"].to(device).float()
    vis = batch["vis"].to(device).float() if batch.get("vis") is not None else None
    keep = batch["keep"].to(device)
    target = batch["target"].to(device).float()
    mask = keep.float()
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = adapter(hidden, keep, vis)
    else:
        pred = adapter(hidden, keep, vis)
    return masked_loss(pred, target, mask, cos_weight, inv_sigma)


@torch.no_grad()
def validate_ta(adapter, val_set, device, cos_weight, inv_sigma):
    adapter.eval()
    tot_mse = tot_cos = 0.0
    v_sum = v_n = t_sum = t_n = 0.0
    for batch in val_set:
        _, rel_mse, cos, cos_tok = run_batch_ta(adapter, batch, device, cos_weight, inv_sigma)
        tot_mse += rel_mse.item()
        tot_cos += cos.item()
        keep = batch["keep"].to(device)
        is_vis = batch["is_vis"].to(device) & keep
        is_txt = keep & ~is_vis
        v_sum += (cos_tok * is_vis).sum().item()
        v_n += is_vis.sum().item()
        t_sum += (cos_tok * is_txt).sum().item()
        t_n += is_txt.sum().item()
    adapter.train()
    n = max(1, len(val_set))
    return {"val_rel_mse": tot_mse / n, "val_cos": tot_cos / n,
            "val_cos_vis": v_sum / max(v_n, 1.0), "val_cos_txt": t_sum / max(t_n, 1.0)}


# ---------------------------------------------------------------- loop

def train(project, pack, log, report, should_stop) -> str:
    """-> 'done' | 'stopped'"""
    tc = project.config["train"]
    device = pick_device(tc.get("device", "auto"))
    torch.manual_seed(int(tc.get("seed", 42)))
    out_dir = project.checkpoints_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_dir = project.shards_dir / "train"
    val_dir = project.shards_dir / "val"
    steps = int(tc["steps"])
    batch_size = int(tc["batch_size"])
    lr = float(tc["lr"])
    warmup = int(tc["warmup"])
    cos_weight = float(tc["cos_weight"])
    sigma_floor = float(tc["sigma_floor"])
    log_every, val_every, save_every = int(tc["log_every"]), int(tc["val_every"]), int(tc["save_every"])
    token_aligned = pack.adapter_kind == KIND_TOKEN_VISION

    mu, sigma, inv_sigma, floored = load_stats(train_dir, sigma_floor)
    log(f"target sigma: min {sigma.min():.3f} med {sigma.median():.3f} max {sigma.max():.3f}; {floored} dims floored")
    sigma_f = torch.maximum(sigma, torch.tensor(sigma_floor))
    stats = (mu.to(device), sigma_f.to(device))
    inv_sigma_dev = inv_sigma.to(device)

    cfg = adapter_config_for(pack, int(tc["width"]), int(tc["depth"]))
    adapter = build_adapter(cfg)
    if token_aligned:
        vp = project.vision_proj_path()
        if not vp.is_file():
            raise RuntimeError(f"{vp} missing: run the precompute stage first (it fits the frozen vision_proj)")
        adapter.set_vision_proj(torch.load(vp, map_location="cpu", weights_only=True)["weight"])
    adapter.to(device)
    gc = tc.get("grad_checkpoint", "auto")
    if gc == "auto":
        adapter.grad_checkpoint = (device.type == "cuda"
                                   and torch.cuda.get_device_properties(device_index(device)).total_memory < 12e9)
    else:
        adapter.grad_checkpoint = gc in ("on", True, "true", "1")
    trainable = [p for p in adapter.parameters() if p.requires_grad]
    log(f"adapter[{cfg.kind}]: {adapter.num_params() / 1e6:.1f}M params ({adapter.num_trained_params() / 1e6:.1f}M "
        f"trained), grad_checkpoint={adapter.grad_checkpoint}, on {device}, cfg={cfg.to_dict()}")
    opt = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), weight_decay=float(tc["weight_decay"]))

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        t = (step - warmup) / max(1, steps - warmup)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

    project.clear_snapshot()       # a request left over from an earlier run
    start_step = 0
    best_val_cos = -1.0
    stream_state = None
    last = out_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        ck_cfg = AdapterConfig.from_ck(ck.get("config") or {}).to_dict()
        if ck_cfg != cfg.to_dict():
            raise RuntimeError(f"checkpoints/last.pt was trained with {ck_cfg} but the project now asks "
                               f"for {cfg.to_dict()}: reset the training stage or restore the settings")
        adapter.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        best_val_cos = ck.get("best_val_cos", -1.0)
        stream_state = ck.get("stream_state")
        if ck.get("torch_rng") is not None:
            torch.set_rng_state(ck["torch_rng"].byte().cpu())
        log(f"resumed from {last} at step {start_step} (best_val_cos {best_val_cos:.4f})")
        ck_steps = (ck.get("train_config") or {}).get("steps")
        if ck_steps is not None and int(ck_steps) != steps:
            log(f"planned steps changed: {int(ck_steps)} -> {steps} (project settings); the lr schedule "
                f"follows the new count")
        if start_step >= steps:
            log(f"training already complete ({start_step} >= {steps} planned steps; raise train.steps to continue)")
            return "done"

    stream = ShardStream(str(train_dir), batch_size, seed=int(tc.get("seed", 42)), state=stream_state)
    train_iter = iter(stream)
    val_set = ValSet(str(val_dir), batch_size)
    log(f"{len(val_set)} val batches")

    if token_aligned:
        def step_fn(batch):
            loss, rel_mse, cos, _ = run_batch_ta(adapter, batch, device, cos_weight, inv_sigma_dev)
            return loss, rel_mse, cos

        def validate_fn():
            return validate_ta(adapter, val_set, device, cos_weight, inv_sigma_dev)
    else:
        def step_fn(batch):
            return run_batch(adapter, batch, device, cos_weight, stats)

        def validate_fn():
            return validate(adapter, val_set, device, cos_weight, stats)

    log_path = out_dir / "log.csv"
    new_log = start_step == 0 and not log_path.exists()
    log_f = open(log_path, "a", newline="")
    logw = csv.writer(log_f)
    if new_log:
        logw.writerow(["step", "loss", "rel_mse", "cos", "val_rel_mse", "val_cos", "val_cos_vis", "val_cos_txt",
                       "lr", "prompts_per_s", "time"])

    val = {"val_rel_mse": float("nan"), "val_cos": float("nan")}
    val_step = None                # the step `val` was measured at

    def save(step, name):
        path = out_dir / name
        tmp = out_dir / (name + ".tmp")
        torch.save({"config": cfg.to_dict(), "model": adapter.state_dict(), "opt": opt.state_dict(),
                    "step": step, "train_config": dict(tc), "best_val_cos": best_val_cos,
                    "stream_state": stream.state(), "torch_rng": torch.get_rng_state(),
                    "mu": mu, "sigma": sigma_f, "standardized": not token_aligned, "pack": pack.id,
                    "val": dict(val) if val_step == step else None}, tmp)
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        replace_atomic(tmp, path)
        # sidecar for the GUI / snapshot manifest (no torch needed to read it);
        # val_cos is the score of THIS checkpoint when it was validated at its step
        vc = val["val_cos"] if val_step == step and val["val_cos"] == val["val_cos"] else None
        write_json_atomic(out_dir / (name.replace(".pt", ".json")),
                          {"step": step, "best_val_cos": best_val_cos, "steps": steps, "val_cos": vc,
                           "val_step": val_step, "time": time.time()})

    def run_validation(step):
        nonlocal val, val_step, best_val_cos
        val = validate_fn()
        val_step = step
        if val["val_cos"] > best_val_cos:
            best_val_cos = val["val_cos"]
            save(step, "best.pt")
        extra = (f" (vis {val['val_cos_vis']:.4f} txt {val['val_cos_txt']:.4f})"
                 if "val_cos_vis" in val else "")
        log(f"  val @ {step}: rel_mse {val['val_rel_mse']:.4f} cos {val['val_cos']:.4f}{extra} "
            f"(best {best_val_cos:.4f})")

    def poll_planned_steps(step):
        """Pick up a changed train.steps from project.json (GUI "Apply" /
        `gguf-trainer set --steps` / a hand edit) without a restart.  Returns
        the (possibly new) count; a count at or below the current step ends
        the run at this step."""
        nonlocal steps, cfg_mtime
        try:
            m = os.stat(project.config_file).st_mtime_ns
        except OSError:
            return steps
        if m == cfg_mtime:
            return steps
        cfg_mtime = m
        new = project.live_train_steps()
        if new is None or new == steps:
            return steps
        log(f"planned steps changed: {steps} -> {new} (project settings edited at step {step}); "
            + ("finishing now" if new <= step else "the lr schedule follows the new count"))
        steps = new
        tc["steps"] = new           # the checkpoint's train_config records the count in force
        return steps

    try:
        cfg_mtime = os.stat(project.config_file).st_mtime_ns
    except OSError:
        cfg_mtime = None

    adapter.train()
    t_log = time.time()
    t_start = time.time()
    n_prompts = 0
    steps_done_here = 0
    step = start_step
    stopped = False
    while step < steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = next(train_iter)
        loss, rel_mse, cos = step_fn(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, float(tc["grad_clip"]))
        opt.step()
        n_prompts += batch["target"].shape[0]
        steps_done_here += 1

        if (step + 1) % log_every == 0:
            steps = poll_planned_steps(step + 1)
            if steps <= step + 1:
                steps = step + 1        # the shortened run ends here: validate + save as the final step
        snapshot = project.snapshot_requested()
        if (step + 1) % val_every == 0 or step + 1 == steps or snapshot:
            run_validation(step + 1)
        if (step + 1) % log_every == 0:
            dt = time.time() - t_log
            pps = n_prompts / max(dt, 1e-6)
            sps = steps_done_here / max(time.time() - t_start, 1e-6)
            eta = (steps - step - 1) / sps if sps > 0 else None
            log(f"step {step + 1}/{steps} loss {loss.item():.4f} rel_mse {rel_mse.item():.4f} "
                f"cos {cos.item():.4f} lr {lr_at(step):.2e} {pps:.1f} p/s")
            logw.writerow([step + 1, f"{loss.item():.5f}", f"{rel_mse.item():.5f}", f"{cos.item():.5f}",
                           f"{val['val_rel_mse']:.5f}", f"{val['val_cos']:.5f}",
                           f"{val['val_cos_vis']:.5f}" if "val_cos_vis" in val else "",
                           f"{val['val_cos_txt']:.5f}" if "val_cos_txt" in val else "",
                           f"{lr_at(step):.3e}", f"{pps:.1f}", f"{time.time():.0f}"])
            log_f.flush()
            report({"step": step + 1, "steps": steps, "loss": loss.item(), "rel_mse": rel_mse.item(),
                    "cos": cos.item(), "best_val_cos": best_val_cos, **val,
                    "lr": lr_at(step), "prompts_per_s": pps, "eta_s": eta})
            t_log = time.time()
            n_prompts = 0
        stopped = should_stop()
        if (step + 1) % save_every == 0 or stopped or snapshot or step + 1 == steps:
            save(step + 1, "last.pt")
        if snapshot:
            project.clear_snapshot()   # the waiting snapshot job proceeds from here
            log(f"snapshot request: validated and saved last.pt at step {step + 1} "
                f"(val cos {val['val_cos']:.4f}); training continues")
        if stopped:
            log(f"stop requested: saved last.pt at step {step + 1}")
            break
        step += 1
    # after the loop `step` is the last step run (0-based), as the for-loop left it
    step = max(start_step, min(step, steps - 1))
    log_f.close()
    if not (out_dir / "best.pt").exists():
        save(step + 1, "best.pt")
    project.clear_snapshot()
    report({"step": step + 1, "steps": steps, "best_val_cos": best_val_cos, **val})
    log(f"{'stopped' if stopped else 'done'} at step {step + 1}, best val cos {best_val_cos:.4f}")
    return "stopped" if stopped else "done"
