"""Precompute stage: resumable sharded teacher targets + student states.

Shards (bf16 bits as int16, written .tmp + fsync + os.replace, skipped when
present — a reboot costs at most the shard in flight).  Each split directory
carries a CONTRACT marker = the pack's shard_contract; shards under another
contract are discarded (with the checkpoints trained on them) and rebuilt.

  resampler packs (text corpus, one line per prompt):
    prompts [S]     t_pack [S*NQ, out_dim]   q_pack [T, in_dim]
    len [S]         ids [T]                  stat_s/stat_s2/stat_n (target moments)
  token-aligned vision packs ((image, instruction) jsonl corpus):
    prompts [S]  n_images [S]  t_pack [T, out_dim]  q_pack [T, in_dim]
    v_pack [Tv, vis_dim] + vis_sample/vis_start/vis_len segment table
    len [S]  ids [T]  stat_s/stat_s2/stat_n over every real token
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import List

import numpy as np
import torch

from .devices import auto_mem_budgets, device_index, pick_device
from .llada_teacher import use_mock_teacher
from .student import load_student, tokenize_student
from .util import replace_atomic


def bf16_bits(t):
    return t.contiguous().view(torch.int16).numpy()


def read_samples(path: pathlib.Path) -> list:
    """train.txt -> [str]; train.jsonl -> [{"text", "images": [...]}]"""
    path = pathlib.Path(path)
    with open(path, encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(l) for l in f if l.strip()]
        return [l.rstrip("\n") for l in f]


def shard_plan(project, split: str):
    corpus = project.corpus_files()[split]
    samples = read_samples(corpus)
    size = int(project.config["precompute"]["val_shard_size" if split == "val" else "shard_size"])
    out_dir = project.shards_dir / split
    n_shards = max(1, (len(samples) + size - 1) // size)
    todo = [i for i in range(n_shards) if not (out_dir / f"{split}-{i:05d}.npz").exists()]
    return samples, size, out_dir, n_shards, todo


def discard_stale_shards(project, split: str, log) -> int:
    """Delete shards written under another teacher contract and archive the
    checkpoints trained on them (checkpoints -> checkpoints.stale-<time>), so
    the pipeline recomputes the targets and retrains instead of resuming."""
    stale = project.stale_shards(split)
    if not stale:
        return 0
    have = project.shard_contract_on_disk(split) or "none"
    want = project.shard_contract()
    log(f"precompute[{split}]: {len(stale)} shard(s) were written under teacher contract '{have}', "
        f"current is '{want}' — their targets are stale and will be recomputed")
    d = project.shards_dir / split
    for f in stale:
        os.remove(d / f)
    try:
        os.remove(d / project.SHARD_CONTRACT_FILE)
    except OSError:
        pass
    ck = project.checkpoints_dir
    if ck.is_dir() and any(ck.iterdir()):
        dst = ck.with_name(f"{ck.name}.stale-{time.strftime('%Y%m%d-%H%M%S')}")
        os.rename(ck, dst)
        log(f"precompute[{split}]: checkpoints trained on the stale shards moved to {dst.name}; "
            f"training restarts from scratch once the shards are rebuilt")
    return len(stale)


def write_shard_contract(project, split: str) -> None:
    d = project.shards_dir / split
    d.mkdir(parents=True, exist_ok=True)
    (d / project.SHARD_CONTRACT_FILE).write_text(project.shard_contract() + "\n", encoding="utf-8")


def auto_budgets(pc: dict, dev: torch.device) -> dict:
    """0 = pick for the card: a 6 GB laptop GPU vs a 24 GB+ desktop card
    (trainer5's two profiles)."""
    big = dev.type == "cuda" and torch.cuda.get_device_properties(device_index(dev)).total_memory > 24e9
    defaults = {"teacher_batch": (512 if big else 128), "tok_budget": (65536 if big else 12288),
                "student_batch": (256 if big else 64), "student_tok_budget": (131072 if big else 24576),
                "vis_tok_budget": (8192 if big else 1024)}
    return {k: (int(pc.get(k) or 0) or v) for k, v in defaults.items()}


def ensure_vision_proj(project, pack, teacher, student_table: torch.Tensor, log) -> torch.Tensor:
    """The frozen vision_proj [in_dim, vis_dim] of a vision pack: fitted once
    per project from the teacher's and the student's embedding tables and
    kept in <project>/vision_proj.pt (deterministic; the engine and the
    precomputed student states share it bit-for-bit)."""
    path = project.vision_proj_path()
    if path.is_file():
        d = torch.load(path, map_location="cpu", weights_only=True)
        w = d["weight"] if isinstance(d, dict) else d
        if tuple(w.shape) != (pack.in_dim, pack.vis_dim):
            raise RuntimeError(f"{path} has shape {tuple(w.shape)}, expected {(pack.in_dim, pack.vis_dim)}")
        log(f"vision_proj: loaded {path.name} (vocab-fit cos {d.get('fit_cos_mean', float('nan')):.4f})"
            if isinstance(d, dict) else f"vision_proj: loaded {path.name}")
        return w.float()
    from .qwen3vl_teacher import fit_vision_proj

    t_table = teacher.embed_table() if teacher is not None else None
    if t_table is None:
        mat = pack.material("teacher", project)
        root = pack.material_path(project, mat) if mat else None
        idx = root / "model.safetensors.index.json" if root else None
        if idx and idx.is_file():
            from .qwen3vl_teacher import load_embed_table

            t_table = load_embed_table(root).float()
        else:
            # mock run without the teacher on disk: a deterministic stand-in
            log("vision_proj: MOCK (teacher not downloaded) — random projection, never usable")
            g = torch.Generator().manual_seed(7)
            t_table = torch.randn(student_table.shape[0], pack.vis_dim, generator=g)
    log(f"vision_proj: fitting {pack.vis_dim} -> {pack.in_dim} over {t_table.shape[0]} vocabulary rows ...")
    w, fit_cos = fit_vision_proj(t_table, student_table.float(), log=log)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"weight": w, "ridge": 1e-4, "fit_cos_mean": fit_cos, "pack": pack.id,
                "vocab": int(t_table.shape[0])}, tmp)
    replace_atomic(tmp, path)
    log(f"vision_proj: wrote {path}")
    return w


class PrecomputeContext:
    """Student + teacher loaded once and shared by the val and train splits."""

    def __init__(self, project, pack, log):
        pc = project.config["precompute"]
        self.dev = pick_device(pc.get("device", "auto"))
        student_gguf = pack.material_path(project, pack.material("student", project))
        hf_dir = pack.material_path(project, pack.material("student_tokenizer", project))
        if not student_gguf or not pathlib.Path(student_gguf).is_file():
            raise RuntimeError("student pig_clip GGUF not set (Setup → Materials)")
        if not hf_dir or not (pathlib.Path(hf_dir) / "config.json").is_file():
            raise RuntimeError("student tokenizer/config not downloaded (Setup → Materials)")
        sdtype = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}[pc.get("student_dtype", "bf16")]
        if self.dev.type == "cpu" and sdtype == torch.float16:
            sdtype = torch.float32
        self.sdtype = sdtype
        self.student, self.tok = load_student(student_gguf, hf_dir, self.dev, sdtype, log)
        self.student_name = pathlib.Path(student_gguf).name
        self.kind = pack.adapter_kind
        self.mock = use_mock_teacher()
        if self.mock:
            self.teacher = pack.build_mock_teacher(log)
        else:
            gpu_mem, cpu_mem = auto_mem_budgets(self.dev, float(pc.get("gpu_mem_gib", 0)),
                                                float(pc.get("cpu_mem_gib", 0)))
            self.teacher = pack.build_teacher(project, self.dev, gpu_mem, cpu_mem, log)
        self.pack = pack
        self.project = project
        self.log = log
        if self.kind == "token_aligned_vision":
            from .vision_data import assert_template_counts

            assert_template_counts(self.tok)
            if not self.mock:
                self._check_tokenizers()
            self.s_embed = self.student.embed_tokens.weight.detach()
            w = ensure_vision_proj(project, pack, None if self.mock else self.teacher, self.s_embed.float().cpu(), log)
            self.w_vis = w.to(self.dev)
            self.budgets = auto_budgets(pc, self.dev)
            log(f"precompute budgets: {self.budgets}")
            self.data_root = project.data_dir

    def _check_tokenizers(self):
        """teacher and student must tokenize the templated prompts identically
        (both are the Qwen BPE; trainer5 asserted it, so do we)."""
        from .vision_data import build_sample

        ttok = getattr(self.teacher, "tok", None)
        if ttok is None:
            return
        for p in ["a sheep in sunglasses", "make it night", "", "Add snow, heavy"]:
            a, _ = build_sample(self.tok, p, [144])
            b, _ = build_sample(ttok, p, [144])
            if a != b:
                raise RuntimeError(f"teacher/student tokenizer mismatch on {p!r}")

    def run_split(self, split: str, report, should_stop) -> bool:
        """-> True when every shard of the split exists, False when stopped."""
        if self.kind == "token_aligned_vision":
            return self._run_split_token_aligned(split, report, should_stop)
        return self._run_split_resampler(split, report, should_stop)

    # ------------------------------------------------------------ resampler
    def _run_split_resampler(self, split: str, report, should_stop) -> bool:
        project, pack, log = self.project, self.pack, self.log
        pc = project.config["precompute"]
        discard_stale_shards(project, split, log)
        prompts, size, out_dir, n_shards, todo = shard_plan(project, split)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_shard_contract(project, split)
        log(f"precompute[{split}]: {len(prompts)} prompts, {n_shards} shards, {len(todo)} to do -> {out_dir}")
        report({"done_shards": n_shards - len(todo), "total_shards": n_shards, "prompts": len(prompts)})
        if not todo:
            return True
        NQ, T_DIM, S_DIM = pack.num_queries, pack.out_dim, pack.in_dim
        teacher_batch = int(pc.get("teacher_batch") or 24)
        tok_budget = int(pc.get("tok_budget") or 12288)
        student_batch = int(pc.get("student_batch") or 64)
        dev = self.dev
        rate_hist = []

        for k, si in enumerate(todo):
            if should_stop():
                log(f"precompute[{split}]: stop requested before shard {si:05d}")
                return False
            t0 = time.time()
            chunk = prompts[si * size: (si + 1) * size]
            S = len(chunk)
            ids_s, lens_s = tokenize_student(chunk, self.tok, pack.format_prompt, pack.max_len_student)
            order = np.argsort(lens_s, kind="stable")           # length-sorted storage
            chunk = [chunk[i] for i in order]
            ids_s, lens_s = ids_s[order], lens_s[order]
            off = np.zeros(S + 1, dtype=np.int64)
            np.cumsum(lens_s, out=off[1:])
            T = int(off[-1])
            t_pack = np.empty((S * NQ, T_DIM), dtype=np.int16)
            q_pack = np.empty((T, S_DIM), dtype=np.int16)
            ids_pack = np.empty(T, dtype=np.int32)
            s = np.zeros(T_DIM, dtype=np.float64)
            s2 = np.zeros(T_DIM, dtype=np.float64)

            # ---- teacher (token budget counts text + query rows) ----
            tt = time.time()
            b0 = 0
            while b0 < S:
                if should_stop():
                    log(f"precompute[{split}]: stop requested inside shard {si:05d} (shard discarded)")
                    return False
                b1 = b0 + 1
                while b1 < S and b1 - b0 < teacher_batch:
                    l_text = self.teacher.text_len(chunk[b1])
                    if (b1 - b0 + 1) * (l_text + NQ) > tok_budget:
                        break
                    b1 += 1
                cap = self.teacher(chunk[b0:b1])                                # [b, NQ, T_DIM]
                t_pack[b0 * NQ: b1 * NQ] = bf16_bits(cap.reshape(-1, T_DIM))
                v = cap.double().reshape(-1, T_DIM).numpy()
                s += v.sum(0)
                s2 += (v * v).sum(0)
                b0 = b1
                report({"done_shards": n_shards - len(todo) + k, "total_shards": n_shards,
                        "shard_progress": b0 / S, "prompts": len(prompts)})
            t_teacher = time.time() - tt

            # ---- student ----
            with torch.no_grad():
                for b0 in range(0, S, student_batch):
                    rows = np.arange(b0, min(b0 + student_batch, S))
                    Lq = int(lens_s[rows].max())
                    ids = torch.from_numpy(ids_s[rows][:, :Lq].astype(np.int64)).to(dev)
                    mask = (torch.arange(Lq, device=dev)[None, :]
                            < torch.from_numpy(lens_s[rows]).to(dev)[:, None]).long()
                    h = self.student(input_ids=ids, attention_mask=mask).last_hidden_state.to(torch.bfloat16).cpu()
                    for j, r in enumerate(rows):
                        li = int(lens_s[r])
                        q_pack[off[r]: off[r + 1]] = bf16_bits(h[j, :li])
                        ids_pack[off[r]: off[r + 1]] = ids_s[r, :li]

            path = out_dir / f"{split}-{si:05d}.npz"
            tmp = out_dir / f"{split}-{si:05d}.tmp.npz"
            np.savez(tmp, prompts=np.array(chunk), t_pack=t_pack, q_pack=q_pack,
                     len=lens_s.astype(np.int32), ids=ids_pack,
                     stat_s=s, stat_s2=s2, stat_n=np.array([S * NQ]), num_queries=np.array([NQ]),
                     meta=np.array([json.dumps({"pack": pack.id, "student": self.student_name,
                                                "contract": project.shard_contract(),
                                                "mock_teacher": self.mock,
                                                "storage": "bf16_bits", "shard": si, "time": time.time()})]))
            with open(tmp, "rb+") as f:
                os.fsync(f.fileno())
            replace_atomic(tmp, path)
            dt = time.time() - t0
            rate_hist.append(S / dt)
            rate = sum(rate_hist[-5:]) / len(rate_hist[-5:])
            left = len(todo) - k - 1
            eta = left * size / rate if rate > 0 else None
            log(f"precompute[{split}] shard {si:05d}: {S} prompts in {dt:.0f}s (teacher {t_teacher:.0f}s, "
                f"{S / dt:.2f} p/s), {left} shards left (~{(eta or 0) / 3600:.1f} h)")
            report({"done_shards": n_shards - left, "total_shards": n_shards, "shard_progress": 0.0,
                    "prompts_per_s": rate, "eta_s": eta, "prompts": len(prompts)})
        log(f"precompute[{split}] done")
        return True

    # ------------------------------------------------------------ token-aligned + vision
    def _run_split_token_aligned(self, split: str, report, should_stop) -> bool:
        from .qwen3vl_teacher import batched_hidden
        from .vision_data import build_sample, preprocess_ref_image

        project, pack, log = self.project, self.pack, self.log
        discard_stale_shards(project, split, log)
        samples, size, out_dir, n_shards, todo = shard_plan(project, split)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_shard_contract(project, split)
        log(f"precompute[{split}]: {len(samples)} samples, {n_shards} shards, {len(todo)} to do -> {out_dir}")
        report({"done_shards": n_shards - len(todo), "total_shards": n_shards, "prompts": len(samples)})
        if not todo:
            return True
        T_DIM, S_DIM, V_DIM = pack.out_dim, pack.in_dim, pack.vis_dim
        bud = self.budgets
        dev = self.dev
        rate_hist: List[float] = []

        def batch_bounds(lens, cap_batch, cap_tokens):
            bounds, b0 = [], 0
            S = len(lens)
            while b0 < S:
                b = b0 + 1
                while b < S and b - b0 < cap_batch:
                    if (b - b0 + 1) * int(lens[b]) > cap_tokens:
                        break
                    b += 1
                bounds.append((b0, b))
                b0 = b
            return bounds

        for k, si in enumerate(todo):
            if should_stop():
                log(f"precompute[{split}]: stop requested before shard {si:05d}")
                return False
            t0 = time.time()
            chunk = samples[si * size: (si + 1) * size]

            # ---- pass 0: preprocess images + tokenize (needs the vision token counts) ----
            prepped = []
            for s in chunk:
                text = s.get("text", "") if isinstance(s, dict) else str(s)
                rels = (s.get("images") or []) if isinstance(s, dict) else []
                imgs, counts = [], []
                for rel in rels:
                    p = pathlib.Path(rel)
                    if not p.is_absolute():
                        p = self.data_root / rel
                    im, n = preprocess_ref_image(p)
                    imgs.append(im)
                    counts.append(n)
                ids, vis = build_sample(self.tok, text, counts)
                if len(ids) > pack.max_len_student:
                    raise RuntimeError(f"sample of {len(ids)} tokens exceeds max_len {pack.max_len_student} "
                                       f"(shorten max_chars or use smaller images): {text[:60]!r}")
                prepped.append({"ids": ids, "vis": vis, "imgs": imgs, "n_imgs": len(imgs), "text": text})

            # ---- pass 1: vision tower ----
            tt = time.time()
            queue, owners = [], []
            for j, p in enumerate(prepped):
                for kk, im in enumerate(p["imgs"]):
                    queue.append(im)
                    owners.append((j, kk))
                p["vis_embeds"] = [None] * len(p["imgs"])
            b0 = 0
            while b0 < len(queue):
                if should_stop():
                    log(f"precompute[{split}]: stop requested inside shard {si:05d} (shard discarded)")
                    return False
                b1, tokens = b0, 0
                while b1 < len(queue):
                    n = (queue[b1].shape[0] // 32) * (queue[b1].shape[1] // 32)
                    if b1 > b0 and tokens + n > bud["vis_tok_budget"]:
                        break
                    tokens += n
                    b1 += 1
                outs = self.teacher.encode_images(queue[b0:b1])
                for o, (j, kk) in zip(outs, owners[b0:b1]):
                    prepped[j]["vis_embeds"][kk] = o
                b0 = b1
            del queue
            for p in prepped:
                p["imgs"] = []
            t_vis = time.time() - tt

            # ---- length-sorted storage ----
            lens_all = np.array([len(p["ids"]) for p in prepped])
            order = np.argsort(lens_all, kind="stable")
            S = len(prepped)
            lens = lens_all[order]
            off = np.zeros(S + 1, dtype=np.int64)
            np.cumsum(lens, out=off[1:])
            T = int(off[-1])
            t_pack = np.empty((T, T_DIM), dtype=np.int16)
            q_pack = np.empty((T, S_DIM), dtype=np.int16)
            ids_pack = np.empty(T, dtype=np.int32)
            vis_sample, vis_start, vis_len, v_rows = [], [], [], []
            for gi, r in enumerate(order):
                p = prepped[r]
                ids_pack[off[gi]: off[gi + 1]] = np.asarray(p["ids"], np.int32)
                for (st, n), v in zip(p["vis"], p["vis_embeds"]):
                    vis_sample.append(gi)
                    vis_start.append(st)
                    vis_len.append(n)
                    v_rows.append(bf16_bits(v))
            v_pack = np.concatenate(v_rows) if v_rows else np.empty((0, V_DIM), dtype=np.int16)
            s_stat = np.zeros(T_DIM, dtype=np.float64)
            s2_stat = np.zeros(T_DIM, dtype=np.float64)

            def hf_batch(rows):
                return [{"ids": prepped[order[g]]["ids"], "vis": prepped[order[g]]["vis"],
                         "vis_embeds": prepped[order[g]]["vis_embeds"]} for g in rows]

            # ---- pass 2: teacher text stack ----
            tt = time.time()
            for b0, b1 in batch_bounds(lens, bud["teacher_batch"], bud["tok_budget"]):
                if should_stop():
                    log(f"precompute[{split}]: stop requested inside shard {si:05d} (shard discarded)")
                    return False
                rows = range(b0, b1)
                hidden = self.teacher.hidden(hf_batch(rows))                    # [b, L, T_DIM] bf16 cpu
                for j, gi in enumerate(rows):
                    li = int(lens[gi])
                    t_pack[off[gi]: off[gi + 1]] = bf16_bits(hidden[j, :li])
                    v = hidden[j, :li].double().numpy()
                    s_stat += v.sum(axis=0)
                    s2_stat += (v * v).sum(axis=0)
                report({"done_shards": n_shards - len(todo) + k, "total_shards": n_shards,
                        "shard_progress": b1 / S, "prompts": len(samples)})
            t_teacher = time.time() - tt

            # ---- pass 3: student (vision through the frozen vision_proj) ----
            with torch.no_grad():
                for b0, b1 in batch_bounds(lens, bud["student_batch"], bud["student_tok_budget"]):
                    rows = range(b0, b1)
                    hidden = batched_hidden(self.student, self.s_embed, hf_batch(rows), dev, self.sdtype,
                                            proj=self.w_vis).to(torch.bfloat16).cpu()
                    for j, gi in enumerate(rows):
                        li = int(lens[gi])
                        q_pack[off[gi]: off[gi + 1]] = bf16_bits(hidden[j, :li])

            path = out_dir / f"{split}-{si:05d}.npz"
            tmp = out_dir / f"{split}-{si:05d}.tmp.npz"
            np.savez(tmp, prompts=np.array([prepped[r]["text"] for r in order]),
                     n_images=np.array([prepped[r]["n_imgs"] for r in order], dtype=np.int32),
                     t_pack=t_pack, q_pack=q_pack, v_pack=v_pack,
                     vis_sample=np.array(vis_sample, dtype=np.int32), vis_start=np.array(vis_start, dtype=np.int32),
                     vis_len=np.array(vis_len, dtype=np.int32),
                     len=lens.astype(np.int32), ids=ids_pack,
                     stat_s=s_stat, stat_s2=s2_stat, stat_n=np.array([T]), num_queries=np.array([0]),
                     meta=np.array([json.dumps({"pack": pack.id, "student": self.student_name,
                                                "contract": project.shard_contract(), "mock_teacher": self.mock,
                                                "teacher_mode": getattr(self.teacher, "mode", "?"),
                                                "tap": "final_norm", "storage": "bf16_bits",
                                                "shard": si, "time": time.time()})]))
            with open(tmp, "rb+") as f:
                os.fsync(f.fileno())
            replace_atomic(tmp, path)
            dt = time.time() - t0
            rate_hist.append(S / dt)
            rate = sum(rate_hist[-5:]) / len(rate_hist[-5:])
            left = len(todo) - k - 1
            eta = left * size / rate if rate > 0 else None
            log(f"precompute[{split}] shard {si:05d}: {S} samples in {dt:.0f}s (vision {t_vis:.0f}s, teacher "
                f"{t_teacher:.0f}s, {S / dt:.2f} s/s), {left} shards left (~{(eta or 0) / 3600:.1f} h)")
            report({"done_shards": n_shards - left, "total_shards": n_shards, "shard_progress": 0.0,
                    "prompts_per_s": rate, "eta_s": eta, "prompts": len(samples)})
        log(f"precompute[{split}] done")
        return True
