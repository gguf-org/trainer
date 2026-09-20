"""Shard reader for both target layouts (see precompute.py) and the
checkpointable infinite training stream (trainer4/trainer8 design: the
stream position is part of the checkpoint, so a resume replays exactly).

  resampler:      t_pack [S*NQ, out_dim] fixed rows per sample, num_queries > 0
  seeded:         num_queries > 0 AND seed_ids [S, NQ] (the teacher tokenizer's
                  window) + seed_len [S]; t_pack [sum(seed_len), out_dim] holds
                  only the real slots, packed in sample order
  token-aligned:  t_pack [T, out_dim] one row per real token, packed like
                  q_pack; v_pack [Tv, vis_dim] raw vision embeds + a
                  (vis_sample, vis_start, vis_len) segment table; n_images
"""

from __future__ import annotations

import os
import threading

import numpy as np
import torch


def shard_files(dir_):
    fs = sorted(f for f in os.listdir(dir_) if f.endswith(".npz") and not f.endswith(".tmp.npz"))
    if not fs:
        raise FileNotFoundError(f"no shards in {dir_}")
    return [os.path.join(dir_, f) for f in fs]


def load_stats(dir_, sigma_floor):
    s = s2 = None
    n = 0.0
    for f in shard_files(dir_):
        z = np.load(f, allow_pickle=False)
        s = z["stat_s"] if s is None else s + z["stat_s"]
        s2 = z["stat_s2"] if s2 is None else s2 + z["stat_s2"]
        n += float(z["stat_n"][0])
    mu = s / n
    var = np.maximum(s2 / n - mu * mu, 1e-6)
    sigma = np.sqrt(var)
    inv_sigma = 1.0 / np.maximum(sigma, sigma_floor)
    floored = int((sigma < sigma_floor).sum())
    return (torch.from_numpy(mu).float(), torch.from_numpy(sigma).float(),
            torch.from_numpy(inv_sigma).float(), floored)


def from_bits(arr):
    return torch.from_numpy(arr).view(torch.bfloat16)


class Shard:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.prompts = z["prompts"]
        self.t_pack = z["t_pack"]
        self.q_pack = z["q_pack"]
        self.len = z["len"].astype(np.int64)
        self.ids = z["ids"]
        self.num_queries = int(z["num_queries"][0]) if "num_queries" in z else 256
        self.off = np.concatenate([[0], np.cumsum(self.len)])
        self.n = len(self.len)
        self.token_aligned = self.num_queries == 0
        self.seeded = "seed_ids" in z
        if self.seeded:
            self.seed_ids = z["seed_ids"].astype(np.int64)
            self.seed_len = z["seed_len"].astype(np.int64)
            self.seed_off = np.concatenate([[0], np.cumsum(self.seed_len)])
        if self.token_aligned:
            self.n_images = z["n_images"] if "n_images" in z else np.zeros(self.n, dtype=np.int32)
            self.v_pack = z["v_pack"] if "v_pack" in z else np.empty((0, 0), dtype=np.int16)
            self.vis_dim = int(self.v_pack.shape[1]) if self.v_pack.ndim == 2 and self.v_pack.shape[1] else 0
            v_off = np.concatenate([[0], np.cumsum(z["vis_len"].astype(np.int64))]) if "vis_len" in z else [0]
            self.vis = [[] for _ in range(self.n)]
            if "vis_sample" in z:
                for k in range(len(z["vis_sample"])):
                    self.vis[int(z["vis_sample"][k])].append(
                        (int(z["vis_start"][k]), int(z["vis_len"][k]), int(v_off[k])))
            # contracts that train on the consumed text rows only (vision_data.supervised_mask):
            # teacher rows under the image slots are absent from t_pack (t_sparse) and
            # rows before sup_start / under the slots are outside the loss
            self.t_sparse = bool(int(z["t_sparse"][0])) if "t_sparse" in z else False
            self.supervise = str(z["supervise"][0]) if "supervise" in z else "all"
            if "sup_start" in z:
                self.sup_start = z["sup_start"].astype(np.int64)
            else:
                # shards written before contracts existed are MageFlow-Edit's
                from .vision_data import start_idx

                self.sup_start = np.array([start_idx(int(n)) for n in self.n_images], dtype=np.int64)
            n_vis = np.array([sum(n for _, n, _ in v) for v in self.vis], dtype=np.int64)
            self.t_off = np.concatenate([[0], np.cumsum(self.len - n_vis)]) if self.t_sparse else self.off

    def batch(self, rows):
        B = len(rows)
        L = int(self.len[rows].max())
        hidden = np.zeros((B, L, self.q_pack.shape[1]), dtype=np.int16)
        keep = np.zeros((B, L), dtype=bool)
        if self.seeded:
            NQ = self.num_queries
            target = np.zeros((B, NQ, self.t_pack.shape[1]), dtype=np.int16)
            seed_mask = np.zeros((B, NQ), dtype=bool)
            for j, r in enumerate(rows):
                n = int(self.seed_len[r])
                target[j, :n] = self.t_pack[self.seed_off[r]: self.seed_off[r + 1]]
                seed_mask[j, :n] = True
                li = int(self.len[r])
                hidden[j, :li] = self.q_pack[self.off[r]: self.off[r + 1]]
                keep[j, :li] = True
            return {
                "prompts": [str(self.prompts[r]) for r in rows],
                "target": from_bits(target),
                "qwen_hidden": from_bits(hidden),
                "keep": torch.from_numpy(keep),
                "seed_ids": torch.from_numpy(self.seed_ids[rows]),
                "seed_mask": torch.from_numpy(seed_mask),
            }
        if not self.token_aligned:
            NQ = self.num_queries
            target = np.empty((B, NQ, self.t_pack.shape[1]), dtype=np.int16)
            for j, r in enumerate(rows):
                target[j] = self.t_pack[r * NQ: (r + 1) * NQ]
                li = int(self.len[r])
                hidden[j, :li] = self.q_pack[self.off[r]: self.off[r + 1]]
                keep[j, :li] = True
            return {
                "prompts": [str(self.prompts[r]) for r in rows],
                "target": from_bits(target),
                "qwen_hidden": from_bits(hidden),
                "keep": torch.from_numpy(keep),
            }
        target = np.zeros((B, L, self.t_pack.shape[1]), dtype=np.int16)
        vis = np.zeros((B, L, self.vis_dim), dtype=np.int16) if self.vis_dim else None
        is_vis = np.zeros((B, L), dtype=bool)
        sup = np.zeros((B, L), dtype=bool)
        for j, r in enumerate(rows):
            li = int(self.len[r])
            hidden[j, :li] = self.q_pack[self.off[r]: self.off[r + 1]]
            keep[j, :li] = True
            for (st, n, vo) in self.vis[r]:
                if vis is not None:
                    vis[j, st: st + n] = self.v_pack[vo: vo + n]
                is_vis[j, st: st + n] = True
            if self.t_sparse:
                target[j, :li][~is_vis[j, :li]] = self.t_pack[self.t_off[r]: self.t_off[r + 1]]
            else:
                target[j, :li] = self.t_pack[self.off[r]: self.off[r + 1]]
            if self.supervise == "consumed_text":
                sup[j, int(self.sup_start[r]): li] = True
                sup[j] &= ~is_vis[j]
            else:
                sup[j, :li] = True
        return {
            "prompts": [str(self.prompts[r]) for r in rows],
            "n_images": [int(self.n_images[r]) for r in rows],
            "target": from_bits(target),
            "qwen_hidden": from_bits(hidden),
            "vis": from_bits(vis) if vis is not None else None,
            "is_vis": torch.from_numpy(is_vis),
            "keep": torch.from_numpy(keep),
            "sup": torch.from_numpy(sup),
            "sup_start": [int(self.sup_start[r]) for r in rows],
        }


class _Prefetcher(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self.shard = None
        self.err = None
        self.start()

    def run(self):
        try:
            self.shard = Shard(self.path)
        except Exception as e:
            self.err = e

    def get(self):
        self.join()
        if self.err:
            raise self.err
        return self.shard


class ShardStream:
    def __init__(self, dir_, batch_size, seed=42, state=None):
        self.files = shard_files(dir_)
        self.batch_size = batch_size
        self.seed = seed
        st = state or {}
        self.epoch = int(st.get("epoch", 0))
        self.shard_i = int(st.get("shard_i", 0))
        self.batch_i = int(st.get("batch_i", 0))

    def state(self):
        return {"epoch": self.epoch, "shard_i": self.shard_i, "batch_i": self.batch_i}

    def _perm(self, *key):
        return np.random.default_rng(list(key)).permutation

    def __iter__(self):
        while True:
            shard_order = self._perm(self.seed, self.epoch)(len(self.files))
            while self.shard_i < len(shard_order):
                fi = int(shard_order[self.shard_i])
                shard = Shard(self.files[fi])
                nxt = None
                if self.shard_i + 1 < len(shard_order):
                    nxt = _Prefetcher(self.files[int(shard_order[self.shard_i + 1])])
                rows = self._perm(self.seed, self.epoch, fi)(shard.n)
                n_batches = (shard.n + self.batch_size - 1) // self.batch_size
                while self.batch_i < n_batches:
                    b = rows[self.batch_i * self.batch_size: (self.batch_i + 1) * self.batch_size]
                    self.batch_i += 1
                    yield shard.batch(b)
                self.batch_i = 0
                self.shard_i += 1
                if nxt is not None:
                    nxt.get()
            self.shard_i = 0
            self.epoch += 1


class ValSet:
    def __init__(self, dir_, batch_size):
        self.batches = []
        for f in shard_files(dir_):
            sh = Shard(f)
            for b0 in range(0, sh.n, batch_size):
                self.batches.append(sh.batch(np.arange(b0, min(b0 + batch_size, sh.n))))

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)
