"""A training project = one directory holding everything a run needs, so a
reboot loses nothing and `gguf-trainer run --project DIR` (or the GUI's
Resume button) picks up exactly where the pipeline stopped.

    project.json      configuration (pack, materials, hyper-parameters)
    state.json        live pipeline state, written atomically by the pipeline
    pipeline.log      stdout/stderr of the detached pipeline process
    pipeline.pid      pid of the running pipeline (stale after a reboot)
    STOP              request file: the pipeline saves and exits when it appears
    SNAPSHOT          request file: the train stage validates + saves last.pt
                      at the current step and removes it (snapshot export)
    snapshots.json    manifest of the step-tagged GGUFs exported mid-run
    snapshot.log / .snapshot.pid / .snapshot.status
                      the detached snapshot-export job (see snapshot.py)
    materials/        downloaded teacher / tokenizer / vision encoder
    data/             train.txt / val.txt corpus
    shards/{val,train}/*.npz   precomputed teacher targets + student states
    checkpoints/      last.pt / best.pt / log.csv
    eval.json         evaluation of the exported GGUF (kept here, not in the
                      output folder, so projects sharing one output folder
                      never overwrite each other's)
    vision_proj.pt    vision packs: the frozen mmproj -> student map
    ../               exported .gguf files land NEXT TO the project folder by
                      default (export.output_dir overrides), e.g.
                      test-trainer/pig_llada_adapter-f16.gguf next to
                      test-trainer/llada_adapter/; mid-run snapshots are
                      step-tagged: test-trainer/pig_llada_adapter-step2500-f16.gguf
"""

from __future__ import annotations

import copy
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional

from .util import now, pid_alive, read_json, write_json_atomic

STAGES = ["corpus", "precompute_val", "precompute_train", "train", "export", "eval"]
STAGE_TITLES = {
    "corpus": "Corpus",
    "precompute_val": "Precompute (val)",
    "precompute_train": "Precompute (train)",
    "train": "Train adapter",
    "export": "Export GGUF",
    "eval": "Evaluate",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "pack": "llada_image",
    "name": "pig_llada_adapter",
    "auto_resume": True,
    "hf_token": "",
    "materials": {},            # material id -> {"path": ...} overrides (local files)
    "corpus": {
        "mode": "presets",      # presets | files
        "presets": ["gustavosta", "midjourney"],
        "extra_files": [],
        "train_file": "",
        "val_file": "",
        "n_train": 60000,
        "n_val": 1024,
        "empty_frac": 0.01,
        "min_chars": 8,
        "max_chars": 1200,
        "seed": 8,
        # image packs: (image, instruction) samples + the text-only share
        "image_presets": [],    # ids from vision_data.IMAGE_PRESETS (downloaded as materials)
        "image_folders": [],    # local folders of images (+ optional <name>.txt captions)
        "n_image": 56000,
        "n_two": 3000,
        "n_text": 27000,
        "n_val_image": 1024,
        "n_val_text": 512,
    },
    "precompute": {
        "device": "auto",
        "placement": "sequential",  # sequential: fill the chosen GPU first, then the others, then CPU | balanced: accelerate's even split
        "shard_size": 1024,
        "val_shard_size": 512,
        "teacher_batch": 24,
        "tok_budget": 12288,
        "student_batch": 64,
        "student_tok_budget": 0,    # vision packs: 0 = auto for the card
        "vis_tok_budget": 0,        # vision packs: merged vision tokens per tower call, 0 = auto
        "teacher_mode": "auto",     # vision packs: auto | gpu (resident) | offload (streamed through the GPU)
        "gpu_mem_gib": 0,       # 0 = auto (total - 1.5 GiB on the chosen device)
        "cpu_mem_gib": 0,       # 0 = auto (available RAM - 6 GiB)
        "student_dtype": "bf16",
    },
    "train": {
        "device": "auto",
        "width": 1024,
        "depth": 6,
        "steps": 20000,
        "batch_size": 32,
        "lr": 2e-4,
        "warmup": 1000,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "cos_weight": 0.5,
        "sigma_floor": 0.03,
        "grad_checkpoint": "auto",
        "log_every": 25,
        "val_every": 500,
        "save_every": 250,
        "seed": 42,
    },
    "export": {
        "output_dir": "",       # default: the folder containing the project folder
        "copy_to": "",          # optional extra destination directory
        "export_sigvq": True,
        "run_eval": True,
    },
}


def validate_steps(v: Any) -> int:
    """train.steps as a positive int (accepts "12000", 12000.0, "12k")."""
    if isinstance(v, str):
        t = v.strip().lower().replace(",", "").replace("_", "")
        if t.endswith("k"):
            t = str(float(t[:-1]) * 1000)
        v = t
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"train.steps must be a positive whole number, got {v!r}")
    if f != f or f < 1 or f != int(f):
        raise ValueError(f"train.steps must be a positive whole number, got {v!r}")
    return int(f)


def _merge(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def settings_path() -> pathlib.Path:
    return pathlib.Path.home() / ".gguf-trainer" / "settings.json"


def load_settings() -> Dict[str, Any]:
    return read_json(settings_path(), {}) or {}


def save_settings(s: Dict[str, Any]) -> None:
    write_json_atomic(settings_path(), s)


def default_projects_root() -> pathlib.Path:
    return pathlib.Path.home() / "gguf-trainer" / "projects"


class Project:
    def __init__(self, path: os.PathLike | str):
        self.path = pathlib.Path(path).expanduser().resolve()
        self.config: Dict[str, Any] = _merge(DEFAULT_CONFIG, read_json(self.config_file, {}) or {})

    # -- files --
    @property
    def config_file(self) -> pathlib.Path:
        return self.path / "project.json"

    @property
    def state_file(self) -> pathlib.Path:
        return self.path / "state.json"

    @property
    def log_file(self) -> pathlib.Path:
        return self.path / "pipeline.log"

    @property
    def pid_file(self) -> pathlib.Path:
        return self.path / "pipeline.pid"

    @property
    def stop_file(self) -> pathlib.Path:
        return self.path / "STOP"

    @property
    def snapshot_file(self) -> pathlib.Path:
        return self.path / "SNAPSHOT"

    @property
    def snapshots_file(self) -> pathlib.Path:
        return self.path / "snapshots.json"

    @property
    def snapshot_log_file(self) -> pathlib.Path:
        return self.path / "snapshot.log"

    @property
    def snapshot_pid_file(self) -> pathlib.Path:
        return self.path / ".snapshot.pid"

    @property
    def snapshot_status_file(self) -> pathlib.Path:
        return self.path / ".snapshot.status"

    @property
    def materials_dir(self) -> pathlib.Path:
        return self.path / "materials"

    @property
    def data_dir(self) -> pathlib.Path:
        return self.path / "data"

    @property
    def shards_dir(self) -> pathlib.Path:
        return self.path / "shards"

    @property
    def checkpoints_dir(self) -> pathlib.Path:
        return self.path / "checkpoints"

    @property
    def output_dir(self) -> pathlib.Path:
        d = (self.config.get("export") or {}).get("output_dir") or ""
        return pathlib.Path(d).expanduser() if d else self.path.parent

    def exists(self) -> bool:
        return self.config_file.is_file()

    def pack(self):
        from .packs import get_pack

        return get_pack(self.config["pack"])

    def eval_path(self) -> pathlib.Path:
        return self.path / "eval.json"

    def vision_proj_path(self) -> pathlib.Path:
        return self.path / "vision_proj.pt"

    def create(self, name: str, pack: str) -> None:
        from .packs import get_pack

        self.path.mkdir(parents=True, exist_ok=True)
        self.config = _merge(_merge(DEFAULT_CONFIG, get_pack(pack).defaults()), {"name": name, "pack": pack})
        self.save_config()
        self.write_state({"status": "idle", "stage": None, "stages": {}, "updated": now()})

    def save_config(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        if cfg is not None:
            self.config = _merge(DEFAULT_CONFIG, cfg)
        self.config["train"]["steps"] = validate_steps(self.config["train"].get("steps"))
        self.path.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.config_file, self.config)

    def set_train_steps(self, steps: Any) -> int:
        """Change the planned step count (project.json) — before, between or
        during runs: a training loop in progress re-reads it (see
        train.py) and finishes at the new count."""
        self.config["train"]["steps"] = validate_steps(steps)
        self.save_config()
        return self.config["train"]["steps"]

    def live_train_steps(self) -> Optional[int]:
        """The planned steps as project.json says RIGHT NOW (None when the
        file is unreadable or mid-write), for the running trainer to pick up
        a change made from the GUI / CLI without a restart."""
        cfg = read_json(self.config_file, None)
        if not isinstance(cfg, dict):
            return None
        try:
            return validate_steps((cfg.get("train") or {}).get("steps"))
        except ValueError:
            return None

    # -- state --
    def read_state(self) -> Dict[str, Any]:
        st = read_json(self.state_file, None)
        if not isinstance(st, dict):
            st = {"status": "idle", "stage": None, "stages": {}, "updated": None}
        st.setdefault("stages", {})
        return st

    def write_state(self, st: Dict[str, Any]) -> None:
        st["updated"] = now()
        write_json_atomic(self.state_file, st)

    def pid(self) -> Optional[int]:
        try:
            return int(self.pid_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def is_running(self) -> bool:
        return pid_alive(self.pid(), marker="gguf_trainer.pipeline")

    def runtime_status(self) -> str:
        """idle | running | interrupted | stopped | failed | done.
        'interrupted' = state says running but no live process (reboot/kill)."""
        st = self.read_state()
        status = st.get("status") or "idle"
        if self.is_running():
            return "running"
        if status == "running":
            return "interrupted"
        return status

    def request_stop(self) -> None:
        self.stop_file.write_text("stop\n")

    def clear_stop(self) -> None:
        try:
            self.stop_file.unlink()
        except OSError:
            pass

    def stop_requested(self) -> bool:
        return self.stop_file.exists()

    # -- snapshot request (train stage: validate + save last.pt NOW, keep going) --
    def request_snapshot(self) -> None:
        self.snapshot_file.write_text("snapshot\n")

    def clear_snapshot(self) -> None:
        try:
            self.snapshot_file.unlink()
        except OSError:
            pass

    def snapshot_requested(self) -> bool:
        return self.snapshot_file.exists()

    def snapshot_path(self, step: int) -> pathlib.Path:
        """Step-tagged export of a checkpoint taken before training finished:
        pig_llada_adapter-step2500-f16.gguf (listed by output_files(): it
        carries the project's export prefix)."""
        return self.output_dir / f"{self.config['name']}-step{int(step)}-f16.gguf"

    def snapshots(self) -> List[Dict[str, Any]]:
        """Manifest of the snapshot exports (newest last), each with the step,
        checkpoint kind, validation cosine at that step and the optional eval."""
        m = read_json(self.snapshots_file, None)
        return list(m.get("snapshots", [])) if isinstance(m, dict) else []

    def checkpoint_info(self, name: str) -> Dict[str, Any]:
        """checkpoints/<name>.json sidecar (step, val_cos, best_val_cos, ...)."""
        ck = self.checkpoints_dir / name
        if not ck.is_file():
            return {}
        info = read_json(self.checkpoints_dir / name.replace(".pt", ".json"), {}) or {}
        info = dict(info)
        info["path"] = str(ck)
        info["mtime"] = ck.stat().st_mtime
        return info

    # -- stage artifacts (what "done" means for each stage) --
    def corpus_files(self) -> Dict[str, pathlib.Path]:
        c = self.config["corpus"]
        if self.pack().needs_images:
            return {"train": self.data_dir / "train.jsonl", "val": self.data_dir / "val.jsonl"}
        if c.get("mode") == "files":
            return {"train": pathlib.Path(c.get("train_file") or ""), "val": pathlib.Path(c.get("val_file") or "")}
        return {"train": self.data_dir / "train.txt", "val": self.data_dir / "val.txt"}

    _line_cache: Dict[str, tuple] = {}

    def n_shards(self, split: str) -> Optional[int]:
        f = self.corpus_files()[split]
        if not f.is_file():
            return None
        st = f.stat()
        key = str(f)
        cached = Project._line_cache.get(key)
        if cached and cached[0] == (st.st_mtime, st.st_size):
            n = cached[1]
        else:
            n = sum(1 for _ in open(f, encoding="utf-8"))
            Project._line_cache[key] = ((st.st_mtime, st.st_size), n)
        size = self.config["precompute"]["val_shard_size" if split == "val" else "shard_size"]
        return max(1, (n + size - 1) // size)

    SHARD_CONTRACT_FILE = "CONTRACT"

    def shard_contract(self) -> str:
        return self.pack().shard_contract

    def shard_contract_on_disk(self, split: str) -> Optional[str]:
        f = self.shards_dir / split / self.SHARD_CONTRACT_FILE
        try:
            return f.read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    def shard_files(self, split: str) -> List[str]:
        d = self.shards_dir / split
        if not d.is_dir():
            return []
        return sorted(f for f in os.listdir(d) if f.endswith(".npz") and not f.endswith(".tmp.npz"))

    def stale_shards(self, split: str) -> List[str]:
        """Shards written under another teacher contract (or none)."""
        fs = self.shard_files(split)
        if fs and self.shard_contract_on_disk(split) != self.shard_contract():
            return fs
        return []

    def done_shards(self, split: str) -> int:
        fs = self.shard_files(split)
        if fs and self.shard_contract_on_disk(split) != self.shard_contract():
            return 0          # stale: precompute will recompute them
        return len(fs)

    def adapter_path(self) -> pathlib.Path:
        return self.output_dir / f"{self.config['name']}-f16.gguf"

    def stage_artifacts(self) -> Dict[str, Dict[str, Any]]:
        """Per-stage completion derived from disk, independent of state.json,
        so a stale state after a reboot never hides real progress."""
        out: Dict[str, Dict[str, Any]] = {}
        cf = self.corpus_files()
        out["corpus"] = {"done": cf["train"].is_file() and cf["val"].is_file(),
                         "train_file": str(cf["train"]), "val_file": str(cf["val"])}
        for split, key in (("val", "precompute_val"), ("train", "precompute_train")):
            total = self.n_shards(split)
            done = self.done_shards(split)
            out[key] = {"done": total is not None and done >= total, "done_shards": done, "total_shards": total}
        last = self.checkpoints_dir / "last.pt"
        best = self.checkpoints_dir / "best.pt"
        last_info = self.checkpoint_info("last.pt")
        best_info = self.checkpoint_info("best.pt")
        ck_step = last_info.get("step")
        out["train"] = {"done": ck_step is not None and ck_step >= int(self.config["train"]["steps"]),
                        "has_last": last.is_file(), "has_best": best.is_file(), "step": ck_step,
                        "steps": int(self.config["train"]["steps"]),
                        "last_val_cos": last_info.get("val_cos"), "best_step": best_info.get("step"),
                        "best_val_cos": best_info.get("best_val_cos")}
        ap = self.adapter_path()
        exported = ap.is_file() and best.is_file() and ap.stat().st_mtime >= best.stat().st_mtime
        out["export"] = {"done": exported, "path": str(ap)}
        ev = self.eval_path()
        out["eval"] = {"done": ev.is_file() and exported and ev.stat().st_mtime >= ap.stat().st_mtime,
                       "path": str(ev)}
        return out

    def export_base(self) -> str:
        """Common prefix of this project's exports: pig_llada_adapter -> pig_llada
        (pig_llada_adapter-f16.gguf, pig_llada_sigvq-f16.gguf, ...)."""
        name = self.config["name"]
        return name[: -len("_adapter")] if name.endswith("_adapter") else name

    def output_files(self) -> List[Dict[str, Any]]:
        """This project's exports in output_dir (the folder is shared with the
        project folder itself and possibly other projects, so only files
        carrying this project's export prefix are listed)."""
        files = []
        base = self.export_base()
        if self.output_dir.is_dir():
            for p in sorted(self.output_dir.iterdir()):
                if p.is_file() and p.suffix == ".gguf" and p.name.startswith(base + "_"):
                    files.append({"name": p.name, "path": str(p), "size": p.stat().st_size,
                                  "mtime": p.stat().st_mtime})
        return files

    def snapshot_job_status(self) -> Dict[str, Any]:
        from .snapshot import job_status

        return job_status(self)

    def summary(self) -> Dict[str, Any]:
        st = self.read_state()
        return {
            "path": str(self.path),
            "exists": self.exists(),
            "config": self.config,
            "state": st,
            "runtime_status": self.runtime_status(),
            "pid": self.pid(),
            "artifacts": self.stage_artifacts(),
            "output_dir": str(self.output_dir),
            "output_files": self.output_files(),
            "eval": read_json(self.eval_path(), None),
            "snapshots": self.snapshots(),
            "snapshot_job": self.snapshot_job_status(),
            "stages": STAGES,
            "stage_titles": STAGE_TITLES,
            "python": sys.executable,
        }


def stage_key_for_split(split: str) -> str:
    return "precompute_val" if split == "val" else "precompute_train"
