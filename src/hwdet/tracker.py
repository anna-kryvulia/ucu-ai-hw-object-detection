"""Experiment tracking that survives crashed sessions.

Every experiment is a *run* with a fixed id (the config name), stored in <outputs>/runs/<run_id>/:

    config.yaml        exact config used
    stages.json        which stages finished, e.g. {"train": "...time...", "eval_val": "..."}
    history.csv        one row per epoch (losses, val mAP, lr, ...)
    metrics.json       final metrics per evaluation, e.g. {"val": {"mAP": .., "AP50": ..}}
    checkpoints/       last.pt / best.pt  (not copied into git)
    predictions/       val.csv / test.csv in the common prediction format (not copied into git)

Because the id is fixed, re-running a notebook after a crash opens the *same* folder: finished stages
are skipped, training resumes from checkpoints/last.pt, and metrics are already on disk.

    run = Run("frcnn_baseline", cfg, paths)
    if not run.is_done("train"):
        train(...)                      # resumes from run.ckpt("last.pt") if it exists
        run.mark_done("train")
    if not run.is_done("eval_val"):
        run.log_metrics("val", evaluate(...))
        run.mark_done("eval_val")

Every log_metrics call also updates <outputs>/experiments.csv: one table with all runs, used
by the comparison notebook and the report. `status(paths)` prints what is done and what is not.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from .config import save_config
from .env import Paths

REGISTRY = "experiments.csv"
SMALL_FILES = ("config.yaml", "stages.json", "history.csv", "metrics.json", "notes.md")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Run:
    def __init__(self, run_id: str, cfg: dict | None, paths: Paths, use_wandb: bool | None = None):
        self.id = run_id
        self.paths = paths
        self.dir = paths.runs / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "checkpoints").mkdir(exist_ok=True)
        (self.dir / "predictions").mkdir(exist_ok=True)
        self.cfg = cfg or {}
        if cfg is not None:
            save_config(cfg, self.dir / "config.yaml")
        self._wandb = None
        if use_wandb if use_wandb is not None else self.cfg.get("wandb", False):
            self._init_wandb()
        print(f"run '{run_id}' at {self.dir}  done: {list(self.stages) or 'nothing yet'}")

    # ----- stages -----
    @property
    def stages(self) -> dict:
        f = self.dir / "stages.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def is_done(self, stage: str) -> bool:
        return stage in self.stages

    def mark_done(self, stage: str) -> None:
        s = self.stages
        s[stage] = _now()
        (self.dir / "stages.json").write_text(json.dumps(s, indent=2))

    def reset(self, stage: str) -> None:
        """Force a stage to run again (e.g. after fixing a bug)."""
        s = self.stages
        s.pop(stage, None)
        (self.dir / "stages.json").write_text(json.dumps(s, indent=2))

    # ----- files -----
    def ckpt(self, name: str = "last.pt") -> Path:
        return self.dir / "checkpoints" / name

    def pred_path(self, split: str) -> Path:
        return self.dir / "predictions" / f"{split}.csv"

    # ----- logging -----
    def log_epoch(self, epoch: int, values: dict[str, Any]) -> None:
        f = self.dir / "history.csv"
        row = pd.DataFrame([{"epoch": epoch, "time": _now(), **values}])
        if f.exists():
            hist = pd.read_csv(f)
            hist = hist[hist["epoch"] != epoch]          # a resumed epoch overwrites itself
            row = pd.concat([hist, row], ignore_index=True)
        row.sort_values("epoch").to_csv(f, index=False)
        if self._wandb:
            self._wandb.log({**values, "epoch": epoch})

    @property
    def history(self) -> pd.DataFrame:
        f = self.dir / "history.csv"
        return pd.read_csv(f) if f.exists() else pd.DataFrame()

    @property
    def metrics(self) -> dict:
        f = self.dir / "metrics.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def log_metrics(self, eval_name: str, values: dict[str, Any]) -> None:
        """Store final metrics of one evaluation (e.g. 'val', 'val_sahi', 'public_lb')."""
        m = self.metrics
        m[eval_name] = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in values.items()}
        (self.dir / "metrics.json").write_text(json.dumps(m, indent=2))
        _upsert_registry(self.paths.outputs / REGISTRY, self.id, eval_name, self.cfg, m[eval_name])
        if self._wandb:
            self._wandb.summary.update({f"{eval_name}/{k}": v for k, v in m[eval_name].items()
                                        if isinstance(v, (int, float))})

    def note(self, text: str) -> None:
        """Free-text observations, kept next to the metrics (handy for the report)."""
        with open(self.dir / "notes.md", "a") as f:
            f.write(f"- {_now()}: {text}\n")

    # ----- wandb (optional, needed only for the visualization bonus) -----
    def _init_wandb(self) -> None:
        try:
            import wandb
            self._wandb = wandb.init(project=self.cfg.get("wandb_project", "ucu-hw2-detection"),
                                     name=self.id, id=self.id, resume="allow", config=self.cfg,
                                     dir=str(self.dir))
        except Exception as e:                       # no login / no network: keep training, log locally only
            print(f"W&B disabled for this run: {e}")
            self._wandb = None

    def finish(self) -> None:
        if self._wandb:
            self._wandb.finish()


def _upsert_registry(path: Path, run_id: str, eval_name: str, cfg: dict, values: dict) -> None:
    row = {
        "run_id": run_id,
        "eval": eval_name,
        "family": cfg.get("family", ""),
        "model": cfg.get("model", {}).get("arch", ""),
        "description": cfg.get("description", ""),
        "updated": _now(),
        **{k: v for k, v in values.items() if isinstance(v, (int, float, str))},
    }
    new = pd.DataFrame([row])
    if path.exists():
        df = pd.read_csv(path)
        df = df[~((df["run_id"] == run_id) & (df["eval"] == eval_name))]
        if len(df):                                  # concat with an empty frame raises a FutureWarning
            new = pd.concat([df, new], ignore_index=True)
    new.to_csv(path, index=False)


def load_registry(paths: Paths) -> pd.DataFrame:
    f = paths.outputs / REGISTRY
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def status(paths: Paths) -> pd.DataFrame:
    """Table of every run: finished stages, last epoch, has checkpoint, val mAP.
    Look at this after a crash to see what still needs to run."""
    rows = []
    for d in sorted(p for p in paths.runs.iterdir() if p.is_dir()):
        stages = json.loads((d / "stages.json").read_text()) if (d / "stages.json").exists() else {}
        hist = pd.read_csv(d / "history.csv") if (d / "history.csv").exists() else pd.DataFrame()
        metrics = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
        rows.append({
            "run_id": d.name,
            "stages_done": ", ".join(stages),
            "last_epoch": int(hist["epoch"].max()) if len(hist) else None,
            "has_last_ckpt": (d / "checkpoints" / "last.pt").exists(),
            "val_mAP": metrics.get("val", {}).get("mAP"),
        })
    return pd.DataFrame(rows)
