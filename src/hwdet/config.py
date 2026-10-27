"""Experiment configs: one YAML per experiment in configs/experiments/.

A config may start with `base: <other config name>`; it inherits everything from that file and
overrides only what it lists. This keeps "baseline vs tuned" diffs small and readable.

    cfg = load_config("frcnn_baseline", train={"epochs": 2})   # overrides for quick debugging
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .env import REPO_ROOT

CONFIG_DIR = REPO_ROOT / "configs" / "experiments"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(name_or_path: str | Path) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("base", None)
    return deep_merge(_read(parent), cfg) if parent else cfg


def load_config(name_or_path: str | Path, **overrides: Any) -> dict:
    cfg = deep_merge(_read(name_or_path), overrides)
    cfg.setdefault("name", Path(name_or_path).stem)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
