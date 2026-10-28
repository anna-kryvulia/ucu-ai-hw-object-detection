"""Weights & Biases extras for the visualization bonus: validation frames with GT and predicted boxes.

Training curves and the config are logged by tracker.Run itself when the config has `wandb: true`.
Login once per machine: set WANDB_API_KEY (Kaggle / Colab secret) or run `wandb login` locally.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data.classes import CLASS_NAMES


def login_from_secrets() -> bool:
    """Reads WANDB_API_KEY from Kaggle / Colab secrets (if present) and logs in."""
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore
            key = UserSecretsClient().get_secret("WANDB_API_KEY")
        except Exception:
            try:
                from google.colab import userdata  # type: ignore
                key = userdata.get("WANDB_API_KEY")
            except Exception:
                key = None
    if not key:
        print("no WANDB_API_KEY found: runs are tracked locally only")
        os.environ["WANDB_MODE"] = "disabled"
        return False
    import wandb
    return wandb.login(key=key, relogin=False)


def log_predictions(run, frames: pd.DataFrame, preds: pd.DataFrame, data_root: Path, n: int = 8,
                    score_thr: float = 0.3, max_side: int = 1280) -> None:
    """Logs n validation frames (spread over clips) with predicted boxes; GT is drawn from the COCO file."""
    import wandb
    from .data.io import annotations_table

    labels = {i + 1: c for i, c in enumerate(CLASS_NAMES)}
    pick = frames.groupby("clip").nth(0).sample(min(n, frames["clip"].nunique()), random_state=0)
    images = []
    for r in pick.itertuples():
        img = Image.open(Path(data_root) / r.file_name).convert("RGB")
        s = min(1.0, max_side / max(img.size))
        img = img.resize((int(img.width * s), int(img.height * s)))
        a = annotations_table(data_root, r.split)
        gt = a[(a.coco_id == r.coco_id) & (a.iscrowd == 0)]
        p = preds[(preds.file_name == r.file_name) & (preds.score >= score_thr)]

        def box(x, y, w, h, c, score=None):
            d = {"position": {"minX": float(x * s), "minY": float(y * s), "maxX": float((x + w) * s),
                              "maxY": float((y + h) * s)}, "domain": "pixel", "class_id": int(c) + 1}
            if score is not None:
                d["scores"] = {"score": float(score)}
            return d
        images.append(wandb.Image(img, caption=r.file_name, boxes={
            "prediction": {"box_data": [box(b.x, b.y, b.w, b.h, b.cls, b.score) for b in p.itertuples()],
                           "class_labels": labels},
            "ground_truth": {"box_data": [box(b.x, b.y, b.w, b.h, b.cls) for b in gt.itertuples()],
                             "class_labels": labels},
        }))
    run._wandb.log({"val_predictions": images})
