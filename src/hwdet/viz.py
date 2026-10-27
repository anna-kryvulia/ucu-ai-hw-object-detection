"""Plots: boxes on frames (GT vs predictions), training curves, comparison bars.

save_fig() writes into <outputs>/figures/, which lives on Drive / Kaggle output, so figures survive
the session. scripts/collect_results.py later copies them into images/ for the report.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from .data.classes import CLASS_NAMES
from .data.io import load_gt
from .env import Paths

COLORS = plt.cm.tab10(np.arange(len(CLASS_NAMES)))


def draw(ax, img, boxes_xywh, classes, scores=None, ignore_xywh=None, title: str = "", lw: float = 1.0):
    ax.imshow(img)
    for i, ((x, y, w, h), c) in enumerate(zip(boxes_xywh, classes)):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=COLORS[int(c)], lw=lw))
        if scores is not None:
            ax.text(x, y - 2, f"{CLASS_NAMES[int(c)]} {scores[i]:.2f}", color=COLORS[int(c)], fontsize=5)
    for x, y, w, h in ([] if ignore_xywh is None else ignore_xywh):
        ax.add_patch(Rectangle((x, y), w, h, fill=True, fc=(1, 1, 1, 0.25), ec="white", ls="--", lw=lw))
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def show_frame(data_root: Path, frame_row, preds: pd.DataFrame | None = None, score_thr: float = 0.3,
               figsize=(16, 6)):
    """GT (left) next to predictions (right) for one frame row from frame_table()."""
    img = Image.open(Path(data_root) / frame_row.file_name)
    cls, boxes, ign = load_gt(Path(data_root), frame_row.file_name, int(frame_row.width), int(frame_row.height))
    n = 1 if preds is None else 2
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    draw(axes[0, 0], img, boxes, cls, ignore_xywh=ign, title=f"GT {frame_row.file_name} ({len(cls)} boxes)")
    if preds is not None:
        p = preds[(preds["file_name"] == frame_row.file_name) & (preds["score"] >= score_thr)]
        draw(axes[0, 1], img, p[["x", "y", "w", "h"]].to_numpy(), p["cls"].to_numpy(), p["score"].to_numpy(),
             title=f"pred >= {score_thr} ({len(p)} boxes)")
    fig.tight_layout()
    return fig


def plot_history(histories: dict[str, pd.DataFrame], metric: str = "val_mAP", ax=None):
    """Same-budget comparison curves, e.g. CNN vs DETR mAP over epochs."""
    ax = ax or plt.gca()
    for name, h in histories.items():
        if metric in h:
            d = h.dropna(subset=[metric])
            ax.plot(d["epoch"], d[metric], marker="o", ms=3, label=name)
    ax.set_xlabel("epoch"); ax.set_ylabel(metric); ax.legend(); ax.grid(alpha=0.3)
    return ax


def save_fig(fig, name: str, paths: Paths, dpi: int = 150) -> Path:
    out = paths.outputs / "figures" / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    if paths.platform == "local":            # straight into the report folder
        fig.savefig(paths.repo / "images" / f"{name}.png", dpi=dpi, bbox_inches="tight")
    return out
