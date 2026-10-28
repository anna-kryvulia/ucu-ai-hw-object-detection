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

from matplotlib.colors import LinearSegmentedColormap

# Categorical palette in a fixed, colour-blind-checked order: one colour per class, always the same.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLORS = PALETTE[:len(CLASS_NAMES)]
SPLIT_COLORS = {"train": PALETTE[0], "val": PALETTE[1], "test": PALETTE[2]}
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"          # primary / secondary / muted text
# one-hue sequential ramp (light -> dark blue) for magnitudes such as heatmaps
BLUES = LinearSegmentedColormap.from_list("hw_blues", ["#f2f7fd", "#b7d3f6", "#6da7ec", "#2a78d6",
                                                       "#1c5cab", "#0d366b"])


def set_style() -> None:
    """Shared matplotlib look for every notebook: light grid, no top/right spines, readable fonts."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "figure.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 10, "axes.labelcolor": INK_2, "axes.edgecolor": INK_3,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#e4e3df", "grid.linewidth": 0.7, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
        "lines.linewidth": 2, "patch.edgecolor": "white",
    })


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
