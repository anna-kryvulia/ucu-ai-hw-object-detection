"""Own pipeline: tiled inference + flip TTA, box-level ensembling, temporal rescoring, fetching old runs.

Every detector here takes a list of CHW float images (0-1) and returns torchvision-style dicts
{"boxes": xyxy pixels, "scores", "labels": 1..8}, so engine.predict / inference_speed work unchanged:

    tiled = TiledDetector(frcnn, grid=(2, 2), overlap=0.2, flip=True)
    preds = predict(tiled, DetectionDataset(val_frames, P.data, targets), device, batch_size=1)
    ens = merge_predictions([preds_a, preds_b], iou=0.5)
    final = temporal_rescore(ens, val_frames, window=1, lam=0.5)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.ops import batched_nms, box_iou

from .data.io import PRED_COLUMNS
from .env import Paths


# ---------------------------------------------------------------- tiling
def tile_windows(W: int, H: int, grid: tuple[int, int], overlap: float) -> list[tuple[int, int, int, int]]:
    """cols x rows windows (x0, y0, x1, y1) covering a W x H image; neighbours overlap by `overlap` of a tile."""
    cols, rows = grid
    tw, th = W / (cols - (cols - 1) * overlap), H / (rows - (rows - 1) * overlap)
    xs = [round(i * tw * (1 - overlap)) for i in range(cols)]
    ys = [round(j * th * (1 - overlap)) for j in range(rows)]
    return [(x, y, min(W, round(x + tw)), min(H, round(y + th))) for y in ys for x in xs]


class TiledDetector:
    """Runs a detector on the full image and on overlapping tiles (optionally also mirrored), maps the
    tile boxes back to image coordinates and merges everything with class-wise NMS.

    A tile box touching an inner tile border (within `margin` px) is dropped: that object is cut by the
    tile and appears whole in the neighbouring tile or in the full image. Images narrower than
    `min_width` are processed only as a whole."""

    def __init__(self, model, grid=(2, 2), overlap: float = 0.2, full: bool = True, flip: bool = False,
                 min_width: int = 0, nms_iou: float = 0.5, max_det: int = 500, chunk: int = 4,
                 margin: float = 2.0):
        self.model, self.grid, self.overlap, self.full, self.flip = model, tuple(grid), overlap, full, flip
        self.min_width, self.nms_iou, self.max_det, self.chunk, self.margin = min_width, nms_iou, max_det, chunk, margin

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def windows(self, W: int, H: int) -> list[tuple[int, int, int, int]]:
        wins = [(0, 0, W, H)] if self.full else []
        if self.grid != (1, 1) and W >= self.min_width:
            wins += tile_windows(W, H, self.grid, self.overlap)
        return wins or [(0, 0, W, H)]

    @torch.no_grad()
    def __call__(self, images):
        return [self._one(img) for img in images]

    def _one(self, img):
        H, W = img.shape[-2:]
        jobs = [(w, False) for w in self.windows(W, H)]
        if self.flip:
            jobs += [(w, True) for w, _ in jobs]
        boxes, scores, labels = [], [], []
        for i in range(0, len(jobs), self.chunk):
            part = jobs[i:i + self.chunk]
            crops = [img[:, y0:y1, x0:x1] for (x0, y0, x1, y1), _ in part]
            outs = self.model([c.flip(-1) if fl else c for c, (_, fl) in zip(crops, part)])
            for ((x0, y0, x1, y1), fl), o in zip(part, outs):
                b = o["boxes"].float()
                if fl:                                   # mirror the boxes back
                    b = torch.stack([(x1 - x0) - b[:, 2], b[:, 1], (x1 - x0) - b[:, 0], b[:, 3]], 1)
                keep = torch.ones(len(b), dtype=torch.bool, device=b.device)
                if x0 > 0:
                    keep &= b[:, 0] > self.margin
                if y0 > 0:
                    keep &= b[:, 1] > self.margin
                if x1 < W:
                    keep &= b[:, 2] < (x1 - x0) - self.margin
                if y1 < H:
                    keep &= b[:, 3] < (y1 - y0) - self.margin
                boxes.append(b[keep] + torch.tensor([x0, y0, x0, y0], device=b.device, dtype=b.dtype))
                scores.append(o["scores"].float()[keep])
                labels.append(o["labels"][keep])
        b, s, lab = torch.cat(boxes), torch.cat(scores), torch.cat(labels)
        k = batched_nms(b, s, lab, self.nms_iou)[: self.max_det]
        return {"boxes": b[k], "scores": s[k], "labels": lab[k]}


# ---------------------------------------------------------------- ensembling
def _xyxy(p: pd.DataFrame) -> torch.Tensor:
    b = torch.tensor(p[["x", "y", "w", "h"]].to_numpy(np.float32))
    b[:, 2:] += b[:, :2]
    return b


def merge_predictions(preds: list[pd.DataFrame], iou: float = 0.5, max_det: int = 500,
                      weights: list[float] | None = None) -> pd.DataFrame:
    """Union of several models' boxes per image, then class-wise NMS: where models agree the best-scoring
    box is kept, and objects found by only one model are added."""
    weights = weights or [1.0] * len(preds)
    p = pd.concat([d.assign(score=d["score"] * w) for d, w in zip(preds, weights)], ignore_index=True)
    out = []
    for _, g in p.groupby("file_name", sort=False):
        keep = batched_nms(_xyxy(g), torch.tensor(g["score"].to_numpy(np.float32)),
                           torch.tensor(g["cls"].to_numpy(np.int64)), iou)[:max_det]
        out.append(g.iloc[keep.numpy()])
    return pd.concat(out, ignore_index=True)[PRED_COLUMNS]


# ---------------------------------------------------------------- temporal
def temporal_rescore(preds: pd.DataFrame, frames: pd.DataFrame, window: int = 1, iou_thr: float = 0.5,
                     lam: float = 0.5) -> pd.DataFrame:
    """Temporal score smoothing (the idea of Seq-NMS, simplified). For each box, look in the `window`
    previous and next frames of the same clip for the best-overlapping box of the same class
    (IoU >= iou_thr; 0 if there is none). New score = (1 - lam) * own score + lam * mean neighbour score.
    Boxes that persist over frames keep their score; one-frame flickers are pushed down."""
    f = frames.set_index("file_name")[["clip", "frame"]]
    p = preds.join(f, on="file_name").reset_index(drop=True)
    new = p["score"].to_numpy(np.float32).copy()
    for _, g in p.groupby("clip", sort=False):
        per = {fr: (gg.index.to_numpy(), _xyxy(gg), torch.tensor(gg["cls"].to_numpy(np.int64)),
                    torch.tensor(gg["score"].to_numpy(np.float32))) for fr, gg in g.groupby("frame")}
        for fr, (idx, b, c, s) in per.items():
            support = []
            for d in range(-window, window + 1):
                if d == 0 or fr + d not in per:
                    continue
                _, nb, nc, ns = per[fr + d]
                if len(nb) == 0:
                    support.append(torch.zeros(len(b)))
                    continue
                iou = box_iou(b, nb)
                iou[c[:, None] != nc[None, :]] = 0
                support.append(((iou >= iou_thr) * ns[None, :]).max(1).values)
            if support:
                new[idx] = ((1 - lam) * s + lam * torch.stack(support).mean(0)).numpy()
    return p.assign(score=new)[PRED_COLUMNS]


# ---------------------------------------------------------------- earlier runs
RUN_FILES = ("config.yaml", "metrics.json", "stages.json", "history.csv", "checkpoints/best.pt",
             "predictions/val.csv", "predictions/test.csv")


def fetch_kaggle_runs(paths: Paths, kernel: str, run_ids: list[str], files: tuple[str, ...] = RUN_FILES) -> None:
    """Make sure runs trained in an earlier Kaggle notebook are in paths.runs.
    On Kaggle: attach that notebook's output as an input (env.setup copies it). Elsewhere: download only
    the needed files with the Kaggle CLI (needs the Kaggle API token)."""
    missing = [r for r in run_ids if not (paths.runs / r / "predictions" / "val.csv").exists()]
    if not missing:
        print("runs present:", ", ".join(run_ids))
        return
    if paths.platform == "kaggle":
        raise FileNotFoundError(f"runs {missing} not found: add the output of '{kernel}' as an input "
                                f"(Add Input -> Your Work -> Notebooks), then run the setup cell again")
    from .data.download import _load_colab_secrets
    _load_colab_secrets()
    pattern = rf"^outputs/runs/({'|'.join(map(re.escape, missing))})/({'|'.join(map(re.escape, files))})$"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["kaggle", "kernels", "output", kernel, "-p", tmp, "--file-pattern", pattern],
                       check=True, capture_output=True, text=True)
        for r in missing:
            src = Path(tmp) / "outputs" / "runs" / r
            if not src.exists():
                raise FileNotFoundError(f"run {r} is not in the output of {kernel}")
            shutil.copytree(src, paths.runs / r, dirs_exist_ok=True)
            print(f"downloaded run {r} from {kernel}")
