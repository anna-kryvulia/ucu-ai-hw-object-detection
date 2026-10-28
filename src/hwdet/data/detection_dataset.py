"""PyTorch dataset for torchvision detectors (Faster R-CNN, RetinaNet, FCOS).

Conversion: COCO annotations -> one target per frame in the torchvision format
    {"boxes": FloatTensor[N, 4] (x1, y1, x2, y2 in pixels), "labels": Int64Tensor[N] (1..8, 0 = background)}

Ignored regions ("don't care" areas): painted with the ImageNet mean colour, which becomes exactly 0 after
the model's own normalisation, and labelled boxes whose centre lies inside such a region are dropped.
The model therefore never sees them as background. At evaluation time pycocotools skips detections in
these regions (iscrowd = 1), so validation images are left untouched.

Images are returned as float tensors in [0, 1]. Resizing and mean/std normalisation happen inside the
torchvision model (GeneralizedRCNNTransform), so they are identical for training and inference.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .io import annotations_table

IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406]) * 255


def build_targets(data_root: Path, frames: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """file_name -> {"boxes": xyxy (N, 4), "labels": 1..8 (N,), "ignore": xyxy (M, 4)} for all frames."""
    out = {}
    for split, part in frames.groupby("split"):
        a = annotations_table(data_root, split)
        a = a[a.coco_id.isin(part.coco_id)].copy()
        a["x2"], a["y2"] = a.x + a.w, a.y + a.h
        objs = a[a.iscrowd == 0]
        ign = a[a.iscrowd == 1].drop_duplicates(["coco_id", "x", "y", "w", "h"])   # stored once per class
        objs_by = {k: g for k, g in objs.groupby("coco_id")}
        ign_by = {k: g for k, g in ign.groupby("coco_id")}
        for r in part.itertuples():
            o, i = objs_by.get(r.coco_id), ign_by.get(r.coco_id)
            out[r.file_name] = {
                "boxes": np.zeros((0, 4), np.float32) if o is None else o[["x", "y", "x2", "y2"]].to_numpy(np.float32),
                "labels": np.zeros(0, np.int64) if o is None else o.cls.to_numpy(np.int64) + 1,
                "ignore": np.zeros((0, 4), np.float32) if i is None else i[["x", "y", "x2", "y2"]].to_numpy(np.float32),
            }
    return out


def centres_inside(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Bool mask over boxes (xyxy): centre lies inside any region (xyxy)."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes), bool)
    cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
    r = regions
    return ((cx[:, None] >= r[None, :, 0]) & (cx[:, None] <= r[None, :, 2]) &
            (cy[:, None] >= r[None, :, 1]) & (cy[:, None] <= r[None, :, 3])).any(1)


class DetectionDataset(Dataset):
    """train=True: ignored regions are masked and the augmentations below are applied.
    train=False: the raw image is returned (validation / test / prediction)."""

    def __init__(self, frames: pd.DataFrame, data_root: Path, targets: dict | None = None, train: bool = False,
                 mask_ignore: bool = True, hflip: float = 0.5, color_jitter: float = 0.0, min_box: float = 2.0):
        self.frames = frames.reset_index(drop=True)
        self.root = Path(data_root)
        self.targets = targets
        self.train = train
        self.mask_ignore = mask_ignore
        self.hflip = hflip
        self.color_jitter = color_jitter
        self.min_box = min_box

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        r = self.frames.iloc[i]
        img = np.asarray(Image.open(self.root / r.file_name).convert("RGB"))
        if self.targets is None:                               # test images: no labels
            return torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255, {}, i

        t = self.targets[r.file_name]
        boxes, labels = t["boxes"].copy(), t["labels"].copy()
        if self.train and self.mask_ignore and len(t["ignore"]):
            img = img.copy()
            for x1, y1, x2, y2 in t["ignore"].astype(int):
                img[max(y1, 0):y2, max(x1, 0):x2] = IMAGENET_MEAN_RGB
            keep = ~centres_inside(boxes, t["ignore"])
            boxes, labels = boxes[keep], labels[keep]

        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255
        if self.train and random.random() < self.hflip:        # horizontal flip: image and boxes together
            img = img.flip(-1)
            W = img.shape[-1]
            boxes = np.stack([W - boxes[:, 2], boxes[:, 1], W - boxes[:, 0], boxes[:, 3]], 1).reshape(-1, 4)
        if self.train and self.color_jitter > 0:               # brightness / contrast jitter
            j = self.color_jitter
            img = ((img - img.mean()) * random.uniform(1 - j, 1 + j) + img.mean()
                   + random.uniform(-j / 3, j / 3)).clamp(0, 1)

        wh = boxes[:, 2:] - boxes[:, :2]
        ok = (wh >= self.min_box).all(1)                       # drop degenerate boxes
        target = {"boxes": torch.as_tensor(boxes[ok], dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.as_tensor(labels[ok], dtype=torch.int64)}
        return img, target, i


def collate(batch):
    return tuple(zip(*batch))
