"""Reading the competition files.

Layout (relative to the data root):
    train|val/images/<clip>/<frame>.jpg
    train|val/labels/<clip>/<frame>.txt     class cx cy w h  (normalised)
    train|val/ignore/<clip>/<frame>.txt     cx cy w h        (normalised, only some frames)
    train|val/annotations_coco.json         same labels, COCO, category_id = class + 1, ignore = iscrowd
    test/images/<clip>/<frame>.jpg, test_images.csv

Common prediction format used everywhere in this repo (one row per box, pixels, top-left corner):
    file_name, cls, score, x, y, w, h        file_name relative to the data root
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PRED_COLUMNS = ["file_name", "cls", "score", "x", "y", "w", "h"]


def frame_key(path: str | Path) -> str:
    """'train/images/clip_007/0000123.jpg' -> 'clip_007/0000123'. Same key for images/labels/coco."""
    p = Path(path)
    return f"{p.parent.name}/{p.stem}"


def label_path(data_root: Path, file_name: str) -> Path:
    return (data_root / file_name.replace("/images/", "/labels/", 1)).with_suffix(".txt")


def ignore_path(data_root: Path, file_name: str) -> Path:
    return (data_root / file_name.replace("/images/", "/ignore/", 1)).with_suffix(".txt")


def read_txt(path: Path, ncols: int) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, ncols), dtype=np.float32)
    return np.loadtxt(path, dtype=np.float32, ndmin=2).reshape(-1, ncols)


def cxcywh_norm_to_xywh_px(a: np.ndarray, w: int, h: int) -> np.ndarray:
    """(N,4) normalised centre boxes -> (N,4) pixel top-left boxes."""
    out = a.copy()
    out[:, 0] = (a[:, 0] - a[:, 2] / 2) * w
    out[:, 1] = (a[:, 1] - a[:, 3] / 2) * h
    out[:, 2] = a[:, 2] * w
    out[:, 3] = a[:, 3] * h
    return out


def load_gt(data_root: Path, file_name: str, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns classes (N,), boxes xywh px (N,4), ignored regions xywh px (M,4)."""
    lab = read_txt(label_path(data_root, file_name), 5)
    ign = read_txt(ignore_path(data_root, file_name), 4)
    return lab[:, 0].astype(int), cxcywh_norm_to_xywh_px(lab[:, 1:], w, h), cxcywh_norm_to_xywh_px(ign, w, h)


@lru_cache(maxsize=4)
def load_coco(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def frame_table(data_root: Path, split: str) -> pd.DataFrame:
    """One row per frame: file_name, split, clip, frame, width, height, coco_id.
    Sizes come from the COCO json, so no image has to be opened."""
    data_root = Path(data_root)
    if split == "test":
        df = pd.read_csv(data_root / "test_images.csv")
        df["split"] = "test"
        df["coco_id"] = -1
        return df[["file_name", "split", "clip", "frame", "width", "height", "image_id", "coco_id"]]

    coco = load_coco(str(data_root / split / "annotations_coco.json"))
    rows = []
    for im in coco["images"]:
        key = frame_key(im["file_name"])
        clip, stem = key.split("/")
        rows.append({
            "file_name": f"{split}/images/{clip}/{stem}.jpg",
            "split": split,
            "clip": clip,
            "frame": int(im.get("frame_id", stem)),
            "width": im["width"],
            "height": im["height"],
            "coco_id": im["id"],
        })
    return pd.DataFrame(rows).sort_values(["clip", "frame"]).reset_index(drop=True)


def annotations_table(data_root: Path, split: str) -> pd.DataFrame:
    """All boxes of a split as one DataFrame (handy for EDA). Includes ignored regions (iscrowd=1)
    and the extra VisDrone fields (occlusion, truncation, track_id) when present."""
    coco = load_coco(str(Path(data_root) / split / "annotations_coco.json"))
    df = pd.DataFrame(coco["annotations"]).drop(columns=["id", "area"], errors="ignore")
    df = df.rename(columns={"image_id": "coco_id"})
    df[["x", "y", "w", "h"]] = np.array(df.pop("bbox").tolist(), dtype=np.float32)
    df["cls"] = df.pop("category_id") - 1                       # iscrowd regions keep their own id
    if "iscrowd" not in df:
        df["iscrowd"] = 0

    frames = frame_table(data_root, split).set_index("coco_id")
    df = df.join(frames[["clip", "frame", "width", "height"]].rename(
        columns={"width": "img_w", "height": "img_h"}), on="coco_id")
    df["area"] = df["w"] * df["h"]
    df["size"] = pd.cut(df["area"], [0, 32 ** 2, 96 ** 2, np.inf], labels=["small", "medium", "large"])
    return df


def empty_preds() -> pd.DataFrame:
    return pd.DataFrame(columns=PRED_COLUMNS)
