"""COCO-style evaluation on any set of labelled frames, from predictions in the common format.

Reports what the assignment asks for (mAP, AP50, AP_small, ...) plus per-class AP.
Ignored regions are iscrowd=1 in the GT, so pycocotools does not penalise detections there.
maxDets is 500, the limit the competition scores, instead of COCO's default 100 (scenes are dense).

Note: the leaderboard uses a custom metric. Check the competition's Evaluation section and, if it
differs from COCO mAP@[.5:.95], add it here so the validation score tracks the leaderboard.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.classes import CLASS_NAMES
from ..data.io import frame_key, load_coco

try:  # faster drop-in replacement, same API
    from faster_coco_eval import COCO, COCOeval_faster as COCOeval
except ImportError:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

MAX_DETS = 500
STAT_NAMES = ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
              "AR_10", "AR_100", "AR_500", "AR_small", "AR_medium", "AR_large"]


def build_gt(frames: pd.DataFrame, data_root: Path) -> tuple[dict, dict]:
    """COCO dict restricted to `frames` (can mix train/ and val/), with fresh image ids.
    Returns (coco_dict, file_name -> image_id)."""
    images, anns, id_of = [], [], {}
    for split, part in frames.groupby("split"):
        src = load_coco(str(Path(data_root) / split / "annotations_coco.json"))
        by_key = {frame_key(im["file_name"]): im for im in src["images"]}
        old_to_new = {}
        for fn in part["file_name"]:
            im = by_key[frame_key(fn)]
            new_id = len(images) + 1
            old_to_new[im["id"]] = new_id
            id_of[fn] = new_id
            images.append({**im, "id": new_id, "file_name": fn})
        for a in src["annotations"]:
            if a["image_id"] in old_to_new:
                anns.append({**a, "id": len(anns) + 1, "image_id": old_to_new[a["image_id"]],
                             "area": a.get("area", a["bbox"][2] * a["bbox"][3])})
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CLASS_NAMES)]
    return {"images": images, "annotations": anns, "categories": cats}, id_of


def evaluate(preds: pd.DataFrame, frames: pd.DataFrame, data_root: Path, per_class: bool = True,
             verbose: bool = False) -> dict:
    gt_dict, id_of = build_gt(frames, data_root)
    gt = COCO()
    gt.dataset = gt_dict
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()

    p = preds[preds["file_name"].isin(id_of)]
    if len(p) == 0:
        return {k: 0.0 for k in STAT_NAMES}
    dets = np.column_stack([p["file_name"].map(id_of), p[["x", "y", "w", "h", "score"]].to_numpy(),
                            p["cls"].to_numpy() + 1])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, "bbox")
        E.params.maxDets = [10, 100, MAX_DETS]
        E.evaluate()
        E.accumulate()
        E.summarize()
    if verbose:
        print(out.getvalue())

    res = dict(zip(STAT_NAMES, map(float, E.stats[:12])))
    if per_class:
        prec = E.eval["precision"]          # [T, R, K, A, M]
        for k, name in enumerate(CLASS_NAMES):
            pk = prec[:, :, k, 0, -1]
            res[f"AP_{name}"] = float(pk[pk > -1].mean()) if (pk > -1).any() else float("nan")
    return res
