"""Error analysis for failure cases: what happened to every detection and to every ground-truth object.

Detections of a frame are matched greedily by score (as in COCO evaluation) at IoU >= 0.5 and labelled:
    correct            same class, matches a ground-truth box not matched yet
    duplicate          same class, but that ground-truth box was already matched by a higher score
    wrong_class        overlaps a ground-truth box of another class
    poor_localisation  0.1 <= IoU < 0.5 with some ground-truth box
    background         no overlap with any object
    ignored            centre inside an ignored region (not scored by the leaderboard)
Ground-truth boxes keep the score of the detection that found them, so recall at any score threshold
is a simple comparison; greedy matching makes the result identical to re-matching at that threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.classes import CLASS_NAMES

OUTCOMES = ["correct", "duplicate", "wrong_class", "poor_localisation", "background", "ignored"]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between xyxy boxes a (N, 4) and b (M, 4)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def match(preds: pd.DataFrame, frames: pd.DataFrame, targets: dict, iou_thr: float = 0.5,
          min_score: float = 0.05) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (detections with `outcome` and box x, y, w, h; ground truth with `found_score` and `size`)."""
    preds = preds[preds.score >= min_score]
    by_frame = dict(tuple(preds.groupby("file_name")))
    det_rows, gt_rows = [], []
    for fn in frames.file_name:
        t = targets[fn]
        gb, gl, ig = t["boxes"], t["labels"] - 1, t["ignore"]
        found = np.full(len(gb), np.nan)
        p = by_frame.get(fn)
        if p is not None and len(p):
            p = p.sort_values("score", ascending=False, kind="stable")
            db = p[["x", "y", "w", "h"]].to_numpy(np.float32)
            db[:, 2:] += db[:, :2]
            I = _iou(db, gb) if len(gb) else np.zeros((len(db), 0))
            cx, cy = (db[:, 0] + db[:, 2]) / 2, (db[:, 1] + db[:, 3]) / 2
            in_ign = ((cx[:, None] >= ig[None, :, 0]) & (cx[:, None] <= ig[None, :, 2]) &
                      (cy[:, None] >= ig[None, :, 1]) & (cy[:, None] <= ig[None, :, 3])).any(1) if len(ig) else \
                np.zeros(len(db), bool)
            for k, (c, s) in enumerate(zip(p.cls.to_numpy(), p.score.to_numpy())):
                row = I[k] if len(gb) else np.zeros(0)
                same = (gl == c)
                free = same & np.isnan(found)
                if free.any() and row[free].max() >= iou_thr:
                    j = np.where(free)[0][row[free].argmax()]
                    found[j] = s
                    out = "correct"
                elif same.any() and row[same].max() >= iou_thr:
                    out = "duplicate"
                elif len(row) and row.max() >= iou_thr:
                    out = "wrong_class"
                elif in_ign[k]:
                    out = "ignored"
                elif len(row) and row.max() >= 0.1:
                    out = "poor_localisation"
                else:
                    out = "background"
                det_rows.append((fn, int(c), float(s), out, *(db[k, :2]), *(db[k, 2:] - db[k, :2])))
        area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
        for c, a, f in zip(gl, area, found):
            gt_rows.append((fn, int(c), float(a), f))
    dets = pd.DataFrame(det_rows, columns=["file_name", "cls", "score", "outcome", "x", "y", "w", "h"])
    gts = pd.DataFrame(gt_rows, columns=["file_name", "cls", "area", "found_score"])
    gts["size"] = pd.cut(gts.area, [0, 32 ** 2, 96 ** 2, np.inf], labels=["small", "medium", "large"])
    return dets, gts


def best_f1_threshold(dets: pd.DataFrame, gts: pd.DataFrame) -> tuple[float, float]:
    """Score threshold with the best F1 (correct detections vs scored false positives vs misses)."""
    best = (0.5, 0.0)
    for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
        d = dets[(dets.score >= thr) & (dets.outcome != "ignored")]
        tp = (d.outcome == "correct").sum()
        f1 = 2 * tp / (len(d) + len(gts) + 1e-9)
        if f1 > best[1]:
            best = (float(thr), float(f1))
    return best


def summary(dets: pd.DataFrame, gts: pd.DataFrame, thr: float) -> dict[str, pd.DataFrame]:
    """Detection outcomes, and recall per object size and per class, at score threshold `thr`."""
    d = dets[dets.score >= thr]
    found = gts.found_score >= thr
    return {
        "outcomes": d.outcome.value_counts().reindex(OUTCOMES, fill_value=0).to_frame("detections"),
        "recall_by_size": gts.assign(found=found).groupby("size", observed=False).found.agg(["mean", "size"])
                             .rename(columns={"mean": "recall", "size": "objects"}),
        "recall_by_class": gts.assign(found=found, cls=gts.cls.map(dict(enumerate(CLASS_NAMES))))
                              .groupby("cls").found.agg(["mean", "size"])
                              .rename(columns={"mean": "recall", "size": "objects"}),
    }


def worst_frames(dets: pd.DataFrame, gts: pd.DataFrame, thr: float, n: int = 3) -> pd.DataFrame:
    """Frames with the most errors at `thr`: missed objects + scored false positives."""
    d = dets[dets.score >= thr]
    fp = d[~d.outcome.isin(["correct", "ignored"])].groupby("file_name").size()
    missed = gts[~(gts.found_score >= thr)].groupby("file_name").size()
    t = pd.DataFrame({"missed": missed, "false_pos": fp}).fillna(0).astype(int)
    t["errors"] = t.missed + t.false_pos
    return t.sort_values("errors", ascending=False).head(n)
