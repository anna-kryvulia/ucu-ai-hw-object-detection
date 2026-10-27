"""Turn test predictions (common format) into the Kaggle CSV, and check it before uploading.

Kaggle format: image_id,PredictionString with groups "class score x y w h" (pixels, top-left corner),
class ids 0-7, empty string for an image with no boxes, at most 500 boxes per image scored.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data.classes import NUM_CLASSES
from .data.io import PRED_COLUMNS

MAX_DETS = 500


def clip_to_image(preds: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    size = frames.set_index("file_name")[["width", "height"]]
    p = preds.join(size, on="file_name")
    x0, y0 = p["x"].clip(lower=0), p["y"].clip(lower=0)
    x1, y1 = (p["x"] + p["w"]).clip(upper=p["width"]), (p["y"] + p["h"]).clip(upper=p["height"])
    p = p.assign(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
    return p[(p["w"] > 0) & (p["h"] > 0)][PRED_COLUMNS].reset_index(drop=True)


def top_k(preds: pd.DataFrame, k: int = MAX_DETS) -> pd.DataFrame:
    """The leaderboard scores at most 500 boxes per image: keep the highest-scoring ones."""
    return (preds.sort_values("score", ascending=False).groupby("file_name", sort=False).head(k)
            .reset_index(drop=True))


def make_submission(preds: pd.DataFrame, test_frames: pd.DataFrame, out_path: Path | None = None,
                    score_thr: float = 0.0) -> pd.DataFrame:
    """test_frames = data.io.frame_table(data_root, 'test') (has image_id and file_name)."""
    p = preds[preds["score"] >= score_thr]
    p = top_k(clip_to_image(p, test_frames), MAX_DETS)
    p = p.sort_values(["file_name", "score"], ascending=[True, False])
    strings = p.groupby("file_name").apply(
        lambda g: " ".join(f"{int(c)} {s:.4f} {x:.1f} {y:.1f} {w:.1f} {h:.1f}"
                           for c, s, x, y, w, h in g[["cls", "score", "x", "y", "w", "h"]].itertuples(index=False)),
        include_groups=False,
    )
    sub = test_frames[["image_id", "file_name"]].copy()
    sub["PredictionString"] = sub["file_name"].map(strings).fillna("")
    sub = sub[["image_id", "PredictionString"]]
    validate_submission(sub, test_frames)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(sub)} rows, {len(p)} boxes)")
    return sub


def validate_submission(sub: pd.DataFrame, test_frames: pd.DataFrame) -> None:
    assert list(sub.columns) == ["image_id", "PredictionString"], sub.columns
    assert set(sub["image_id"]) == set(test_frames["image_id"]), "image_id set differs from test_images.csv"
    assert sub["image_id"].is_unique
    size = test_frames.set_index("image_id")[["width", "height"]]
    for image_id, s in zip(sub["image_id"], sub["PredictionString"].fillna("")):
        vals = s.split()
        assert len(vals) % 6 == 0, f"{image_id}: groups of 6 expected"
        assert len(vals) // 6 <= MAX_DETS, f"{image_id}: more than {MAX_DETS} boxes"
        W, H = size.loc[image_id]
        for i in range(0, len(vals), 6):
            c, sc, x, y, w, h = int(vals[i]), *map(float, vals[i + 1:i + 6])
            assert 0 <= c < NUM_CLASSES and 0 <= sc <= 1, f"{image_id}: bad class/score"
            assert x >= -1 and y >= -1 and x + w <= W + 1 and y + h <= H + 1, f"{image_id}: box outside image"
    print("submission looks valid")
