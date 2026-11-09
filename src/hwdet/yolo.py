"""YOLO (Ultralytics) in the shared format: dataset export, resumable training, torchvision-style wrapper.

    n = export_split(train_frames, targets, P.data, P.cache / "yolo" / "train", **cfg["export"])
    train_yolo(cfg, run, data_yaml)
    det = YoloDetector(run.ckpt("best.pt"), **predict_kwargs(cfg))   # then engine.predict(det, ...)
"""
from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from .data.classes import CLASS_NAMES
from .data.detection_dataset import IMAGENET_MEAN_RGB, centres_inside
from .pipeline import tile_windows


# ---------------------------------------------------------------- dataset export
def _export_frame(job) -> int:
    """One frame -> image(s) + YOLO label file(s): ignored regions painted over, boxes clipped to each crop."""
    src, stem, t, out_dir, long_side, tiles, overlap, min_width, min_visible, mask = job
    img = np.array(Image.open(src).convert("RGB"))
    H, W = img.shape[:2]
    boxes, labels = t["boxes"], t["labels"] - 1                   # YOLO classes are 0..7
    if mask and len(t["ignore"]):
        for x1, y1, x2, y2 in t["ignore"].astype(int):
            img[max(y1, 0):y2, max(x1, 0):x2] = IMAGENET_MEAN_RGB
        keep = ~centres_inside(boxes, t["ignore"])
        boxes, labels = boxes[keep], labels[keep]
    wins = [(0, 0, W, H)]
    if tiles and W >= min_width:
        wins += tile_windows(W, H, tuple(tiles), overlap)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for k, (x0, y0, x1, y1) in enumerate(wins):
        b = boxes.copy()
        b[:, [0, 2]] = b[:, [0, 2]].clip(x0, x1) - x0
        b[:, [1, 3]] = b[:, [1, 3]].clip(y0, y1) - y0
        bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        ok = (bw * bh >= min_visible * area) & (bw >= 2) & (bh >= 2)   # mostly inside the crop
        cw, ch = x1 - x0, y1 - y0
        crop = Image.fromarray(img[y0:y1, x0:x1])
        s = min(1.0, long_side / max(cw, ch))
        if s < 1:
            crop = crop.resize((round(cw * s), round(ch * s)), Image.BILINEAR)
        name = f"{stem}_t{k}" if k else stem
        crop.save(out_dir / "images" / f"{name}.jpg", quality=92)
        rows = np.column_stack([labels[ok], (b[ok, 0] + b[ok, 2]) / 2 / cw, (b[ok, 1] + b[ok, 3]) / 2 / ch,
                                bw[ok] / cw, bh[ok] / ch])
        np.savetxt(out_dir / "labels" / f"{name}.txt", rows, fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f"])
    return len(wins)


def export_split(frames: pd.DataFrame, targets: dict, data_root: Path, out_dir: Path, long_side: int = 1280,
                 tile_grid=None, tile_overlap: float = 0.2, tile_min_width: int = 2000, min_visible: float = 0.5,
                 mask_ignore: bool = True, workers: int = 4) -> int:
    """Write frames (+ tiles of large frames) as a YOLO folder out_dir/images, out_dir/labels.
    Returns the number of images; skips the work if the folder is already complete."""
    out_dir = Path(out_dir)
    done = out_dir / "done.txt"
    if done.exists():
        return int(done.read_text())
    shutil.rmtree(out_dir, ignore_errors=True)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)
    jobs = [(Path(data_root) / r.file_name, f"{r.clip}_{Path(r.file_name).stem}", targets[r.file_name], out_dir,
             long_side, tile_grid, tile_overlap, tile_min_width, min_visible, mask_ignore)
            for r in frames.itertuples()]
    with ProcessPoolExecutor(workers) as ex:
        n = sum(ex.map(_export_frame, jobs, chunksize=16))
    done.write_text(str(n))
    return n


def write_data_yaml(root: Path) -> Path:
    path = Path(root) / "data.yaml"
    path.write_text(yaml.safe_dump({"path": str(root), "train": "train/images", "val": "val/images",
                                    "names": dict(enumerate(CLASS_NAMES))}, sort_keys=False))
    return path


# ---------------------------------------------------------------- training
def train_yolo(cfg: dict, run, data_yaml: Path) -> pd.DataFrame:
    """Ultralytics training into run.dir/ultra (resumes from last.pt after a crash);
    the best weights are copied to run.ckpt('best.pt') and the per-epoch log to history.csv."""
    from ultralytics import YOLO
    m, t = cfg["model"], cfg["train"]
    ultra = run.dir / "ultra"
    last = ultra / "weights" / "last.pt"
    if last.exists():
        print("resuming from", last)
        YOLO(str(last)).train(resume=True)
    else:
        model = YOLO(m["arch"]).load(m["weights"])        # COCO weights into every layer with a matching shape
        model.train(data=str(data_yaml), imgsz=m["imgsz"], epochs=t["epochs"], time=t["hours"],
                    batch=t["batch_size"], workers=t["workers"], optimizer=t["optimizer"],
                    close_mosaic=t["close_mosaic"], patience=t["patience"], seed=cfg.get("seed", 42),
                    deterministic=False, amp=t.get("amp", True), plots=False, verbose=False,
                    project=str(run.dir), name="ultra", exist_ok=True)
    shutil.copy2(ultra / "weights" / "best.pt", run.ckpt("best.pt"))
    res = pd.read_csv(ultra / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    hist = pd.DataFrame({"epoch": res["epoch"] - (1 if res["epoch"].min() == 1 else 0),
                         "box_loss": res["train/box_loss"], "cls_loss": res["train/cls_loss"],
                         "dfl_loss": res["train/dfl_loss"], "val_mAP": res["metrics/mAP50-95(B)"],
                         "val_AP50": res["metrics/mAP50(B)"]})
    for row in hist.to_dict("records"):
        run.log_epoch(int(row.pop("epoch")), row)
    return hist


# ---------------------------------------------------------------- inference wrapper
def predict_kwargs(cfg: dict) -> dict:
    m = cfg["model"]
    return {"imgsz": m["imgsz"], "conf": m["conf"], "iou": m["iou"], "max_det": m["max_det"]}


class YoloDetector:
    """Ultralytics model behind the torchvision interface: list of CHW float RGB images in 0-1 ->
    list of {"boxes" xyxy pixels, "scores", "labels" 1..8}."""

    def __init__(self, weights: Path, imgsz: int = 1280, conf: float = 0.001, iou: float = 0.5, max_det: int = 500):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.imgsz, self.conf, self.iou, self.max_det = imgsz, conf, iou, max_det
        self.device = "cpu"

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    @torch.no_grad()
    def __call__(self, images):
        arrs = [np.ascontiguousarray((im.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[..., ::-1])
                for im in images]                                    # Ultralytics expects BGR uint8
        res = self.model.predict(arrs, imgsz=self.imgsz, conf=self.conf, iou=self.iou, max_det=self.max_det,
                                 half=self.device == "cuda", device=0 if self.device == "cuda" else self.device,
                                 verbose=False)
        return [{"boxes": r.boxes.xyxy, "scores": r.boxes.conf, "labels": r.boxes.cls.long() + 1} for r in res]
