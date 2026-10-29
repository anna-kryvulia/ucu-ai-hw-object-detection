"""Training / evaluation for torchvision detectors (Faster R-CNN now, RetinaNet / FCOS later), crash-safe.

    run = run_experiment("frcnn_baseline", build_frcnn, P, data)   # train -> evaluate on val -> save

Every stage is tracked by hwdet.tracker.Run, so re-running a notebook cell after a crash skips finished
stages and resumes training from checkpoints/last.pt (written after every epoch).

Saved per run (outputs/runs/<run_id>/): config, per-epoch history, val metrics (mAP, AP50, AP_small, ...,
speed), val predictions (and test predictions + submission when asked), last/best checkpoints.
Keeping predictions and checkpoints means the bonus tasks (temporal post-processing, Grad-CAM,
embeddings) can reuse a trained model without training it again.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import load_config
from .data.detection_dataset import DetectionDataset, collate
from .data.io import PRED_COLUMNS
from .env import Paths, device as get_device, seed_everything
from .eval.coco_eval import evaluate
from .eval.speed import measure_fps
from .submission import make_submission
from .tracker import Run


@dataclass
class Data:
    """Frames and targets shared by all experiments of a notebook (built once)."""
    root: Path
    train: pd.DataFrame          # training frames (already subsampled)
    val: pd.DataFrame            # full validation split: final metrics
    monitor: pd.DataFrame        # validation subset: mAP after each epoch
    targets: dict                # file_name -> boxes / labels / ignore regions
    test: pd.DataFrame | None = None


# ---------------------------------------------------------------- optimisation
def _optimizer(model, t: dict):
    params = [p for p in model.parameters() if p.requires_grad]
    if t.get("optimizer", "sgd") == "adamw":
        return torch.optim.AdamW(params, lr=t["lr"], weight_decay=t.get("weight_decay", 1e-4))
    return torch.optim.SGD(params, lr=t["lr"], momentum=0.9, weight_decay=t.get("weight_decay", 1e-4))


def _lr_lambda(t: dict, iters_per_epoch: int):
    total, warm = t["epochs"] * iters_per_epoch, max(1, t.get("warmup_iters", 0))
    steps = [int(e * iters_per_epoch) for e in t.get("lr_steps", [])]

    def f(i):
        w = min(1.0, (i + 1) / warm)
        if t.get("schedule", "cosine") == "step":
            return w * 0.1 ** sum(i >= s for s in steps)
        return w * 0.5 * (1 + math.cos(math.pi * min(i, total) / total))
    return f


# ---------------------------------------------------------------- prediction
@torch.no_grad()
def predict(model, ds: DetectionDataset, device: str, batch_size: int = 4, workers: int = 2,
            desc: str = "predict") -> pd.DataFrame:
    """Run the model on a dataset -> predictions in the common format (pixels, top-left corner, cls 0..7)."""
    model.eval().to(device)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate)
    rows = []
    for imgs, _, idx in tqdm(dl, desc=desc, leave=False):
        outs = model([im.to(device) for im in imgs])
        for o, i in zip(outs, idx):
            fn = ds.frames.file_name.iloc[i]
            b = o["boxes"].float().cpu().numpy()
            for (x1, y1, x2, y2), s, c in zip(b, o["scores"].float().cpu().numpy(), o["labels"].cpu().numpy()):
                if c < 1:                                # index 0 = background / unused (RetinaNet, FCOS)
                    continue
                rows.append((fn, int(c) - 1, float(s), x1, y1, x2 - x1, y2 - y1))
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def inference_speed(model, ds: DetectionDataset, device: str, n: int = 50) -> dict:
    """ms per image at batch size 1, including the model's own resize / normalisation and NMS."""
    model.eval().to(device)
    imgs = [ds[i][0] for i in np.linspace(0, len(ds) - 1, n + 5).astype(int)]

    @torch.no_grad()
    def one(x):
        return model([x.to(device)])
    return measure_fps(one, imgs, warmup=5, n=n)


# ---------------------------------------------------------------- training
def fit(model, run: Run, cfg: dict, data: Data, device: str) -> None:
    t = cfg["train"]
    train_ds = DetectionDataset(data.train, data.root, data.targets, train=True,
                                mask_ignore=cfg["data"].get("mask_ignore", True),
                                hflip=t.get("hflip", 0.5), color_jitter=t.get("color_jitter", 0.0))
    monitor_ds = DetectionDataset(data.monitor, data.root, data.targets)
    dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, num_workers=t.get("workers", 2),
                    collate_fn=collate, drop_last=True, persistent_workers=t.get("workers", 2) > 0)
    model.to(device)
    opt = _optimizer(model, t)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(t, len(dl)))
    use_amp = t.get("amp", True) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start, best = 0, -1.0

    if run.ckpt("last.pt").exists():                               # resume after a crash
        ck = torch.load(run.ckpt("last.pt"), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed after epoch {ck['epoch']}")

    for epoch in range(start, t["epochs"]):
        model.train()
        sums, n, t0 = {}, 0, time.time()
        for imgs, targets, _ in tqdm(dl, desc=f"epoch {epoch}", leave=False):
            imgs = [im.to(device, non_blocking=True) for im in imgs]
            targets = [{k: v.to(device) for k, v in tg.items()} for tg in targets]
            with torch.autocast("cuda", enabled=use_amp):
                losses = model(imgs, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                print("non-finite loss, batch skipped:", {k: v.detach().item() for k, v in losses.items()})
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t.get("clip_grad", 10.0))
            scale = scaler.get_scale()
            scaler.step(opt); scaler.update()
            if scaler.get_scale() >= scale:          # AMP skips the step when it lowers the scale
                sched.step()
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.detach().item()
            n += 1

        row = {k: v / max(n, 1) for k, v in sums.items()}
        row.update(lr=opt.param_groups[0]["lr"], train_min=(time.time() - t0) / 60)
        if (epoch + 1) % t.get("eval_every", 1) == 0 or epoch == t["epochs"] - 1:
            preds = predict(model, monitor_ds, device, t["batch_size"], t.get("workers", 2), desc="monitor")
            m = evaluate(preds, data.monitor, data.root, per_class=False)
            row.update({f"val_{k}": m[k] for k in ("mAP", "AP50", "AP_small")})
            if m["mAP"] > best:
                best = m["mAP"]
                torch.save(model.state_dict(), run.ckpt("best.pt"))
        run.log_epoch(epoch, row)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best}, run.ckpt("last.pt"))
        print(f"epoch {epoch}: " + ", ".join(f"{k}={v:.4g}" for k, v in row.items()))


def load_weights(model, run: Run, which: str = "best", device: str = "cpu"):
    path = run.ckpt(f"{which}.pt") if run.ckpt(f"{which}.pt").exists() else run.ckpt("last.pt")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state))
    return model.eval().to(device)


# ---------------------------------------------------------------- one experiment, end to end
def run_experiment(config: str, build_model: Callable[[dict], torch.nn.Module], paths: Paths, data: Data,
                   predict_test: bool = False, run_id: str | None = None, **overrides) -> Run:
    """train -> best checkpoint -> full-val predictions + metrics + speed [-> test predictions + submission].
    Safe to call again after a crash: finished stages are skipped."""
    cfg = load_config(config, **overrides)
    run = Run(run_id or cfg["name"], cfg, paths)
    seed_everything(cfg.get("seed", 42))
    dev = get_device()
    model = build_model(cfg)
    t = cfg["train"]

    if not run.is_done("train"):
        fit(model, run, cfg, data, dev)
        run.mark_done("train")
    model = load_weights(model, run, "best", dev)

    if not run.is_done("eval_val"):
        val_ds = DetectionDataset(data.val, data.root, data.targets)
        preds = predict(model, val_ds, dev, t["batch_size"], t.get("workers", 2), desc="val")
        preds.to_csv(run.pred_path("val"), index=False)
        metrics = evaluate(preds, data.val, data.root)
        metrics.update(inference_speed(model, val_ds, dev))
        run.log_metrics("val", metrics)
        run.mark_done("eval_val")
        if run._wandb:
            from .wandb_utils import log_predictions
            log_predictions(run, data.val, preds, data.root)

    if predict_test and not run.is_done("test"):
        test_ds = DetectionDataset(data.test, data.root)
        preds = predict(model, test_ds, dev, t["batch_size"], t.get("workers", 2), desc="test")
        preds.to_csv(run.pred_path("test"), index=False)
        make_submission(preds, data.test, run.dir / "submission.csv")
        run.mark_done("test")

    print(pd.Series(run.metrics.get("val", {})).round(4).to_string())
    run.finish()
    return run
