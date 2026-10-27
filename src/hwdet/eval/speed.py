"""Inference speed, measured the same way for every model (required in the report)."""
from __future__ import annotations

import time
from typing import Callable, Sequence


def measure_fps(predict_one: Callable[[object], object], inputs: Sequence, warmup: int = 5,
                n: int = 50) -> dict:
    """predict_one(x) runs the model on one image (batch size 1, including pre/post-processing)."""
    try:
        import torch
        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except ImportError:
        sync = lambda: None  # noqa: E731
    for x in inputs[:warmup]:
        predict_one(x)
    sync()
    sample = list(inputs[warmup:warmup + n]) or list(inputs[:n])
    t0 = time.perf_counter()
    for x in sample:
        predict_one(x)
    sync()
    dt = (time.perf_counter() - t0) / len(sample)
    return {"ms_per_img": dt * 1000, "fps": 1 / dt}
