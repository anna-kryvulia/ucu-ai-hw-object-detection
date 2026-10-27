"""Where am I running? One place that resolves data/output folders for local, Colab and Kaggle.

Every notebook calls `env.setup()` first, so the rest of the code never hardcodes a path.

    local  : data -> <repo>/data            outputs -> <repo>/outputs
    colab  : data -> /content/data          outputs -> Google Drive (survives a crashed session)
    kaggle : data -> /kaggle/input/<comp>   outputs -> /kaggle/working/outputs (kept on "Save Version")

Override any of them with the HWDET_DATA / HWDET_OUTPUTS / HWDET_CACHE environment variables.
"""
from __future__ import annotations

import os
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

COMPETITION = "multiple-object-video-detection-hw-2-ucu-ai-2026"
REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVE_ROOT = Path("/content/drive/MyDrive/ucu-hw2")


def detect_platform() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists():
        return "kaggle"
    if "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    return "local"


@dataclass
class Paths:
    platform: str
    data: Path      # folder that contains train/ val/ test/ test_images.csv
    outputs: Path   # persistent: runs/<run_id>/ with checkpoints, metrics, predictions
    cache: Path     # fast throwaway disk: split lists, masked images
    repo: Path = REPO_ROOT

    @property
    def runs(self) -> Path:
        return self.outputs / "runs"

    def __str__(self) -> str:
        return "\n".join(f"{k:>8}: {v}" for k, v in self.__dict__.items())


def _kaggle_data_dir() -> Path:
    for cand in (Path("/kaggle/input") / COMPETITION, Path("/kaggle/input/competitions") / COMPETITION):
        if cand.exists():
            return cand
    return Path("/kaggle/input") / COMPETITION


def get_paths(mount_drive: bool = True) -> Paths:
    platform = detect_platform()
    if platform == "kaggle":
        data, outputs, cache = _kaggle_data_dir(), Path("/kaggle/working/outputs"), Path("/tmp/hwdet_cache")
    elif platform == "colab":
        if mount_drive and not Path("/content/drive/MyDrive").exists():
            from google.colab import drive  # type: ignore
            drive.mount("/content/drive")
        data, outputs, cache = Path("/content/data"), DRIVE_ROOT / "outputs", Path("/content/cache")
    else:
        data, outputs, cache = REPO_ROOT / "data", REPO_ROOT / "outputs", REPO_ROOT / "outputs" / "cache"

    paths = Paths(
        platform=platform,
        data=Path(os.environ.get("HWDET_DATA", data)),
        outputs=Path(os.environ.get("HWDET_OUTPUTS", outputs)),
        cache=Path(os.environ.get("HWDET_CACHE", cache)),
    )
    for p in (paths.outputs, paths.runs, paths.cache):
        p.mkdir(parents=True, exist_ok=True)
    return paths


def restore_kaggle_outputs(paths: Paths) -> None:
    """On Kaggle, outputs of a previous notebook version can be attached as an input.
    Copy any runs found there into /kaggle/working so training resumes instead of restarting."""
    if paths.platform != "kaggle":
        return
    for prev_runs in Path("/kaggle/input").glob("*/outputs/runs"):
        for run_dir in prev_runs.iterdir():
            dst = paths.runs / run_dir.name
            if not dst.exists():
                shutil.copytree(run_dir, dst)
                print(f"restored run {run_dir.name} from {prev_runs}")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def setup(seed: int = 42, mount_drive: bool = True) -> Paths:
    """First call in every notebook. Safe to re-run."""
    paths = get_paths(mount_drive=mount_drive)
    restore_kaggle_outputs(paths)
    seed_everything(seed)
    print(paths)
    print(f"  device: {device()}")
    return paths
