"""Get the competition data onto whatever machine we are on.

* Kaggle notebook: nothing to do, the data is mounted read-only at /kaggle/input/<competition>.
* Colab: the first time, download the zip with the Kaggle API and keep a copy on Google Drive.
  Afterwards unzip it from Drive to the fast local disk (/content/data) at the start of each session.
  Copying one zip is much faster than reading ~30k small files from Drive.
* Local Mac: `labels_only=True` downloads only the annotation files (a few MB) so EDA runs locally.

Kaggle credentials, any of:
  * new API token: ~/.kaggle/access_token file, or the KAGGLE_API_TOKEN environment variable
  * legacy key:    ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY environment variables
On Colab, store KAGGLE_API_TOKEN (or the legacy pair) in the "Secrets" panel (key icon).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from ..env import COMPETITION, DRIVE_ROOT, Paths

LABEL_FILES = ["train/annotations_coco.json", "val/annotations_coco.json", "train/data.yaml",
               "test_images.csv"]


def _load_colab_secrets() -> None:
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        return
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        if k in os.environ:
            continue
        try:
            os.environ[k] = userdata.get(k)
        except Exception:          # secret not defined or notebook access not granted
            pass


def _kaggle(*args: str) -> None:
    subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, *args], check=True)


def _unzip(zip_path: Path, dst: Path) -> None:
    print(f"unzipping {zip_path} -> {dst}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dst)


def download_frames(paths: Paths, file_names: list[str], with_labels: bool = True) -> None:
    """Fetch a few individual frames (and their label / ignore files) for local visual checks.
    file_names are relative to the data root, e.g. 'val/images/<clip>/0000001.jpg'."""
    _load_colab_secrets()
    for fn in file_names:
        wanted = [fn]
        if with_labels and "/images/" in fn:
            stem = fn.rsplit(".", 1)[0]
            wanted += [stem.replace("/images/", "/labels/", 1) + ".txt", stem.replace("/images/", "/ignore/", 1) + ".txt"]
        for f in wanted:
            if (paths.data / f).exists():
                continue
            r = subprocess.run(["kaggle", "competitions", "download", "-c", COMPETITION, "-f", f, "-p", str(paths.data),
                                "-q"], capture_output=True, text=True)
            if r.returncode != 0 and "/ignore/" not in f:   # ignore files exist only for some frames
                print(f"could not download {f}: {r.stdout.strip()[-200:]} {r.stderr.strip()[-200:]}")


def is_ready(data_root: Path, labels_only: bool = False) -> bool:
    need = ["val/annotations_coco.json", "test_images.csv"]
    if not labels_only:
        need.append("train/images")
    return all((data_root / n).exists() for n in need)


def ensure_data(paths: Paths, labels_only: bool = False) -> Path:
    data = paths.data
    if paths.platform == "kaggle" or is_ready(data, labels_only):
        print(f"data ready at {data}")
        return data
    data.mkdir(parents=True, exist_ok=True)
    _load_colab_secrets()

    if labels_only:
        for f in LABEL_FILES:
            target = data / f
            if target.exists():
                continue
            _kaggle("-f", f, "-p", str(data))    # the CLI keeps the file's folder path under -p
            z = target.parent / (Path(f).name + ".zip")
            if z.exists():                       # the API zips large single files
                _unzip(z, target.parent)
                z.unlink()
        return data

    zip_name = f"{COMPETITION}.zip"
    cached = DRIVE_ROOT / "data" / zip_name if paths.platform == "colab" else data / zip_name
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        _kaggle("-p", str(cached.parent))
    local_zip = cached
    if paths.platform == "colab":                 # copy once to local disk, unzip from there
        local_zip = paths.cache / zip_name
        if not local_zip.exists():
            shutil.copy(cached, local_zip)
    _unzip(local_zip, data)
    if paths.platform == "colab":
        local_zip.unlink()
    return data
