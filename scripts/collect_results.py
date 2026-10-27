"""Copy the small, report-relevant files of every run into the repo (results/), plus figures into images/.

Training happens on Colab/Kaggle and writes to Drive / Kaggle output. Point this script at that
outputs folder (Google Drive for desktop mounts it locally, or download the Kaggle output) and
commit what it copies. Checkpoints and prediction CSVs are NOT copied (too big for git).

    python scripts/collect_results.py --src "~/Google Drive/My Drive/ucu-hw2/outputs"
    python scripts/collect_results.py --src ~/Downloads/kaggle-output/outputs
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

from hwdet.env import REPO_ROOT
from hwdet.tracker import REGISTRY, SMALL_FILES

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="an outputs/ folder containing runs/ and experiments.csv")
ap.add_argument("--no-figures", action="store_true")
args = ap.parse_args()

src = Path(args.src).expanduser()
dst = REPO_ROOT / "results"
for run_dir in sorted((src / "runs").iterdir()):
    if not run_dir.is_dir():
        continue
    out = dst / "runs" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    for name in SMALL_FILES:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, out / name)
    for name in ("results.csv", "results.png", "confusion_matrix_normalized.png"):   # Ultralytics extras
        if (run_dir / "ultra" / name).exists():
            shutil.copy2(run_dir / "ultra" / name, out / name)
    print("collected", run_dir.name)

if (src / REGISTRY).exists():                     # merge registries (newest row per run/eval wins)
    frames = [pd.read_csv(src / REGISTRY)]
    if (dst / REGISTRY).exists():
        frames.insert(0, pd.read_csv(dst / REGISTRY))
    reg = pd.concat(frames, ignore_index=True).sort_values("updated")
    reg.drop_duplicates(["run_id", "eval"], keep="last").to_csv(dst / REGISTRY, index=False)
    print("updated", dst / REGISTRY)

if not args.no_figures and (src / "figures").exists():
    for f in (src / "figures").glob("*.png"):
        shutil.copy2(f, REPO_ROOT / "images" / f.name)
    print("copied figures into images/")
