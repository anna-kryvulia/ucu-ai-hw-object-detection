"""Build (and validate) a Kaggle submission from test predictions in the common format.

    python scripts/make_submission.py outputs/runs/<run_id>/predictions/test.csv --out submissions/<name>.csv
"""
import argparse

import pandas as pd

from hwdet.data.io import frame_table
from hwdet.env import get_paths
from hwdet.submission import make_submission

ap = argparse.ArgumentParser()
ap.add_argument("preds")
ap.add_argument("--out", required=True)
ap.add_argument("--score-thr", type=float, default=0.0)
args = ap.parse_args()

paths = get_paths(mount_drive=False)
make_submission(pd.read_csv(args.preds), frame_table(paths.data, "test"), args.out, score_thr=args.score_thr)
