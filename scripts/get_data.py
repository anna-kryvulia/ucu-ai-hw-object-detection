"""Download competition data for this machine.

    python scripts/get_data.py --labels-only   # local Mac: annotations only (few MB), enough for EDA
    python scripts/get_data.py                 # full dataset (Colab / a machine with disk space)
"""
import argparse

from hwdet.data.download import ensure_data
from hwdet.env import get_paths

ap = argparse.ArgumentParser()
ap.add_argument("--labels-only", action="store_true")
args = ap.parse_args()
print(ensure_data(get_paths(mount_drive=False), labels_only=args.labels_only))
