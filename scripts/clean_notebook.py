"""Remove progress-bar widgets from executed notebooks.  python scripts/clean_notebook.py NOTEBOOK [NOTEBOOK ...]

Kaggle saves every tqdm bar as a Jupyter widget, but not the widget state, so VS Code / GitHub show
"Could not render content for 'application/vnd.jupyter.widget-view+json'" in their place.
All other outputs (text, tables, figures) are kept.
"""
import argparse
import json

WIDGET = "application/vnd.jupyter.widget-view+json"

ap = argparse.ArgumentParser()
ap.add_argument("notebooks", nargs="+")
args = ap.parse_args()

for path in args.notebooks:
    with open(path) as f:
        nb = json.load(f)
    removed = 0
    for cell in nb["cells"]:
        outs = cell.get("outputs", [])
        keep = [o for o in outs if WIDGET not in o.get("data", {})]
        removed += len(outs) - len(keep)
        if outs:
            cell["outputs"] = keep
    nb["metadata"].pop("widgets", None)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"{path}: removed {removed} widget outputs")
