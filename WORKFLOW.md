# HW2 workflow

`README.md`, `index.html`, `assets/` and `.github/` belong to the course template. **Do not edit them.**
The report is `report.md` plus `images/`.

## Layout

```
notebooks/      one notebook per task, added as you go (copy _template.ipynb)
configs/        experiment configs (YAML), added as you go
src/hwdet/      helpers shared by all notebooks
  env.py          detects Mac / Colab / Kaggle and sets data + output folders
  config.py       YAML configs with `base:` inheritance
  tracker.py      runs: stages done, per-epoch history, metrics, experiments.csv (crash-safe)
  data/           competition file reading + data download
  eval/           COCO metrics (mAP, AP50, AP_small, per class), inference speed
  submission.py   Kaggle CSV writer + validator
  viz.py          boxes on frames, training curves, save_fig()
scripts/        get_data, status, make_submission, collect_results
results/        small files committed to git: experiments.csv, runs/<id>/ metrics & history
submissions/    submitted CSVs
data/, outputs/ local only, git-ignored
```

## Where things run

- **Mac:** EDA, summary, report. `python scripts/get_data.py --labels-only` downloads only the annotations.
- **Colab / Kaggle:** training. Push to GitHub. The setup cell clones the repo and installs `requirements.txt`.
  - Colab: data is downloaded once to Google Drive (`MyDrive/ucu-hw2/`), and outputs are saved on Drive.
    Put `KAGGLE_API_TOKEN` in Colab Secrets.
  - Kaggle: create the notebook from the competition page, with Internet on. Outputs go to `/kaggle/working/outputs`.
    To keep them, *Save Version*, then attach that version's output to the next session.

## After a crash

Each run lives in `outputs/runs/<run_id>/` with `stages.json`, `history.csv`, `metrics.json` and `checkpoints/`.
Guard each step with `if not run.is_done("train"): ...; run.mark_done("train")`, and save
checkpoints to `run.ckpt("last.pt")`. Re-running then skips finished work. `status(P)` shows what is done.

## Results into the repo

```bash
python scripts/collect_results.py --src "<Drive or Kaggle outputs folder>"
```

Copies metrics, histories, configs and figures (not checkpoints) into `results/` and `images/`.

## Prediction format

`file_name, cls, score, x, y, w, h`: pixels, top-left corner, cls 0 to 7. Used by `eval/` and `submission.py`.
