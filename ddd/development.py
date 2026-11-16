"""Scores of the development folds of one config. See docs/PLAN.md section 8 and decision P12."""
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from ddd import REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "development"
SCORES = ["val_loss", "val_accuracy", "val_class_mean_accuracy", "val_macro_f1"]
RUN_FILES = ["config.json", "environment.json", "history.csv", "metrics.json"]


def summarize(runs_dir, name, n_folds=3, out_dir=OUT_DIR):
    """Copy the run files of `<name>_f0` to `<name>_f<n-1>` into `out_dir`, and write
    `<name>_folds.json` with the scores of each fold and their mean."""
    runs_dir, out_dir = Path(runs_dir), Path(out_dir)
    folds, base = [], None
    for k in range(n_folds):
        run = runs_dir / f"{name}_f{k}"
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        # Decision P12: the last epoch. The folds differ in the fold only.
        rest = {key: v for key, v in cfg.items() if key != "fold"}
        if cfg["fold"] != k or m["epoch"] != cfg["epochs"] - 1 or base not in (None, rest):
            raise ValueError(f"{run} is not the complete run of fold {k} of {name}")
        base = rest
        with open(run / "history.csv", newline="", encoding="utf-8") as f:
            seconds = sum(float(r["epoch_seconds"]) for r in csv.DictReader(f))
        folds.append({"run": run.name, "fold": k, **{s: m[s] for s in SCORES},
                      "sum_epoch_seconds": round(seconds, 2)})
        (out_dir / run.name).mkdir(parents=True, exist_ok=True)
        for f in RUN_FILES:
            shutil.copyfile(run / f, out_dir / run.name / f)
    summary = {"config": name, "folds": folds,
               "mean": {s: float(np.mean([f[s] for f in folds])) for s in SCORES}}
    (out_dir / f"{name}_folds.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                encoding="utf-8")
    return summary


def gray_scores(runs_dir, name, n_folds=3, out_dir=OUT_DIR):
    """Score the last checkpoint of each fold run on its validation drivers, in color and in gray.
    Write `<name>_gray.json` with the scores of the file `metrics.json` too, and the tables of the
    error analysis: the confusion matrix of all folds in each mode, `<name>_val_classes.csv`, and
    `<name>_val_drivers.csv`. No test driver is read."""
    # torch only here, so the summary above runs with no torch.
    import torch
    from torch.utils.data import DataLoader

    from ddd.data import CODES, StateFarmImages, eval_transform
    from ddd.metrics import class_scores, scores
    from ddd.models import create
    from ddd.train import _parts, environment, evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds, drivers = [], []
    total = {mode: np.zeros((10, 10), dtype=np.int64) for mode in ("color", "gray")}
    for k in range(n_folds):
        run = Path(runs_dir) / f"{name}_f{k}"
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        ck = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        if cfg["fold"] != k or ck["epoch"] != cfg["epochs"] - 1:
            raise ValueError(f"{run} is not the complete run of fold {k} of {name}")
        model = create({**cfg, "weights": None})  # the checkpoint holds all weights
        model.load_state_dict(ck["model"])
        model.to(device)
        _, val_rows = _parts(cfg)
        loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        fold = {"run": run.name, "fold": k, "images": len(val_rows), "file": {s: m[s] for s in SCORES}}
        rows_of = {d: [r for r in val_rows if r["subject"] == d] for d in sorted({r["subject"] for r in val_rows})}
        per_driver = {d: {"fold": k, "driver": d, "images": len(rows)} for d, rows in rows_of.items()}
        for mode in ("color", "gray"):
            tf = eval_transform(cfg.get("input_size", 224), gray=mode == "gray")
            c_fold, loss_sum = np.zeros((10, 10), dtype=np.int64), 0.0
            for d, rows in rows_of.items():
                dl = DataLoader(StateFarmImages(rows, cfg["data_root"], tf), batch_size=cfg["batch_size"],
                                num_workers=cfg["num_workers"], pin_memory=True)
                c, loss = evaluate(model, dl, device, loss_fn)
                c_fold += c
                loss_sum += loss * len(rows)
                per_driver[d].update({f"{mode}_{key}": v for key, v in scores(c).items()})
            total[mode] += c_fold
            fold[mode] = {"val_loss": loss_sum / len(val_rows),
                          **{f"val_{key}": v for key, v in scores(c_fold).items()}}
        folds.append(fold)
        drivers += per_driver.values()
    out = {"config": name, "folds": folds,
           "mean": {mode: {s: float(np.mean([f[mode][s] for f in folds])) for s in SCORES}
                    for mode in ("file", "color", "gray")},
           "environment": environment()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_gray.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    for mode, c in total.items():
        with open(out_dir / f"{name}_val_confusion_{mode}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true/pred"] + CODES)
            w.writerows([code] + row.tolist() for code, row in zip(CODES, c))
    per_class = {mode: class_scores(c) for mode, c in total.items()}
    with open(out_dir / f"{name}_val_classes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "images"] + [f"{mode}_{s}" for mode in total for s in ("precision", "recall", "f1")])
        for i, code in enumerate(CODES):
            w.writerow([code, int(total["color"][i].sum())]
                       + [per_class[mode][i][s] for mode in total for s in ("precision", "recall", "f1")])
    with open(out_dir / f"{name}_val_drivers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(drivers[0]))
        w.writeheader()
        w.writerows(drivers)
    return out
