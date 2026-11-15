"""Score the last epoch of a run on the test drivers. This opens the test lock.
See docs/PLAN.md sections 4, 5, and 7, and decision P9."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ddd import REPO_ROOT
from ddd.data import CODES, StateFarmImages, eval_transform
from ddd.metrics import binary_scores, scores
from ddd.models import create
from ddd.splits import TEST_ACCESS_LOG, load_split
from ddd.statefarm import load_rows
from ddd.train import environment, evaluate

OUT_DIR = REPO_ROOT / "results" / "replication"


def _all_scores(m):
    return {**scores(m), **binary_scores(m)}


def evaluate_run(run_dir, reason, out_dir=OUT_DIR, log_path=TEST_ACCESS_LOG):
    """Write `<run>_test_metrics.json`, `<run>_test_confusion.csv`, and `<run>_test_drivers.csv`."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    # Decision P9: only the last epoch. Check before the unlock, so a refusal writes no log row.
    if ck["epoch"] != cfg["epochs"] - 1:
        raise ValueError(f"last.pt holds epoch {ck['epoch']}, not the last epoch {cfg['epochs'] - 1}")
    split = load_split(cfg["split"])
    rows = load_rows(Path(cfg["data_root"]) / "driver_imgs_list.csv", split, "test",
                     unlock_test=True, reason=reason, log_path=log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create({**cfg, "weights": None})  # the checkpoint holds all weights
    model.load_state_dict(ck["model"])
    model.to(device)
    tf = eval_transform(cfg.get("input_size", 224))
    per_driver = {}
    for d in sorted({r["subject"] for r in rows}):
        ds = StateFarmImages([r for r in rows if r["subject"] == d], cfg["data_root"], tf)
        dl = DataLoader(ds, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                        pin_memory=True)
        per_driver[d] = evaluate(model, dl, device)
    total = np.sum(list(per_driver.values()), axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    name = run_dir.name
    metrics = {"run": name, "epoch": ck["epoch"], "reason": reason,
               "test_drivers": sorted(per_driver), "images": int(total.sum()),
               **_all_scores(total), "environment": environment()}
    (out_dir / f"{name}_test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n",
                                                       encoding="utf-8")
    with open(out_dir / f"{name}_test_confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true/pred"] + CODES)
        w.writerows([c] + row.tolist() for c, row in zip(CODES, total))
    with open(out_dir / f"{name}_test_drivers.csv", "w", newline="", encoding="utf-8") as f:
        w = None
        for d, m in per_driver.items():
            row = {"driver": d, "images": int(m.sum()), **_all_scores(m)}
            if w is None:
                w = csv.DictWriter(f, list(row))
                w.writeheader()
            w.writerow(row)
    return metrics
