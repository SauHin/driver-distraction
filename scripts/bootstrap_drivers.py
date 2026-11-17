"""A bootstrap interval over the test drivers of the pooled accuracy (docs/PLAN.md section 5).
Reads saved driver tables only. It reads no test image, so it does not open the test lock."""
import argparse
import csv
import json
from pathlib import Path

from ddd.metrics import bootstrap_accuracy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tables", nargs="+", help="files <run>_test_drivers.csv")
    p.add_argument("--reps", type=int, default=10000)
    p.add_argument("--out", default="results/final/bootstrap.json")
    a = p.parse_args()
    out = {"reps": a.reps, "seed": 0, "level": 0.95, "runs": []}
    for path in a.tables:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        acc, n = [float(r["accuracy"]) for r in rows], [int(r["images"]) for r in rows]
        lo, hi = bootstrap_accuracy(acc, n, a.reps)
        pooled = sum(x * k for x, k in zip(acc, n)) / sum(n)
        out["runs"].append({"table": Path(path).as_posix(), "drivers": len(rows), "accuracy": pooled,
                            "interval_low": lo, "interval_high": hi})
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
