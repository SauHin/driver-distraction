"""Copy the speed of a measurement run into results/compute/first_run.json under one key."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--key", required=True, help="for example laptop or colab")
    p.add_argument("--note", default="")
    p.add_argument("--out", default="results/compute/first_run.json")
    a = p.parse_args()
    run_dir = Path(a.run_dir)
    with open(run_dir / "history.csv", newline="", encoding="utf-8") as f:
        history = list(csv.DictReader(f))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    out = Path(a.out)
    data = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    data[a.key] = {"run": run_dir.name, "config_name": cfg["name"], "batch_size": cfg["batch_size"],
                   "environment": json.loads((run_dir / "environment.json").read_text(encoding="utf-8")),
                   "epochs": history, "note": a.note}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"{out}: {a.key}")


if __name__ == "__main__":
    main()
