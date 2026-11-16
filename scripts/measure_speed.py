"""Measure the processed frames per second of each config on this computer, on the GPU and the CPU.
Writes one JSON file. See ddd/speed.py for the meaning of a frame."""
import argparse
import json
import platform
from pathlib import Path

import torch

from ddd.speed import measure
from ddd.train import environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("configs", nargs="+")
    p.add_argument("--out", default="results/development/speed_laptop.json")
    p.add_argument("--frames", type=int, default=50)
    a = p.parse_args()
    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    rows = []
    for path in a.configs:
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        for d in devices:
            row = {"config": cfg["name"], "model": cfg["model"], **measure(cfg, torch.device(d), a.frames)}
            print(json.dumps(row), flush=True)
            rows.append(row)
    out = {"rows": rows, "cpu": platform.processor(), "torch_threads": torch.get_num_threads(),
           "environment": environment()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
