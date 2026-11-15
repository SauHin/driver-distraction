"""Train the runs of a queue file one at a time, each in its own process.
A run folder with metrics.json is complete, so the script skips it.
After a disconnect, start the script again. Each run continues from its last.pt."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from ddd.train import pending_runs


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("queue", help="text file: one config path and one fold on each line")
    p.add_argument("--runs", default="runs", help="parent folder of the run folders")
    p.add_argument("--logs", default="logs", help="folder for one log file of each run")
    a = p.parse_args()
    Path(a.logs).mkdir(parents=True, exist_ok=True)
    for cfg, fold, name in pending_runs(a.queue, a.runs):
        print(_now(), "start", name, flush=True)
        with open(Path(a.logs) / f"{name}.log", "a", encoding="utf-8") as log:
            code = subprocess.call([sys.executable, "scripts/train.py", cfg, "--fold", str(fold),
                                    "--runs", a.runs, "--name", name],
                                   stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if code:
            sys.exit(f"{_now()} {name} stopped with exit code {code}")
    print(_now(), "queue done", flush=True)


if __name__ == "__main__":
    main()
