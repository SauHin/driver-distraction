"""Make demo videos for app/live.py: the classes c0 to c9 in order, for each driver.
Each video gets a CSV of the true class of each part. The files stay in data/demo/, not in git.
Without --test, the script reads development drivers only. With --test, it reads the test drivers
for the demo of the final model (decision D23), and it opens the test lock one time."""
import argparse
import csv
from pathlib import Path

import cv2

from ddd.live import NAMES, demo_rows, demo_run
from ddd.splits import load_split
from ddd.statefarm import load_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("drivers", nargs="+", help="drivers, for example p014, or 'all'")
    p.add_argument("--test", action="store_true",
                   help="test drivers for the demo of the final model (decision D23). Opens the test lock.")
    p.add_argument("--seconds", type=float, default=3.0, help="seconds of each class")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--split", default="splits/statefarm_v2.json")
    p.add_argument("--data-root", default="data/raw/statefarm")
    p.add_argument("--out", default="data/demo")
    a = p.parse_args()
    split = load_split(a.split)
    rows = load_rows(Path(a.data_root) / "driver_imgs_list.csv", split, "test" if a.test else "dev",
                     unlock_test=a.test, reason="Phase 4 demo videos of the test drivers, model D19, decision D23")
    part = sorted({r["subject"] for r in rows})
    drivers = part if a.drivers == ["all"] else a.drivers
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in drivers:
        if d not in part:
            raise SystemExit(f"{d} is not a {'test' if a.test else 'development'} driver")
        seq = demo_rows(rows, d, int(a.seconds * a.fps))
        video = cv2.VideoWriter(str(out / f"{d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (640, 480))
        schedule, last = [], None
        for n, r in enumerate(seq):
            video.write(cv2.imread(str(Path(a.data_root) / "imgs" / "train" / r["classname"] / r["img"])))
            if r["classname"] != last:
                schedule.append((n / a.fps, r["classname"]))
                last = r["classname"]
        video.release()
        with open(out / f"{d}_truth.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["start_seconds", "class", "name"])
            w.writerows((f"{s:.1f}", c, NAMES[int(c[1:])]) for s, c in schedule)
        print(f"{out / d}.mp4: {len(seq)} frames. Model: {demo_run(split, d)} (it did not train on {d}).")


if __name__ == "__main__":
    main()
