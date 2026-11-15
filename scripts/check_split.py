"""Check a split for fold balance and for near copies of images across drivers."""
import argparse
import json
from pathlib import Path

import numpy as np

from PIL import Image

from ddd.split_check import (dhash, fold_stats, histogram, nearest_other_group,
                              nearest_same_group, scene_groups)
from ddd.splits import load_split
from ddd.statefarm import read_driver_list


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/raw/statefarm")
    p.add_argument("--split", default="splits/statefarm_v1.json")
    p.add_argument("--audit", default="results/data_audit/statefarm.json")
    p.add_argument("--out", default="results/data_audit/statefarm_split_check.json")
    p.add_argument("--pairs", default="data/derived/statefarm_closest_pairs.json",
                   help="closest pairs with image names; stays out of git")
    a = p.parse_args()
    root = Path(a.root)
    split = load_split(a.split)
    audit = json.loads(Path(a.audit).read_text(encoding="utf-8"))
    rows = read_driver_list(root / "driver_imgs_list.csv")
    # The part of each driver. The fold numbers mark the validation fold of a dev driver.
    part = {d: "test" for d in split["test"]}
    for k, fold in enumerate(split["folds"]):
        part.update({d: f"fold{k}" for d in fold["val"]})

    hashes, thumb_sum, thumb_n = [], {}, {}
    for r in rows:
        with Image.open(root / "imgs" / "train" / r["classname"] / r["img"]) as im:
            gray = im.convert("L")
        hashes.append(dhash(gray))
        d = r["subject"]
        thumb_sum[d] = thumb_sum.get(d, 0) + np.asarray(gray.resize((32, 24)), dtype=np.float64)
        thumb_n[d] = thumb_n.get(d, 0) + 1
    hashes = np.stack(hashes)
    scenes = scene_groups({d: (thumb_sum[d] / thumb_n[d]).ravel() for d in thumb_sum}, k=2)
    scene_of = {d: n for n, g in enumerate(scenes) for d in g}
    parts_of = {"test": split["test"], **{f"fold{k}_val": f["val"] for k, f in enumerate(split["folds"])},
                **{f"fold{k}_train": f["train"] for k, f in enumerate(split["folds"])}}
    drivers = [r["subject"] for r in rows]
    parts = [part[d] for d in drivers]
    cross_driver, arg = nearest_other_group(hashes, drivers)
    cross_part, _ = nearest_other_group(hashes, parts)
    same_driver = nearest_same_group(hashes, drivers)

    report = {
        "split": split["name"],
        "method": "64-bit dHash of a 9x8 gray thumbnail. Distance: Hamming, 0 to 64.",
        "folds": fold_stats(audit, split),
        "scene_groups": {"method": "Average linkage on the mean 32x24 gray image of each driver, 2 groups.",
                         "groups": scenes,
                         "drivers_per_group": {p: [sum(scene_of[d] == n for d in ds)
                                                   for n in range(len(scenes))]
                                               for p, ds in parts_of.items()}},
        "nearest_distance_same_driver": histogram(same_driver),
        "nearest_distance_other_driver": histogram(cross_driver),
        "nearest_distance_other_part": histogram(cross_part),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    order = np.argsort(cross_driver)[:40]
    pairs = [{"distance": int(cross_driver[i]),
              "a": f"{rows[i]['subject']}/{rows[i]['classname']}/{rows[i]['img']}",
              "b": f"{rows[arg[i]]['subject']}/{rows[arg[i]]['classname']}/{rows[arg[i]]['img']}"}
             for i in order]
    pairs_path = Path(a.pairs)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_path.write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")
    print(f"{out}; closest pairs in {pairs_path}")


if __name__ == "__main__":
    main()
