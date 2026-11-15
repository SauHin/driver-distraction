"""Write a StateFarm split file from the public ViT-DD split lists.

v1: a shuffle with seed 2026. v2: folds balanced by scene group and by validation image count.
"""
import argparse
import csv
import json
from pathlib import Path

from ddd.splits import make_balanced_split, make_split

VITDD = "https://github.com/PurdueDigitalTwin/ViT-DD"


def subjects(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return {r["subject"] for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", choices=["v1", "v2"], default="v2")
    p.add_argument("--lists", default="data/external/vitdd/annotations/SFDDD",
                   help="folder with train_list_d_01.csv and val_list_d_01.csv")
    p.add_argument("--audit", default="results/data_audit/statefarm.json",
                   help="v2: image counts of each driver")
    p.add_argument("--scenes", default="results/data_audit/statefarm_split_check.json",
                   help="v2: scene groups of the drivers")
    p.add_argument("--seed", type=int, default=2026, help="v1 only")
    a = p.parse_args()
    lists = Path(a.lists)
    dev = subjects(lists / "train_list_d_01.csv")
    test = subjects(lists / "val_list_d_01.csv")
    source = (f"{VITDD}, annotations/SFDDD/train_list_d_01.csv (dev) "
              "and val_list_d_01.csv (test)")
    if a.version == "v1":
        body = make_split(dev, test, seed=a.seed)
    else:
        audit = json.loads(Path(a.audit).read_text(encoding="utf-8"))
        images = {d: sum(c.values()) for d, c in audit["images_per_driver_and_class"].items()}
        scenes = json.loads(Path(a.scenes).read_text(encoding="utf-8"))["scene_groups"]["groups"]
        groups = [[d for d in g if d in dev] for g in scenes]
        body = {"method": "Each fold gets an equal share of the development drivers of each scene "
                          f"group. Of these assignments, it takes the one with the smallest range of "
                          f"validation image counts. Scene groups: {a.scenes}. Image counts: {a.audit}.",
                **make_balanced_split(groups, images, test)}
    split = {"name": f"statefarm_{a.version}", "source": source, **body}
    out = Path(f"splits/statefarm_{a.version}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"{out}: {len(split['dev'])} dev drivers, {len(split['test'])} test drivers")


if __name__ == "__main__":
    main()
