"""Write the StateFarm data audit to a JSON file."""
import argparse
import json
from pathlib import Path

from ddd.audit import audit_statefarm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/raw/statefarm",
                   help="folder with driver_imgs_list.csv and imgs/train/")
    p.add_argument("--images", default=None, help="class folders; default ROOT/imgs/train")
    p.add_argument("--out", default="results/data_audit/statefarm.json")
    a = p.parse_args()
    root = Path(a.root)
    report = audit_statefarm(root / "driver_imgs_list.csv", a.images or root / "imgs" / "train")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{out}: {report['rows']} rows, {report['drivers']} drivers, "
          f"{len(report['missing_files'])} missing, {report['exact_copies']['groups']} copy groups")


if __name__ == "__main__":
    main()
