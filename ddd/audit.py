"""Audit of the StateFarm label file and images. See docs/PLAN.md section 3.5."""
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from ddd.manifest import sha256_file
from ddd.statefarm import read_driver_list


def audit_statefarm(csv_path, images_dir):
    """Counts, missing files, image sizes, and exact copies.

    `images_dir` holds one folder for each class, as in the Kaggle `imgs/train` folder.
    """
    rows = read_driver_list(csv_path)
    images_dir = Path(images_dir)

    per_driver = defaultdict(Counter)
    for r in rows:
        per_driver[r["subject"]][r["classname"]] += 1
    row_keys = Counter((r["classname"], r["img"]) for r in rows)

    missing, unreadable = [], []
    sizes, modes = Counter(), Counter()
    by_hash = defaultdict(list)
    for r in rows:
        path = images_dir / r["classname"] / r["img"]
        if not path.is_file():
            missing.append(f"{r['classname']}/{r['img']}")
            continue
        try:
            with Image.open(path) as im:
                sizes[f"{im.width}x{im.height}"] += 1
                modes[im.mode] += 1
        except OSError:
            unreadable.append(f"{r['classname']}/{r['img']}")
        by_hash[sha256_file(path)].append(r)

    listed = {f"{c}/{i}" for c, i in row_keys}
    on_disk = {p.relative_to(images_dir).as_posix()
               for p in images_dir.glob("*/*") if p.is_file()}

    copies = []
    for h, group in sorted(by_hash.items()):
        if len(group) > 1:
            copies.append({
                "sha256": h,
                "images": [f"{r['subject']}/{r['classname']}/{r['img']}" for r in group],
                "drivers": sorted({r["subject"] for r in group}),
                "classes": sorted({r["classname"] for r in group}),
            })

    return {
        "label_file": Path(csv_path).name,
        "rows": len(rows),
        "drivers": len(per_driver),
        "classes": sorted({r["classname"] for r in rows}),
        "images_per_class": dict(sorted(Counter(r["classname"] for r in rows).items())),
        "images_per_driver": {d: sum(c.values()) for d, c in sorted(per_driver.items())},
        "images_per_driver_and_class": {d: dict(sorted(c.items()))
                                        for d, c in sorted(per_driver.items())},
        "duplicate_rows": sorted(f"{c}/{i}" for (c, i), n in row_keys.items() if n > 1),
        "missing_files": sorted(missing),
        "unreadable_files": sorted(unreadable),
        "files_not_in_label_file": sorted(on_disk - listed),
        "image_sizes": dict(sizes.most_common()),
        "image_modes": dict(modes.most_common()),
        "exact_copies": {
            "groups": len(copies),
            "groups_across_drivers": sum(len(g["drivers"]) > 1 for g in copies),
            "groups_across_classes": sum(len(g["classes"]) > 1 for g in copies),
            "list": copies,
        },
    }
