"""Shared fixtures. The StateFarm fixture is tiny and synthetic. It holds no dataset image."""
import csv
from pathlib import Path

import pytest
from PIL import Image

from ddd import REPO_ROOT

SPLIT_PATH = REPO_ROOT / "splits" / "statefarm_v2.json"
SPLIT_V1_PATH = REPO_ROOT / "splits" / "statefarm_v1.json"


def make_statefarm_fixture(root):
    """A fake StateFarm folder with known defects.

    Defects: img_3 is missing, img_4 is an exact copy of img_1 under another driver,
    img_5 has another size, and extra.jpg is not in the label file.
    """
    root = Path(root)
    rows = [("p001", "c0", "img_1.jpg", (10, 20, 30)),
            ("p001", "c1", "img_2.jpg", (40, 50, 60)),
            ("p002", "c1", "img_3.jpg", None),
            ("p002", "c0", "img_4.jpg", (10, 20, 30)),
            ("p002", "c2", "img_5.jpg", (70, 80, 90))]
    for subject, cls, img, color in rows:
        if color is None:
            continue
        size = (8, 6) if img != "img_5.jpg" else (6, 8)
        folder = root / "imgs" / "train" / cls
        folder.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(folder / img, format="PNG")
    Image.new("RGB", (8, 6), (1, 2, 3)).save(root / "imgs" / "train" / "c2" / "extra.jpg",
                                              format="PNG")
    with open(root / "driver_imgs_list.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "classname", "img"])
        w.writerows(r[:3] for r in rows)
    return root


@pytest.fixture
def statefarm_root(tmp_path):
    return make_statefarm_fixture(tmp_path / "statefarm")
