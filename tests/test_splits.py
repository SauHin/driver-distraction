import csv
import json

import pytest

from conftest import SPLIT_PATH, SPLIT_V1_PATH
from ddd import REPO_ROOT
from ddd.splits import load_split, make_balanced_split, make_split

# The public ViT-DD test list val_list_d_01.csv holds these drivers.
PUBLIC_TEST = ["p002", "p012", "p026", "p035", "p041", "p045", "p050", "p075"]
PUBLIC_LISTS = REPO_ROOT / "data" / "external" / "vitdd" / "annotations" / "SFDDD"
AUDIT = REPO_ROOT / "results" / "data_audit" / "statefarm.json"
SCENES = REPO_ROOT / "results" / "data_audit" / "statefarm_split_check.json"


@pytest.fixture(params=[SPLIT_PATH, SPLIT_V1_PATH], ids=lambda p: p.stem)
def split(request):
    return load_split(request.param)


def test_no_driver_in_two_parts(split):
    assert not set(split["dev"]) & set(split["test"])
    assert len(split["dev"]) + len(split["test"]) == 26
    for fold in split["folds"]:
        assert not set(fold["train"]) & set(fold["val"])
        assert not set(fold["train"]) & set(split["test"])
        assert not set(fold["val"]) & set(split["test"])


def test_test_part_equals_public_list(split):
    assert split["test"] == PUBLIC_TEST


@pytest.mark.skipif(not PUBLIC_LISTS.is_dir(), reason="public ViT-DD lists not in data/external")
def test_parts_equal_public_csv_files(split):
    def subjects(name):
        with open(PUBLIC_LISTS / name, newline="", encoding="utf-8") as f:
            return sorted({r["subject"] for r in csv.DictReader(f)})
    assert split["test"] == subjects("val_list_d_01.csv")
    assert split["dev"] == subjects("train_list_d_01.csv")


def test_folds_divide_dev_part(split):
    vals = [fold["val"] for fold in split["folds"]]
    assert len(vals) == 3
    assert all(len(v) == 6 for v in vals)
    assert sorted(d for v in vals for d in v) == split["dev"]
    for fold in split["folds"]:
        assert sorted(fold["train"] + fold["val"]) == split["dev"]


def test_v1_file_matches_seeded_shuffle():
    split = load_split(SPLIT_V1_PATH)
    again = make_split(split["dev"], split["test"], seed=2026)
    assert split["seed"] == 2026
    assert split["folds"] == again["folds"]


def test_v2_file_matches_balanced_search():
    split = load_split(SPLIT_PATH)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    images = {d: sum(c.values()) for d, c in audit["images_per_driver_and_class"].items()}
    scenes = json.loads(SCENES.read_text(encoding="utf-8"))["scene_groups"]["groups"]
    groups = [[d for d in g if d in split["dev"]] for g in scenes]
    assert split["folds"] == make_balanced_split(groups, images, split["test"])["folds"]
    for fold in split["folds"]:
        assert [len(set(fold["val"]) & set(g)) for g in groups] == [2, 4]


def test_balanced_split_takes_smallest_range():
    images = {"a": 10, "b": 1, "c": 5, "d": 5, "e": 1, "f": 9}
    got = make_balanced_split([["a", "b"], ["c", "d", "e", "f"]], images, ["t"], n_folds=2)
    # Fold 0 holds "a". With "c" and "e" or "d" and "e", the counts are 16 and 15.
    # The tie goes to the first in sorted order.
    assert [f["val"] for f in got["folds"]] == [["a", "c", "e"], ["b", "d", "f"]]


def test_make_split_refuses_overlap():
    with pytest.raises(ValueError):
        make_split(["p001", "p002", "p003"], ["p003"], n_folds=3)
