import csv

import pytest

from conftest import SPLIT_PATH
from ddd.splits import drivers, load_split
from ddd.statefarm import load_rows


@pytest.fixture
def split():
    return load_split(SPLIT_PATH)


def test_test_part_refused_without_unlock(split, tmp_path):
    log = tmp_path / "test_access_log.csv"
    with pytest.raises(PermissionError):
        drivers(split, "test", log_path=log)
    assert not log.exists()


def test_unlock_needs_reason(split, tmp_path):
    log = tmp_path / "test_access_log.csv"
    with pytest.raises(ValueError):
        drivers(split, "test", unlock_test=True, log_path=log)
    assert not log.exists()


def test_each_unlock_adds_a_row(split, tmp_path):
    log = tmp_path / "test_access_log.csv"
    for n in (1, 2):
        got = drivers(split, "test", unlock_test=True, reason=f"unit test {n}", log_path=log)
        assert got == split["test"]
        with open(log, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n
        assert rows[-1]["reason"] == f"unit test {n}"
        assert rows[-1]["split"] == split["name"]


def test_loader_obeys_lock(statefarm_root, tmp_path):
    split = {"name": "fixture", "dev": ["p001"], "test": ["p002"],
             "folds": [{"train": [], "val": ["p001"]}]}
    csv_path = statefarm_root / "driver_imgs_list.csv"
    log = tmp_path / "log.csv"
    with pytest.raises(PermissionError):
        load_rows(csv_path, split, "test", log_path=log)
    assert {r["subject"] for r in load_rows(csv_path, split, "val", fold=0)} == {"p001"}
    rows = load_rows(csv_path, split, "test", unlock_test=True, reason="unit test", log_path=log)
    assert {r["subject"] for r in rows} == {"p002"}
