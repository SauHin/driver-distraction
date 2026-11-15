"""StateFarm label file and the rows of one split part."""
import csv

from ddd.splits import TEST_ACCESS_LOG, drivers


def read_driver_list(csv_path):
    """Rows of `driver_imgs_list.csv` as dicts with keys subject, classname, img."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_rows(csv_path, split, part, fold=None, unlock_test=False, reason=None,
              log_path=TEST_ACCESS_LOG):
    """The label rows of the drivers of one part. The test part obeys the test lock."""
    keep = set(drivers(split, part, fold, unlock_test, reason, log_path))
    return [r for r in read_driver_list(csv_path) if r["subject"] in keep]
