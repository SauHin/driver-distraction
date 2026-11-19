"""Split by driver and the test lock. See docs/PLAN.md section 4."""
import csv
import json
import random
import subprocess
from datetime import datetime, timezone
from itertools import combinations, product

from ddd import REPO_ROOT

TEST_ACCESS_LOG = REPO_ROOT / "results" / "test_access_log.csv"


def make_split(dev_drivers, test_drivers, seed=2026, n_folds=3):
    """Shuffle the sorted development drivers with `seed` and cut them into equal folds."""
    dev = sorted(set(dev_drivers))
    test = sorted(set(test_drivers))
    if set(dev) & set(test):
        raise ValueError(f"drivers in two parts: {sorted(set(dev) & set(test))}")
    if len(dev) % n_folds:
        raise ValueError(f"{len(dev)} development drivers do not divide into {n_folds} folds")
    order = list(dev)
    random.Random(seed).shuffle(order)
    size = len(dev) // n_folds
    folds = []
    for k in range(n_folds):
        val = sorted(order[k * size:(k + 1) * size])
        folds.append({"train": sorted(set(dev) - set(val)), "val": val})
    return {"seed": seed, "test": test, "dev": dev, "folds": folds}


def _partitions(items, n):
    """Each way to cut `items` into n labeled chunks of equal size, in order."""
    if n == 1:
        yield (tuple(items),)
        return
    for chunk in combinations(items, len(items) // n):
        rest = [x for x in items if x not in chunk]
        for tail in _partitions(rest, n - 1):
            yield (chunk,) + tail


def make_balanced_split(groups, images, test_drivers, n_folds=3):
    """Folds with an equal share of each scene group and the most equal validation image counts.

    `groups` holds the development drivers of each scene group. `images` maps each driver
    to its image count. The search tries each assignment, so the result needs no seed.
    Ties go to the first assignment in sorted order.
    """
    groups = [sorted(g) for g in groups]
    dev = sorted(d for g in groups for d in g)
    test = sorted(set(test_drivers))
    if len(set(dev)) != len(dev):
        raise ValueError("a driver is in two scene groups")
    if set(dev) & set(test):
        raise ValueError(f"drivers in two parts: {sorted(set(dev) & set(test))}")
    for g in groups:
        if len(g) % n_folds:
            raise ValueError(f"a scene group of {len(g)} drivers does not divide into {n_folds} folds")
    # The first group fixes the order of the folds: its chunks start in sorted order.
    first = [p for p in _partitions(groups[0], n_folds)
             if all(p[k][0] < p[k + 1][0] for k in range(n_folds - 1))]
    others = [list(_partitions(g, n_folds)) for g in groups[1:]]
    best = None
    for p0 in first:
        for rest in product(*others):
            vals = [sorted(p0[k] + sum((r[k] for r in rest), ())) for k in range(n_folds)]
            counts = [sum(images[d] for d in v) for v in vals]
            key = (max(counts) - min(counts), vals)
            if best is None or key < best:
                best = key
    folds = [{"train": sorted(set(dev) - set(v)), "val": v} for v in best[1]]
    return {"test": test, "dev": dev, "folds": folds}


def driver_subset(drivers, k, seed):
    """The first `k` drivers after a shuffle of the sorted list with `seed` (decision P21).
    A smaller `k` gives a part of the drivers of a larger `k`."""
    order = sorted(drivers)
    if not 0 < k <= len(order):
        raise ValueError(f"cannot take {k} of {len(order)} drivers")
    random.Random(seed).shuffle(order)
    return sorted(order[:k])


def load_split(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "none"


def log_test_access(split_name, reason, log_path=TEST_ACCESS_LOG):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_utc", "git_commit", "split", "reason"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    _git_commit(), split_name, reason])


def drivers(split, part, fold=None, unlock_test=False, reason=None,
            log_path=TEST_ACCESS_LOG):
    """Return the drivers of one part: "dev", "train", "val", or "test".

    "train" and "val" need `fold`. "test" needs `unlock_test=True` and a reason.
    Each test unlock adds a row to the test access log.
    """
    if part == "dev":
        return list(split["dev"])
    if part in ("train", "val"):
        if fold is None:
            raise ValueError(f"part {part!r} needs a fold")
        return list(split["folds"][fold][part])
    if part == "test":
        if not unlock_test:
            raise PermissionError("the test drivers are locked, see docs/PLAN.md section 4")
        if not reason:
            raise ValueError("a test unlock needs a reason")
        log_test_access(split.get("name", "unknown"), reason, log_path)
        return list(split["test"])
    raise ValueError(f"unknown part {part!r}")
