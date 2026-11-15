import numpy as np
from PIL import Image

from ddd.split_check import (dhash, fold_stats, histogram, nearest_other_group,
                              nearest_same_group, scene_groups)


def test_fold_stats_counts_and_shares():
    audit = {"classes": ["c0", "c1"],
             "images_per_driver_and_class": {"a": {"c0": 3, "c1": 1}, "b": {"c0": 1, "c1": 1}}}
    split = {"folds": [{"train": ["a"], "val": ["b"]}]}
    (row,) = fold_stats(audit, split)
    assert row["train_images"] == 4 and row["val_images"] == 2
    assert row["train_class_share"] == {"c0": 0.75, "c1": 0.25}


def test_dhash_same_image_same_hash():
    im = Image.linear_gradient("L")
    assert dhash(im).shape == (8,)
    assert (dhash(im) == dhash(im.resize((64, 64)))).all()


def test_scene_groups_join_similar_drivers():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=50), rng.normal(size=50)
    means = {"p1": a, "p2": a + 0.1 * rng.normal(size=50), "p3": b, "p4": b + 0.1 * rng.normal(size=50)}
    assert scene_groups(means, k=2) == [["p1", "p2"], ["p3", "p4"]]


def test_nearest_distances_respect_groups():
    h = np.array([[0] * 8, [1] + [0] * 7, [255] * 8], dtype=np.uint8)
    other, arg = nearest_other_group(h, ["x", "x", "y"])
    assert list(other) == [64, 63, 63] and arg[2] == 1
    assert list(nearest_same_group(h, ["x", "x", "y"])) == [1, 1, 65]
    assert histogram([0, 1, 1, 40], top=2) == {"0": 1, "1": 2, "2": 0, ">2": 1}
