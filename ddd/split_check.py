"""Check of a split: balance of the folds and near copies across drivers."""
from collections import Counter

import numpy as np
from PIL import Image

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def fold_stats(audit, split):
    """Images and class shares of each fold, from the audit counts."""
    m = audit["images_per_driver_and_class"]
    classes = audit["classes"]
    out = []
    for k, fold in enumerate(split["folds"]):
        row = {"fold": k}
        for part in ("train", "val"):
            per_class = {c: sum(m[d].get(c, 0) for d in fold[part]) for c in classes}
            n = sum(per_class.values())
            row[f"{part}_images"] = n
            row[f"{part}_class_share"] = {c: round(v / n, 4) for c, v in per_class.items()}
        out.append(row)
    return out


def dhash(gray, size=8):
    """64-bit difference hash of a gray PIL image: the sign of the step between neighbor pixels."""
    g = np.asarray(gray.resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel())


def scene_groups(means, k=2):
    """Group drivers by the mean gray thumbnail of their images (average linkage, correlation).

    The groups show drivers that share a car or a camera position.
    """
    names = sorted(means)
    x = np.stack([(means[n] - means[n].mean()) / means[n].std() for n in names])
    sim = x @ x.T / x.shape[1]
    groups = [[i] for i in range(len(names))]
    while len(groups) > k:
        score, i, j = max((sim[np.ix_(a, b)].mean(), i, j)
                          for i, a in enumerate(groups) for j, b in enumerate(groups) if i < j)
        groups[i] += groups.pop(j)
    return sorted(sorted(names[i] for i in g) for g in groups)


def nearest_other_group(hashes, groups):
    """For each hash, the smallest Hamming distance to a hash of another group, and its index."""
    hashes = np.asarray(hashes, dtype=np.uint8)
    groups = np.asarray(groups)
    best = np.full(len(hashes), 65, dtype=np.int16)
    arg = np.full(len(hashes), -1)
    for i in range(len(hashes)):
        d = _POPCOUNT[hashes ^ hashes[i]].sum(axis=1, dtype=np.int16)
        d[groups == groups[i]] = 65
        j = int(d.argmin())
        best[i], arg[i] = d[j], j
    return best, arg


def nearest_same_group(hashes, groups):
    """For each hash, the smallest Hamming distance to another hash of the same group."""
    hashes = np.asarray(hashes, dtype=np.uint8)
    groups = np.asarray(groups)
    best = np.full(len(hashes), 65, dtype=np.int16)
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        h = hashes[idx]
        for n, i in enumerate(idx):
            d = _POPCOUNT[h ^ hashes[i]].sum(axis=1, dtype=np.int16)
            d[n] = 65
            best[i] = d.min()
    return best


def histogram(values, top=16):
    """Counts of each distance up to `top`, and one count above it."""
    c = Counter(int(v) for v in values)
    out = {str(k): c.get(k, 0) for k in range(top + 1)}
    out[f">{top}"] = sum(v for k, v in c.items() if k > top)
    return out
