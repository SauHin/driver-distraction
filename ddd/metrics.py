"""Metrics of docs/PLAN.md section 5, from a confusion matrix."""
import numpy as np


def confusion(targets, preds, n=10):
    m = np.zeros((n, n), dtype=np.int64)
    np.add.at(m, (np.asarray(targets), np.asarray(preds)), 1)
    return m


def scores(m):
    """Accuracy, class-mean accuracy, and macro F1. A class with no images is left out of
    the class mean. A class with no images and no predictions is left out of macro F1."""
    m = np.asarray(m, dtype=np.float64)
    tp = np.diag(m)
    true_n, pred_n = m.sum(axis=1), m.sum(axis=0)
    has = true_n > 0
    recall = tp[has] / true_n[has]
    seen = (true_n + pred_n) > 0
    f1 = 2 * tp[seen] / (true_n[seen] + pred_n[seen])
    return {"accuracy": float(tp.sum() / m.sum()),
            "class_mean_accuracy": float(recall.mean()),
            "macro_f1": float(f1.mean())}


def class_scores(m):
    """Precision, recall, and F1 of each class. A class with no predictions has precision 0."""
    m = np.asarray(m, dtype=np.float64)
    tp, true_n, pred_n = np.diag(m), m.sum(axis=1), m.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_n > 0, tp / pred_n, 0.0)
        recall = np.where(true_n > 0, tp / true_n, 0.0)
        f1 = np.where(true_n + pred_n > 0, 2 * tp / (true_n + pred_n), 0.0)
    return [{"precision": float(p), "recall": float(r), "f1": float(f)}
            for p, r, f in zip(precision, recall, f1)]


def bootstrap_accuracy(accuracy, images, reps=10000, seed=0, level=0.95):
    """A percentile interval of the pooled accuracy over drivers. Each repeat draws the drivers
    with replacement and pools their correct images. `accuracy` and `images` hold one value per driver."""
    acc, n = np.asarray(accuracy, dtype=np.float64), np.asarray(images, dtype=np.float64)
    idx = np.random.default_rng(seed).integers(0, len(acc), (reps, len(acc)))
    pooled = (acc[idx] * n[idx]).sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.quantile(pooled, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def binary_scores(m):
    """Precision, recall, and F1 of the status "distracted": classes c1 to c9 against c0.
    A distracted image with the wrong distracted class still has the correct status."""
    m = np.asarray(m, dtype=np.float64)
    tp, fp, fn = m[1:, 1:].sum(), m[0, 1:].sum(), m[1:, 0].sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"distracted_precision": float(precision), "distracted_recall": float(recall),
            "distracted_f1": float(f1)}
