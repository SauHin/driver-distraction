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
