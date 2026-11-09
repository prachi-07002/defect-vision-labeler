"""Evaluation: accuracy, per-class precision/recall/F1, confusion matrix."""

import numpy as np

from .generate import CLASSES


def confusion_matrix(y_true, y_pred, classes=None):
    classes = list(classes or CLASSES)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def per_class_metrics(y_true, y_pred, classes=None):
    """Return {class: {precision, recall, f1, support}}."""
    classes = list(classes or CLASSES)
    cm = confusion_matrix(y_true, y_pred, classes)
    out = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"precision": round(prec, 3), "recall": round(rec, 3),
                  "f1": round(f1, 3), "support": int(cm[i, :].sum())}
    return out


def format_confusion_matrix(cm, classes=None):
    classes = list(classes or CLASSES)
    header = "true \\ pred | " + " | ".join(f"{c:>8}" for c in classes)
    lines = [header, "-" * len(header)]
    for i, c in enumerate(classes):
        lines.append(f"{c:>10} | " + " | ".join(f"{v:8d}" for v in cm[i]))
    return "\n".join(lines)


def classification_report(y_true, y_pred, classes=None):
    classes = list(classes or CLASSES)
    cm = confusion_matrix(y_true, y_pred, classes)
    lines = [f"accuracy: {accuracy(y_true, y_pred):.3f}",
             format_confusion_matrix(cm, classes), "",
             f"{'class':>8} {'prec':>6} {'rec':>6} {'f1':>6} {'n':>4}"]
    for c, m in per_class_metrics(y_true, y_pred, classes).items():
        lines.append(f"{c:>8} {m['precision']:6.3f} {m['recall']:6.3f} "
                     f"{m['f1']:6.3f} {m['support']:4d}")
    return "\n".join(lines)
