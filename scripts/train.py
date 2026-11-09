#!/usr/bin/env python3
"""Train a classifier on labeled data and evaluate on a held-out split.

Usage: python -m scripts.train [--labels labels_ground_truth.csv]
       [--model logreg|centroid] [--test-frac 0.25] [--seed 7]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defect_vision.classifier import LogisticRegression, NearestCentroid  # noqa: E402
from defect_vision.evaluate import classification_report  # noqa: E402
from defect_vision.features import extract_all, standardize  # noqa: E402
from defect_vision.generate import load_pgm  # noqa: E402
from defect_vision.labeling import load_labels_csv  # noqa: E402


def load_labeled(data_dir, labels_path):
    rows = load_labels_csv(labels_path)
    images, labels = [], []
    for r in rows:
        images.append(load_pgm(os.path.join(data_dir, r["file"])))
        labels.append(r["label"])
    return np.stack(images), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "data"))
    ap.add_argument("--labels", default=None)
    ap.add_argument("--model", choices=["logreg", "centroid"], default="logreg")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    labels_path = args.labels or os.path.join(base, "labels_ground_truth.csv")
    images, labels = load_labeled(args.data, labels_path)
    X = extract_all(images)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(labels))
    n_test = int(len(labels) * args.test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X[train_idx], [labels[i] for i in train_idx]
    X_test, y_test = X[test_idx], [labels[i] for i in test_idx]
    X_train, mu, sd = standardize(X_train)
    X_test = (X_test - mu) / sd

    model = LogisticRegression() if args.model == "logreg" else NearestCentroid()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"model: {args.model} | train={len(y_train)} test={len(y_test)}")
    print(classification_report(y_test, pred))
    if isinstance(model, LogisticRegression):
        print(f"final train loss: {model.loss_history_[-1]:.4f} "
              f"(started {model.loss_history_[0]:.4f})")


if __name__ == "__main__":
    main()
