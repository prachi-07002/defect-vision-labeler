#!/usr/bin/env python3
"""End-to-end demo: generate data -> label (ground truth + rules) -> train
both classifiers -> evaluate. Prints headline results.

Usage: python -m scripts.demo [--n-per-class N]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defect_vision.classifier import LogisticRegression, NearestCentroid  # noqa: E402
from defect_vision.evaluate import (accuracy, classification_report,  # noqa: E402
                                    confusion_matrix)
from defect_vision.features import extract_all, standardize  # noqa: E402
from defect_vision.generate import generate_dataset  # noqa: E402
from defect_vision.labeling import rule_label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=100)
    args = ap.parse_args()

    print("== 1. generate synthetic surface images ==")
    images, labels = generate_dataset(args.n_per_class, seed=42)
    print(f"generated {len(labels)} images "
          f"({args.n_per_class} per class: clean/scratch/dent/crack)")

    print("\n== 2. rule-based labeling (no ground truth used) ==")
    rule_labels = [rule_label(im) for im in images]
    rule_acc = accuracy(labels, rule_labels)
    print(f"rule labeler accuracy vs ground truth: {rule_acc:.3f}")

    print("\n== 3. train classifiers on engineered features ==")
    X = extract_all(images)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(labels))
    n_test = len(labels) // 4
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    Xtr, ytr = X[train_idx], [labels[i] for i in train_idx]
    Xte, yte = X[test_idx], [labels[i] for i in test_idx]
    Xtr, mu, sd = standardize(Xtr)
    Xte = (Xte - mu) / sd

    for name, model in [("nearest-centroid", NearestCentroid()),
                        ("logistic regression", LogisticRegression())]:
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        print(f"\n-- {name} --")
        print(classification_report(yte, pred))
        print("confusion matrix (rows=true, cols=pred):")
        print(confusion_matrix(yte, pred))


if __name__ == "__main__":
    main()
