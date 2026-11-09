"""Labeling workflow: load ground truth, auto-label by rule, or label
interactively at the terminal with a downsampled ASCII preview.

Labels are stored as CSV: image_id, file, label, source
(source is one of: ground_truth | rule | human).
"""

import csv
import os

import numpy as np

from .features import extract_features
from .generate import CLASSES, load_pgm

# Characters from dark to bright for the ASCII preview.
_ASCII = " .:-=+*#%@"


def ascii_preview(img, width=32, height=16):
    """Downsample an image to a small ASCII-art string."""
    h, w = img.shape
    ys = np.linspace(0, h - 1, height).astype(int)
    xs = np.linspace(0, w - 1, width).astype(int)
    small = img[np.ix_(ys, xs)]
    lo, hi = small.min(), small.max()
    span = hi - lo if hi > lo else 1.0
    idx = ((small - lo) / span * (len(_ASCII) - 1)).astype(int)
    return "\n".join("".join(_ASCII[v] for v in row) for row in idx)


def rule_label(img):
    """Heuristic label from engineered features (no ground truth needed).

    Rules, in order:
    - strong bright line -> scratch (line_score / max_row_bright ~22 vs ~0)
    - compact dark mass in the centre, low edge peak -> dent
      (centre_dark ~0.095, grad_max ~0.11)
    - many dark pixels spread over many rows, sharp edges -> crack
      (dark_spread ~0.35, grad_max ~0.40)
    - otherwise clean (all defect features ~0)
    """
    f = extract_features(img)
    (mean, std, grad_mean, grad_max, bright_frac, dark_frac,
     max_row_dark, max_col_dark, max_row_bright, centre_dark,
     dark_spread, line_score) = f
    if line_score >= 12 and max_row_bright >= 12:
        return "scratch"
    if centre_dark >= 0.03 and grad_max < 0.20:
        return "dent"
    if dark_frac >= 0.01 and dark_spread >= 0.25:
        return "crack"
    return "clean"


def load_manifest(directory):
    with open(os.path.join(directory, "manifest.csv")) as f:
        return list(csv.DictReader(f))


def load_labels_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_labels_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "file", "label", "source"])
        w.writeheader()
        w.writerows(rows)


def labels_from_ground_truth(directory):
    """Label every image in a dataset dir with its manifest (ground-truth) label."""
    rows = []
    for r in load_manifest(directory):
        rows.append({"image_id": r["image_id"], "file": r["file"],
                     "label": r["label"], "source": "ground_truth"})
    return rows


def labels_from_rules(directory):
    """Label every image in a dataset dir with the heuristic rule labeler."""
    rows = []
    for r in load_manifest(directory):
        img = load_pgm(os.path.join(directory, r["file"]))
        rows.append({"image_id": r["image_id"], "file": r["file"],
                     "label": rule_label(img), "source": "rule"})
    return rows


def interactive_label(directory, existing=None, limit=None):
    """Prompt the user to label images one by one at the terminal.

    existing: optional dict image_id -> label to skip already-labeled images.
    Returns list of label rows (only the ones labeled in this session).
    """
    existing = existing or {}
    rows = []
    manifest = load_manifest(directory)
    shown = 0
    for r in manifest:
        if r["image_id"] in existing:
            continue
        img = load_pgm(os.path.join(directory, r["file"]))
        print(f"\n--- {r['image_id']} ({r['file']}) ---")
        print(ascii_preview(img))
        print("Classes: " + ", ".join(f"{i}={c}" for i, c in enumerate(CLASSES)))
        while True:
            ans = input("Label [0-3] (or 's' to skip, 'q' to quit): ").strip().lower()
            if ans == "q":
                return rows
            if ans == "s":
                break
            if ans in ("0", "1", "2", "3"):
                label = CLASSES[int(ans)]
                rows.append({"image_id": r["image_id"], "file": r["file"],
                             "label": label, "source": "human"})
                break
            print("Please enter 0-3, 's' or 'q'.")
        shown += 1
        if limit is not None and shown >= limit:
            break
    return rows
