#!/usr/bin/env python3
"""Label images in data/ and write labels to a CSV.

Modes:
  --mode ground_truth : copy the manifest labels (known from generation)
  --mode rule         : label with the heuristic rule labeler
  --mode interactive  : prompt the user per image (ASCII preview shown)

Usage: python -m scripts.label --mode rule --out labels_rule.csv [--limit N]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defect_vision.labeling import (  # noqa: E402
    interactive_label, labels_from_ground_truth, labels_from_rules,
    load_labels_csv, write_labels_csv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ground_truth", "rule", "interactive"],
                    default="rule")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--append-to", default=None,
                    help="existing labels CSV to keep and add to")
    args = ap.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(base, f"labels_{args.mode}.csv")

    if args.mode == "ground_truth":
        rows = labels_from_ground_truth(args.data)
    elif args.mode == "rule":
        rows = labels_from_rules(args.data)
    else:
        existing = {}
        if args.append_to and os.path.exists(args.append_to):
            existing = {r["image_id"]: r["label"]
                        for r in load_labels_csv(args.append_to)}
        rows = interactive_label(args.data, existing, args.limit)
        if args.append_to and os.path.exists(args.append_to):
            rows = load_labels_csv(args.append_to) + rows

    write_labels_csv(out, rows)
    from collections import Counter
    print(f"wrote {len(rows)} labels to {out}")
    print(Counter(r["label"] for r in rows))
    if args.mode == "rule":
        truth = {r["image_id"]: r["label"]
                 for r in labels_from_ground_truth(args.data)}
        agree = sum(1 for r in rows if truth[r["image_id"]] == r["label"])
        print(f"rule labeler agrees with ground truth on "
              f"{agree}/{len(rows)} ({agree / len(rows):.1%})")


if __name__ == "__main__":
    main()
