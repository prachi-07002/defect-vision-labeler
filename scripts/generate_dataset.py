#!/usr/bin/env python3
"""Generate the synthetic defect dataset into data/ (ASCII PGM + manifest.csv).

Usage: python -m scripts.generate_dataset [--n-per-class N] [--seed S] [--out DIR]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defect_vision.generate import save_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), "data"))
    args = ap.parse_args()
    rows = save_dataset(args.out, args.n_per_class, args.seed)
    from collections import Counter
    print(f"wrote {len(rows)} images to {args.out}")
    print(Counter(r["label"] for r in rows))


if __name__ == "__main__":
    main()
