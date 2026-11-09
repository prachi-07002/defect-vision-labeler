# defect-vision-labeler

Synthetic surface-defect image generation, a labeling workflow, and a
from-scratch NumPy defect classifier — a prototype pipeline for the kind of
work in the Spring 2027 FTR project **"AI implementation for smart
manufacturing"** (Prof. Martin Byung-Guk Jun / Tyson Oh): inspecting
industrial surfaces, labeling defect data, and training models to spot
scratches, dents, and cracks on manufactured parts.

Pure **Python stdlib + NumPy only** — no OpenCV, PIL, or scikit-learn.
All images are stored as plain-text **ASCII PGM (P2)** files; labels live in
CSV. No binary files anywhere in the repo.

## What it does

1. **Generates** synthetic 64×64 grayscale surface images in 4 classes:
   `clean`, `scratch` (thin bright line), `dent` (dark blob + bright rim),
   `crack` (jagged branching dark polyline) — with illumination gradients,
   machining texture, and sensor noise.
2. **Labels** them three ways: ground truth from generation, a heuristic
   rule labeler (no ground truth needed), or an interactive terminal
   workflow that shows an ASCII preview and asks you to pick the class.
3. **Classifies** with two from-scratch NumPy models on 12 hand-engineered
   features (gradient energy, bright/dark pixel fractions, line score,
   centre-dark mass, …): nearest-centroid and multinomial logistic
   regression (softmax + gradient descent).
4. **Evaluates** with accuracy, per-class precision/recall/F1, and a
   confusion matrix.

## How to run

```bash
cd defect-vision-labeler

# 1. Generate 400 images (100 per class) into data/
python -m scripts.generate_dataset

# 2. Label them (ground truth / rule-based / interactive)
python -m scripts.label --mode ground_truth
python -m scripts.label --mode rule
python -m scripts.label --mode interactive --limit 20

# 3. Train + evaluate (75/25 split, seed 7)
python -m scripts.train --model logreg
python -m scripts.train --model centroid

# All-in-one demo (generates in memory, labels, trains, evaluates)
python -m scripts.demo

# Tests
python -m unittest discover -s tests -v
```

## Headline results (measured, 400 images, 100-image held-out test set)

| Model | Accuracy | Notes |
|---|---|---|
| Logistic regression (from scratch) | **1.000** (100/100) | perfect confusion matrix; train loss 1.3853 → 0.0379 |
| Nearest-centroid | **0.970** (97/100) | scratch 1.000 recall; dent 0.952, crack 0.920 recall |
| Rule-based labeler (no ground truth) | **0.805** (322/400) | dent precision 1.000, crack precision 1.000; clean recall 1.000, scratch recall 0.650 |

Logistic regression confusion matrix (rows = true, cols = predicted):

```
            clean  scratch  dent  crack
clean          26        0     0      0
scratch         0       28     0      0
dent            0        0    21      0
crack           0        0     0     25
```

Per-class (logistic regression): precision = recall = F1 = 1.000 for all four
classes on the 100-image test set. Nearest-centroid per-class F1: clean 0.945,
scratch 1.000, dent 0.976, crack 0.958.

The rule labeler is deliberately the weakest link — it shows why a learned
model beats hand-written heuristics on noisy inspection data, and it gives
the interactive labeler a starting point to correct.

## Tests

**15/15 unit tests pass** (`python -m unittest discover -s tests`): image
generation validity and balance, PGM text-only round-trip, feature
discrimination, rule-labeler accuracy, CSV label round-trip, both
classifiers ≥ 0.70 accuracy, confusion-matrix/metric correctness.

## Project layout

```
defect_vision/
  generate.py    # synthetic clean/scratch/dent/crack images + ASCII PGM I/O
  features.py    # 12 engineered features (gradients, line score, dark mass…)
  labeling.py    # ground-truth / rule / interactive labeling, CSV I/O
  classifier.py  # NearestCentroid + softmax logistic regression (NumPy)
  evaluate.py    # accuracy, confusion matrix, per-class precision/recall/F1
scripts/
  generate_dataset.py  # build data/ (400 .pgm + manifest.csv)
  label.py             # label data/ -> labels_*.csv
  train.py             # train + evaluate on labeled data
  demo.py              # end-to-end run, prints all headline results
tests/test_defect_vision.py
data/                  # generated dataset (400 .pgm + manifest.csv)
labels_ground_truth.csv
labels_rule.csv
```

## Application talking point

> "Built a defect-inspection pipeline from scratch in NumPy: generated
> synthetic surface-defect images, built a three-mode labeling workflow,
> and trained a logistic-regression classifier that hits 100% accuracy on a
> 100-image held-out test set (97% for a nearest-centroid baseline), with
> the rule-based labeler at 80.5% showing why learned models beat
> hand-written inspection heuristics."
