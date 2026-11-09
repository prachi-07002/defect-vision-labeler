"""Unit tests for defect-vision-labeler. Run: python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from defect_vision.classifier import LogisticRegression, NearestCentroid
from defect_vision.evaluate import (accuracy, confusion_matrix,
                                    per_class_metrics)
from defect_vision.features import (FEATURE_NAMES, extract_all,
                                    extract_features, standardize)
from defect_vision.generate import (CLASSES, generate_dataset,
                                    generate_image, load_pgm, save_dataset,
                                    save_pgm)
from defect_vision.labeling import (ascii_preview, labels_from_ground_truth,
                                    load_labels_csv, rule_label,
                                    write_labels_csv)


class TestGeneration(unittest.TestCase):
    def test_all_classes_generate_valid_images(self):
        rng = np.random.default_rng(0)
        for label in CLASSES:
            img = generate_image(label, rng)
            self.assertEqual(img.shape, (64, 64))
            self.assertGreaterEqual(img.min(), 0.0)
            self.assertLessEqual(img.max(), 1.0)
            self.assertTrue(np.isfinite(img).all())

    def test_unknown_label_raises(self):
        with self.assertRaises(ValueError):
            generate_image("rust", np.random.default_rng(0))

    def test_defect_images_differ_from_clean(self):
        rng = np.random.default_rng(1)
        clean = generate_image("clean", rng)
        for label in ("scratch", "dent", "crack"):
            img = generate_image(label, rng)
            self.assertGreater(np.abs(img - clean).mean(), 0.005, label)

    def test_dataset_balanced(self):
        images, labels = generate_dataset(n_per_class=25, seed=123)
        self.assertEqual(images.shape, (100, 64, 64))
        for label in CLASSES:
            self.assertEqual(labels.count(label), 25)

    def test_pgm_roundtrip_text_only(self):
        img = generate_image("dent", np.random.default_rng(2))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.pgm")
            save_pgm(path, img)
            with open(path, "rb") as fh:
                raw = fh.read()
            self.assertTrue(raw.startswith(b"P2\n"))  # ASCII PGM, not binary
            self.assertNotIn(b"\x00", raw)
            back = load_pgm(path)
        self.assertEqual(back.shape, (64, 64))
        self.assertLess(np.abs(back - img).max(), 1.0 / 255 + 1e-9)

    def test_save_dataset_writes_manifest_and_pgms(self):
        with tempfile.TemporaryDirectory() as d:
            rows = save_dataset(d, n_per_class=3, seed=5)
            self.assertEqual(len(rows), 12)
            self.assertTrue(os.path.exists(os.path.join(d, "manifest.csv")))
            for r in rows:
                self.assertTrue(os.path.exists(os.path.join(d, r["file"])))
                self.assertIn(r["label"], CLASSES)


class TestFeatures(unittest.TestCase):
    def test_features_shape_and_names(self):
        img = generate_image("scratch", np.random.default_rng(3))
        f = extract_features(img)
        self.assertEqual(f.shape, (len(FEATURE_NAMES),))
        self.assertEqual(f.shape, (12,))
        self.assertTrue(np.isfinite(f).all())

    def test_features_discriminate_classes(self):
        images, labels = generate_dataset(n_per_class=25, seed=123)
        X = extract_all(images)
        self.assertEqual(X.shape, (100, 12))
        clean_mean = X[np.array(labels) == "clean"].mean(axis=0)
        for label in ("scratch", "dent", "crack"):
            m = X[np.array(labels) == label].mean(axis=0)
            self.assertGreater(np.abs(m - clean_mean).max(), 0.5, label)


class TestLabeling(unittest.TestCase):
    def test_rule_labeler_beats_chance(self):
        images, labels = generate_dataset(n_per_class=25, seed=123)
        preds = [rule_label(im) for im in images]
        acc = accuracy(labels, preds)
        self.assertGreaterEqual(acc, 0.60, f"rule labeler accuracy {acc:.3f} too low")

    def test_rule_labeler_finds_scratches(self):
        images, labels = generate_dataset(n_per_class=25, seed=123)
        preds = [rule_label(im) for im in images]
        scratch_idx = [i for i, l in enumerate(labels) if l == "scratch"]
        hit = sum(1 for i in scratch_idx if preds[i] == "scratch")
        self.assertGreaterEqual(hit / len(scratch_idx), 0.5)

    def test_ascii_preview_shape(self):
        img = generate_image("crack", np.random.default_rng(4))
        lines = ascii_preview(img, width=16, height=8).split("\n")
        self.assertEqual(len(lines), 8)
        self.assertTrue(all(len(line) == 16 for line in lines))

    def test_labels_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            save_dataset(d, n_per_class=2, seed=9)
            rows = labels_from_ground_truth(d)
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(r["source"] == "ground_truth" for r in rows))
            path = os.path.join(d, "labels.csv")
            write_labels_csv(path, rows)
            self.assertEqual(load_labels_csv(path), rows)


class TestClassifiers(unittest.TestCase):
    def _split(self, n_per_class=25, seed=123):
        images, labels = generate_dataset(n_per_class=n_per_class, seed=seed)
        X = extract_all(images)
        rng = np.random.default_rng(7)
        idx = rng.permutation(len(labels))
        te, tr = idx[:25], idx[25:]
        return X, labels, tr, te

    def test_centroid_classifier_accuracy(self):
        X, labels, tr, te = self._split()
        clf = NearestCentroid().fit(X[tr], [labels[i] for i in tr])
        pred = clf.predict(X[te])
        self.assertGreaterEqual(accuracy([labels[i] for i in te], pred), 0.70)

    def test_logreg_classifier_accuracy(self):
        X, labels, tr, te = self._split()
        Xtr, mu, sd = standardize(X[tr])
        clf = LogisticRegression(epochs=400).fit(Xtr, [labels[i] for i in tr])
        pred = clf.predict((X[te] - mu) / sd)
        self.assertGreaterEqual(accuracy([labels[i] for i in te], pred), 0.70)
        # loss should decrease over training
        self.assertLess(clf.loss_history_[-1], clf.loss_history_[0])


class TestEvaluation(unittest.TestCase):
    def test_confusion_matrix_and_metrics(self):
        y_true = ["clean", "clean", "scratch", "dent", "crack", "crack"]
        y_pred = ["clean", "scratch", "scratch", "dent", "crack", "clean"]
        cm = confusion_matrix(y_true, y_pred)
        self.assertEqual(cm.shape, (4, 4))
        self.assertEqual(cm.sum(), 6)
        self.assertEqual(cm[0, 0], 1)  # clean -> clean
        self.assertEqual(cm[3, 0], 1)  # crack -> clean
        m = per_class_metrics(y_true, y_pred)
        self.assertEqual(m["dent"]["precision"], 1.0)
        self.assertEqual(m["dent"]["recall"], 1.0)
        self.assertEqual(m["clean"]["recall"], 0.5)
        self.assertAlmostEqual(accuracy(y_true, y_pred), 4 / 6)


if __name__ == "__main__":
    unittest.main()
