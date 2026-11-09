"""Classifiers from scratch in NumPy:

- NearestCentroid: classify by nearest class mean in feature space.
- LogisticRegression: multinomial (softmax) logistic regression trained
  with full-batch gradient descent.
"""

import numpy as np

from .generate import CLASSES


class NearestCentroid:
    def __init__(self):
        self.centroids_ = None
        self.classes_ = list(CLASSES)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.centroids_ = np.stack([
            X[np.array(y) == c].mean(axis=0) for c in self.classes_
        ])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        d = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return np.array([self.classes_[i] for i in d.argmin(axis=1)])


class LogisticRegression:
    def __init__(self, lr=0.1, epochs=500, l2=1e-3, seed=0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.classes_ = list(CLASSES)
        self.W_ = None
        self.b_ = None
        self.loss_history_ = []

    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        k = len(self.classes_)
        y_idx = np.array([self.classes_.index(v) for v in y])
        rng = np.random.default_rng(self.seed)
        self.W_ = rng.normal(0, 0.01, size=(d, k))
        self.b_ = np.zeros(k)
        Y = np.eye(k)[y_idx]
        for _ in range(self.epochs):
            P = self._softmax(X @ self.W_ + self.b_)
            loss = -float(np.log(P[np.arange(n), y_idx] + 1e-12).mean())
            loss += self.l2 * float((self.W_ ** 2).sum())
            self.loss_history_.append(loss)
            G = (P - Y) / n
            dW = X.T @ G + 2 * self.l2 * self.W_
            db = G.sum(axis=0)
            self.W_ -= self.lr * dW
            self.b_ -= self.lr * db
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return self._softmax(X @ self.W_ + self.b_)

    def predict(self, X):
        return np.array([self.classes_[i] for i in self.predict_proba(X).argmax(axis=1)])
