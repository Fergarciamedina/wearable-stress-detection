"""
Majority Class Baseline Classifier
COEN 6322: AI for Wearable Health Applications
"""

from typing import Optional
import numpy as np
from sklearn.dummy import DummyClassifier


class MajorityBaselineClassifier:
    """Standard baseline classifier predicting the empirical majority class."""

    def __init__(self):
        self.clf = DummyClassifier(strategy="most_frequent")
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityBaselineClassifier":
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        return self.clf.predict_proba(X)
