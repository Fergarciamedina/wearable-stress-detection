"""
Random Forest Model Wrapper & Serialization
COEN 6322: AI for Wearable Health Applications
"""

import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class StressRandomForest:
    """Standardized Random Forest classifier with integrated feature scaling."""

    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 10, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StressRandomForest":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save(self, filepath: str) -> None:
        """Saves deployment dictionary containing model, scaler, and metadata."""
        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, filepath: str) -> "StressRandomForest":
        """Loads model and scaler from pickled artifact."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)
        instance = cls(
            n_estimators=payload.get("n_estimators", 100),
            max_depth=payload.get("max_depth", 10),
            random_state=payload.get("random_state", 42)
        )
        instance.model = payload["model"]
        instance.scaler = payload["scaler"]
        instance.is_fitted = True
        return instance
