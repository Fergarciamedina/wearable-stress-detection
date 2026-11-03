"""
Movement Contextualizer — H1 ACC-Aware Contextualization
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Detects high-movement windows from ACC features and suppresses false STRESS
predictions during exercise by converting them to REST (label 0).

Design decisions
----------------
* Movement threshold calibrated on **training** exercise windows only
  (25th percentile of DYNAMIC_SVM_MEAN).  Using the 25th percentile means
  the gate fires on the majority of true exercise windows while remaining
  conservative enough not to mislabel sedentary artifacts.
* At inference time the contextualizer only converts predicted-STRESS → REST;
  it never changes REST → STRESS, and it never touches true-STRESS detections.
"""

from typing import List
import numpy as np

from src.research.acc_aware.acc_features import extract_acc_features


class MovementContextualizer:
    """
    Suppresses exercise false alarms using an ACC movement gate.

    Typical workflow
    ----------------
    1. Call ``fit_threshold(exercise_acc_windows)`` on training exercise windows.
    2. Call ``contextualize(y_pred, acc_windows)`` at evaluation time.
    """

    def __init__(self, percentile: float = 25.0, fs: float = 32.0) -> None:
        """
        Parameters
        ----------
        percentile : float
            Percentile of DYNAMIC_SVM_MEAN (across training exercise windows)
            used as the movement gate threshold.  Lower ⟹ more conservative
            (fewer windows flagged as high-movement).
        fs : float
            ACC sampling frequency in Hz.
        """
        self.percentile = percentile
        self.fs = fs
        self.threshold_: float = 0.0
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training-time interface
    # ------------------------------------------------------------------

    def fit_threshold(self, exercise_acc_windows: List[np.ndarray]) -> "MovementContextualizer":
        """
        Calibrates the movement gate using labelled exercise ACC windows.

        Parameters
        ----------
        exercise_acc_windows : list of np.ndarray, each shape (N, 3)
            ACC windows from **training** subjects' exercise sessions.
            At least one non-empty window is required.

        Returns
        -------
        self
        """
        dyn_means: List[float] = []
        for w in exercise_acc_windows:
            feats = extract_acc_features(w, fs=self.fs)
            # Index 2 = DYNAMIC_SVM_MEAN (see ACC_FEATURE_NAMES)
            dyn_means.append(float(feats[2]))

        if len(dyn_means) == 0:
            # Fallback: gate never fires
            self.threshold_ = float("inf")
        else:
            self.threshold_ = float(np.percentile(dyn_means, self.percentile))

        self._is_fitted = True
        return self

    def fit_threshold_from_features(self, dyn_svm_means: np.ndarray) -> "MovementContextualizer":
        """
        Alternative fit accepting pre-computed DYNAMIC_SVM_MEAN values.

        Parameters
        ----------
        dyn_svm_means : np.ndarray
            1-D array of DYNAMIC_SVM_MEAN values from training exercise windows.
        """
        arr = np.asarray(dyn_svm_means, dtype=np.float64).ravel()
        self.threshold_ = float(np.percentile(arr, self.percentile)) if len(arr) > 0 else float("inf")
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference-time interface
    # ------------------------------------------------------------------

    def is_high_movement(self, acc_window: np.ndarray) -> bool:
        """Returns True if the window's dynamic SVM mean exceeds the threshold."""
        if not self._is_fitted:
            raise RuntimeError("Call fit_threshold() before is_high_movement().")
        feats = extract_acc_features(acc_window, fs=self.fs)
        return float(feats[2]) >= self.threshold_

    def contextualize(
        self,
        y_pred: np.ndarray,
        acc_windows: List[np.ndarray],
    ) -> np.ndarray:
        """
        Applies the movement gate to a batch of predictions.

        For every window where ``is_high_movement`` is True, a STRESS prediction
        (1) is overridden to REST (0).  All other predictions are unchanged.

        Parameters
        ----------
        y_pred : np.ndarray, shape (N,)
            Binary classifier predictions (0 = REST, 1 = STRESS).
        acc_windows : list of np.ndarray, len N
            Corresponding ACC windows for each prediction.

        Returns
        -------
        np.ndarray, shape (N,), dtype int
            Contextualized predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit_threshold() before contextualize().")

        y_ctx = np.array(y_pred, dtype=int)
        for i, acc_w in enumerate(acc_windows):
            if y_ctx[i] == 1 and self.is_high_movement(acc_w):
                y_ctx[i] = 0
        return y_ctx

