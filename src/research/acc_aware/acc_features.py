"""
ACC Feature Extraction Module — H1 ACC-Aware Contextualization
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Extracts seven movement-intensity features from a 30-second 3-axis accelerometer
window recorded at 32 Hz (Empatica E4 raw integer units; gravity ≈ 64 units).

Features
--------
1. SVM_MEAN       — mean of the signal vector magnitude
2. SVM_STD        — standard deviation of SVM
3. DYNAMIC_SVM_MEAN  — mean of gravity-subtracted SVM (dynamic acceleration)
4. DYNAMIC_SVM_STD   — std of gravity-subtracted SVM
5. SMA            — signal magnitude area (area under the 3-axis mean absolute value)
6. ENERGY         — mean squared SVM
7. JERK_MEAN      — mean of the discrete first-order derivative of SVM
"""

from typing import List
import numpy as np

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
ACC_FEATURE_NAMES: List[str] = [
    "SVM_MEAN",
    "SVM_STD",
    "DYNAMIC_SVM_MEAN",
    "DYNAMIC_SVM_STD",
    "SMA",
    "ENERGY",
    "JERK_MEAN",
]
_N_FEATURES = len(ACC_FEATURE_NAMES)  # 7

# Rolling-window length for gravity removal (1 second at 32 Hz)
_GRAVITY_WINDOW_SAMPLES = 32


def _remove_gravity(svm: np.ndarray, window: int = _GRAVITY_WINDOW_SAMPLES) -> np.ndarray:
    """Subtract a 1-second rolling mean to isolate dynamic (movement) acceleration."""
    if len(svm) < window:
        return svm - np.mean(svm)
    kernel = np.ones(window, dtype=np.float64) / window
    gravity_estimate = np.convolve(svm, kernel, mode="same")
    return svm - gravity_estimate


def extract_acc_features(acc_window: np.ndarray, fs: float = 32.0) -> np.ndarray:
    """
    Extracts 7 movement-intensity features from a 30-second ACC window.

    Parameters
    ----------
    acc_window : np.ndarray, shape (N, 3)
        Raw 3-axis accelerometer samples in Empatica E4 integer units.
        N is expected to be 960 (32 Hz × 30 s) but any length ≥ 2 is accepted.
    fs : float
        Sampling frequency in Hz (default 32.0 for Empatica E4 ACC).

    Returns
    -------
    np.ndarray, shape (7,), dtype float32
        Feature vector in the order defined by ``ACC_FEATURE_NAMES``.
        Returns zeros if the input is empty or malformed.
    """
    zero = np.zeros(_N_FEATURES, dtype=np.float32)

    if acc_window is None:
        return zero

    arr = np.asarray(acc_window, dtype=np.float64)
    if arr.ndim == 1:
        # Treat as single-axis; replicate to (N, 3) so the math is uniform
        arr = np.stack([arr, arr, arr], axis=1)
    if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) < 2:
        return zero

    # Signal Vector Magnitude — total acceleration magnitude per sample
    svm = np.sqrt(np.sum(arr ** 2, axis=1))  # shape (N,)

    # Dynamic SVM — gravity removed via rolling subtraction
    dynamic_svm = _remove_gravity(svm)

    # Feature 1 & 2: raw SVM statistics
    svm_mean = float(np.mean(svm))
    svm_std = float(np.std(svm))

    # Feature 3 & 4: dynamic SVM statistics
    dyn_mean = float(np.mean(np.abs(dynamic_svm)))
    dyn_std = float(np.std(dynamic_svm))

    # Feature 5: Signal Magnitude Area — mean of per-axis absolute values summed
    sma = float(np.mean(np.sum(np.abs(arr), axis=1)))

    # Feature 6: Energy — mean squared SVM
    energy = float(np.mean(svm ** 2))

    # Feature 7: Jerk — mean of first-order differences of SVM
    jerk = np.abs(np.diff(svm)) * fs  # units: (E4 units)/s
    jerk_mean = float(np.mean(jerk)) if len(jerk) > 0 else 0.0

    features = np.array(
        [svm_mean, svm_std, dyn_mean, dyn_std, sma, energy, jerk_mean],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

