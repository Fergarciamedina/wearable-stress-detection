"""
Scientific Evaluation Metrics and Statistical Interaction Module
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Implements:
- Macro-F1 and Balanced Accuracy (primary metrics).
- Precision, Recall, F1 per class, and Confusion Matrix.
- Participant-clustered Bootstrap 95% Confidence Intervals.
- Paired model contrasts (RF vs. CNN) and interaction effects.
- Exercise False-Stress Rate (FAR) on out-of-distribution physical exercise.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes classification metrics for binary REST (0) vs. STRESS (1).
    Avoids zero-division errors.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    # Recall per class
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    balanced_acc = 0.5 * (rec_0 + rec_1)

    # Precision per class
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # F1 per class
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0
    macro_f1 = 0.5 * (f1_0 + f1_1)

    return {
        "balanced_acc": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "precision_rest": round(prec_0, 4),
        "recall_rest": round(rec_0, 4),
        "f1_rest": round(f1_0, 4),
        "precision_stress": round(prec_1, 4),
        "recall_stress": round(rec_1, 4),
        "f1_stress": round(f1_1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def compute_clustered_bootstrap_ci(subject_metrics: List[float], n_bootstrap: int = 1000,
                                   confidence_level: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    """
    Computes subject-clustered bootstrap confidence interval.
    Samples participants with replacement to account for intra-subject window correlation.
    Returns: (mean, ci_lower, ci_upper)
    """
    if len(subject_metrics) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.RandomState(seed)
    n = len(subject_metrics)
    data = np.array(subject_metrics, dtype=np.float64)
    point_estimate = float(np.mean(data))

    boot_means = []
    for _ in range(n_bootstrap):
        sample_indices = rng.choice(n, size=n, replace=True)
        boot_means.append(np.mean(data[sample_indices]))

    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_means, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))

    return round(point_estimate, 4), round(ci_lower, 4), round(ci_upper, 4)


def compute_paired_difference_ci(scores_a: List[float], scores_b: List[float],
                                 n_bootstrap: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    """
    Computes paired bootstrap confidence interval for difference (Model A - Model B).
    If 95% CI excludes 0, difference is statistically significant.
    """
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    return compute_clustered_bootstrap_ci(diffs, n_bootstrap=n_bootstrap, seed=seed)


def compute_exercise_far(y_exercise_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes Exercise False-Stress Rate:
    Exercise FAR = (Windows classified as STRESS) / (Total eligible exercise windows).
    """
    total = len(y_exercise_pred)
    if total == 0:
        return {"total_windows": 0, "false_stress_windows": 0, "exercise_far": 0.0}

    false_stress = int(np.sum(y_exercise_pred == 1))
    far = false_stress / total
    return {
        "total_windows": total,
        "false_stress_windows": false_stress,
        "exercise_far": round(far, 4)
    }
