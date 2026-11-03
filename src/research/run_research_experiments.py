"""
Research Experiment Runner — H1, H2, and Combined Contextualization Evaluation
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Executes a strict 34-fold Leave-One-Subject-Out (LOSO) evaluation for four
experiment configurations (E0–E3):

  E0  — Baseline Random Forest (EDA+PPG, 20 features)
  E1  — RF + H1 ACC-aware movement gate
  E2  — RF + H2 KNN physiological similarity (k=7, with exercise label=2)
  E3  — RF + H1 ACC-gate + H2 KNN similarity (sequential combination)

Per fold:
  1. Train RF on N-1 stress-session subjects (EDA+PPG, 20 features).
  2. Fit H1 threshold on training-subject exercise windows.
  3. Fit H2 KNN on training-subject EDA+PPG features + exercise windows.
  4. Evaluate each E0–E3 on the held-out test subject's stress windows
     (binary REST/STRESS) and exercise windows (false-alarm evaluation).
  5. Compute macro-F1 (binary, stress-session) and Exercise FAR.

Results are saved to:
  - data/results/research/comparison/research_comparison.csv  (per-config summary)
  - data/results/research/comparison/fold_predictions.csv     (per-fold per-config)

Usage:
    python -m src.research.run_research_experiments [--data-dir PATH]

The script honours the WEARABLE_DATA_DIR environment variable and the
standard ProjectConfig fallback logic.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path bootstrap — ensure 'src' is importable regardless of cwd
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve()
for _root in [_here.parent, _here.parent.parent, _here.parent.parent.parent]:
    if (_root / "src").is_dir():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

from src.config import ProjectConfig
from src.preprocessing.signal_loader import SignalLoader
from src.preprocessing.segmentation import WindowSegmenter
from src.features.feature_extractor import FeatureExtractor
from src.evaluation.metrics import compute_binary_metrics, compute_exercise_far
from src.evaluation.splits import load_splits_manifest, generate_loso_manifest

from src.research.acc_aware.acc_features import extract_acc_features
from src.research.acc_aware.movement_context import MovementContextualizer
from src.research.similarity_aware.knn_retrieval import PhysiologicalKNN
from src.research.similarity_aware.similarity_context import SimilarityContextualizer
from src.research.fusion.fusion_policy import FusionPolicy, EXPERIMENT_CONFIGS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SENSOR_CONFIG = "EDA+PPG"  # Research phase uses only the fusion sensor set
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=8,
    min_samples_split=4,
    class_weight="balanced",
    n_jobs=-1,
)
KNN_K = 7
ACC_FS = 32.0
EX_THRESHOLD = 0.40      # H2: exercise fraction gate
STRESS_THRESHOLD = 0.15   # H2: stress reinforcement fraction (informational only)
ACC_PERCENTILE = 25.0     # H1: percentile of DYNAMIC_SVM_MEAN on training exercise windows


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _extract_acc_array(windows: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
    """Returns the acc_window (or None) from each window dict."""
    return [w.get("acc_window") for w in windows]


def _collect_acc_exercise_windows(
    subject_exercise_windows: Dict[str, List[Dict[str, Any]]],
    train_subjects: List[str],
) -> List[np.ndarray]:
    """Collects non-None ACC exercise windows from training subjects."""
    acc_wins: List[np.ndarray] = []
    for s in train_subjects:
        for w in subject_exercise_windows.get(s, []):
            aw = w.get("acc_window")
            if aw is not None and len(aw) > 0:
                acc_wins.append(aw)
    return acc_wins


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------

def run_fold(
    fold: Dict[str, Any],
    subject_features: Dict[str, np.ndarray],
    subject_labels: Dict[str, np.ndarray],
    subject_acc_windows: Dict[str, List[Optional[np.ndarray]]],
    exercise_features: Dict[str, np.ndarray],
    exercise_acc_windows: Dict[str, List[Optional[np.ndarray]]],
    subject_exercise_windows: Dict[str, List[Dict[str, Any]]],
    seed: int,
) -> Dict[str, Any]:
    """
    Executes a single LOSO fold for all four configurations (E0–E3).

    Returns a dict keyed by experiment name ('E0'–'E3'), each containing:
        macro_f1, balanced_acc, f1_rest, f1_stress, exercise_far, n_exercise_windows
    """
    test_subj = fold["test_subject"]
    train_subjs = fold["train_subjects"]
    val_subjs = fold["val_subjects"]

    # -----------------------------------------------------------------------
    # 1. Build RF feature matrices
    # -----------------------------------------------------------------------
    X_train = np.concatenate([subject_features[s] for s in train_subjs], axis=0)
    y_train = np.concatenate([subject_labels[s] for s in train_subjs], axis=0)
    X_test = subject_features[test_subj]
    y_test = subject_labels[test_subj]

    # Train-only scaling — strict zero leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------------------------------------------------
    # 2. Train Random Forest
    # -----------------------------------------------------------------------
    rf = RandomForestClassifier(**RF_PARAMS, random_state=seed)
    rf.fit(X_train_scaled, y_train)
    y_pred_base = rf.predict(X_test_scaled)

    # -----------------------------------------------------------------------
    # 3. H1: Fit ACC movement gate on training exercise windows
    # -----------------------------------------------------------------------
    train_exercise_acc = _collect_acc_exercise_windows(subject_exercise_windows, train_subjs)
    acc_ctx = MovementContextualizer(percentile=ACC_PERCENTILE, fs=ACC_FS)
    if train_exercise_acc:
        acc_ctx.fit_threshold(train_exercise_acc)
    else:
        # Fallback: gate never fires
        acc_ctx.fit_threshold_from_features(np.array([float("inf")]))

    # -----------------------------------------------------------------------
    # 4. H2: Build KNN on training stress + exercise features
    # -----------------------------------------------------------------------
    knn = PhysiologicalKNN(k=KNN_K)
    X_train_ex_parts = []
    for s in train_subjs:
        if s in exercise_features and len(exercise_features[s]) > 0:
            X_train_ex_parts.append(exercise_features[s])
    X_train_ex = (
        np.concatenate(X_train_ex_parts, axis=0) if X_train_ex_parts else None
    )
    # KNN uses raw (unscaled) features; the KNN.fit() applies its own StandardScaler
    X_train_rs_raw = np.concatenate([subject_features[s] for s in train_subjs], axis=0)
    knn.fit(X_train_rs_raw, y_train, X_train_ex)

    sim_ctx = SimilarityContextualizer(knn, ex_threshold=EX_THRESHOLD, stress_threshold=STRESS_THRESHOLD)

    # -----------------------------------------------------------------------
    # 5. Evaluate all four configurations on the held-out stress session
    # -----------------------------------------------------------------------
    test_acc_wins = subject_acc_windows.get(test_subj, [None] * len(y_test))
    X_test_raw = subject_features[test_subj]

    results: Dict[str, Any] = {}
    for exp_name, exp_cfg in EXPERIMENT_CONFIGS.items():
        policy = FusionPolicy(
            config=exp_cfg,
            acc_ctx=acc_ctx if exp_cfg.use_h1_acc else None,
            sim_ctx=sim_ctx if exp_cfg.use_h2_knn else None,
        )
        y_ctx = policy.fuse(
            y_pred_base,
            acc_windows=test_acc_wins if exp_cfg.use_h1_acc else None,
            X_features=X_test_raw if exp_cfg.use_h2_knn else None,
        )
        m = compute_binary_metrics(y_test, y_ctx)
        results[exp_name] = {
            "macro_f1": m["macro_f1"],
            "balanced_acc": m["balanced_acc"],
            "f1_rest": m["f1_rest"],
            "f1_stress": m["f1_stress"],
        }

    # -----------------------------------------------------------------------
    # 6. Evaluate exercise FAR for each configuration
    # -----------------------------------------------------------------------
    ex_feats = exercise_features.get(test_subj)
    ex_acc_ws = exercise_acc_windows.get(test_subj, [])
    n_ex = 0

    if ex_feats is not None and len(ex_feats) > 0:
        n_ex = len(ex_feats)
        ex_feats_scaled = scaler.transform(ex_feats)
        y_ex_pred_base = rf.predict(ex_feats_scaled)

        for exp_name, exp_cfg in EXPERIMENT_CONFIGS.items():
            policy = FusionPolicy(
                config=exp_cfg,
                acc_ctx=acc_ctx if exp_cfg.use_h1_acc else None,
                sim_ctx=sim_ctx if exp_cfg.use_h2_knn else None,
            )
            y_ex_ctx = policy.fuse(
                y_ex_pred_base,
                acc_windows=ex_acc_ws if exp_cfg.use_h1_acc else None,
                X_features=ex_feats if exp_cfg.use_h2_knn else None,
            )
            far_res = compute_exercise_far(y_ex_ctx)
            results[exp_name]["exercise_far"] = far_res["exercise_far"]
            results[exp_name]["n_exercise_windows"] = n_ex
    else:
        for exp_name in EXPERIMENT_CONFIGS:
            results[exp_name]["exercise_far"] = float("nan")
            results[exp_name]["n_exercise_windows"] = 0

    return results


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_research_experiments(data_dir: Optional[str] = None) -> None:
    """
    Executes the full 34-fold LOSO research evaluation and saves results.
    """
    cfg = ProjectConfig(data_root=data_dir)
    research_dir = cfg.results_dir / "research" / "comparison"
    research_dir.mkdir(parents=True, exist_ok=True)

    out_summary = research_dir / "research_comparison.csv"
    out_folds = research_dir / "fold_predictions.csv"

    loader = SignalLoader(str(cfg.raw_data_dir))
    segmenter = WindowSegmenter(window_len_s=30.0, stride_s=15.0)
    feat_extractor = FeatureExtractor(eda_fs=4.0, bvp_fs=64.0)

    all_candidate_ids = (
        [f"S{i:02d}" for i in range(1, 19) if i != 2]
        + [f"f{i:02d}" for i in range(1, 19)]
    )

    print("=" * 70)
    print("COEN 6322 Research Phase — ACC-Aware + Similarity-Aware LOSO")
    print("=" * 70)
    print(f"\n[Phase 1] Loading and segmenting all sessions...")

    subject_stress_windows: Dict[str, List[Dict[str, Any]]] = {}
    subject_exercise_windows: Dict[str, List[Dict[str, Any]]] = {}
    eligible_paired: List[str] = []

    for s_id in all_candidate_ids:
        sess = loader.load_subject_stress_session(s_id)
        if sess is not None:
            wins = segmenter.segment_stress_session(sess)
            if len(wins) > 0:
                subject_stress_windows[s_id] = wins
                if sess["bvp_valid"] and s_id != "f07":
                    eligible_paired.append(s_id)

        ex_wins: List[Dict[str, Any]] = []
        for st in ["AEROBIC", "ANAEROBIC"]:
            ex_sess = loader.load_subject_exercise_session(s_id, st)
            if ex_sess is not None:
                ex_wins.extend(segmenter.segment_exercise_session(ex_sess))
        if ex_wins:
            subject_exercise_windows[s_id] = ex_wins

    print(f"Eligible paired subjects: {len(eligible_paired)}")
    print(f"Subjects with exercise sessions: {len(subject_exercise_windows)}")

    # Load or generate splits
    splits_path = str(cfg.splits_dir / "loso_splits.json")
    if os.path.exists(splits_path):
        manifest = load_splits_manifest(splits_path)
        print(f"Loaded {len(manifest)} existing LOSO folds from {splits_path}")
    else:
        manifest = generate_loso_manifest(eligible_paired, seed=42)
        print(f"Generated {len(manifest)} LOSO folds")

    # Pre-compute EDA+PPG features for all eligible subjects
    print("\n[Phase 2] Pre-computing EDA+PPG features...")
    subject_features: Dict[str, np.ndarray] = {}
    subject_labels: Dict[str, np.ndarray] = {}
    subject_acc_windows: Dict[str, List[Optional[np.ndarray]]] = {}

    exercise_features: Dict[str, np.ndarray] = {}
    exercise_acc_windows: Dict[str, List[Optional[np.ndarray]]] = {}

    for s_id in eligible_paired:
        wins = subject_stress_windows[s_id]
        subject_labels[s_id] = np.array([w["label"] for w in wins], dtype=int)
        subject_features[s_id] = np.array(
            [feat_extractor.extract_features_for_sensor_config(w, SENSOR_CONFIG) for w in wins]
        )
        subject_acc_windows[s_id] = [w.get("acc_window") for w in wins]

        if s_id in subject_exercise_windows:
            ex_wins = subject_exercise_windows[s_id]
            exercise_features[s_id] = np.array(
                [feat_extractor.extract_features_for_sensor_config(w, SENSOR_CONFIG) for w in ex_wins]
            )
            exercise_acc_windows[s_id] = [w.get("acc_window") for w in ex_wins]

    print("Pre-computation complete.")

    # Determine already-completed folds (resume support)
    done_folds: set = set()
    fold_rows_existing: List[Dict[str, Any]] = []
    if out_folds.exists():
        try:
            prev = pd.read_csv(out_folds)
            done_folds = set(zip(prev["fold_idx"], prev["experiment"]))
            fold_rows_existing = prev.to_dict("records")
            print(f"Resuming: {len(done_folds) // len(EXPERIMENT_CONFIGS)} folds already done.")
        except Exception as e:
            print(f"Could not load previous fold results: {e}")

    # Main LOSO loop
    print(f"\n[Phase 3] Running {len(manifest)} × {len(EXPERIMENT_CONFIGS)} experiment folds...")
    fold_rows: List[Dict[str, Any]] = list(fold_rows_existing)
    t0 = time.time()

    for fold in manifest:
        fold_idx = fold["fold_idx"]
        test_subj = fold["test_subject"]

        # Skip if all 4 configs already done for this fold
        if all((fold_idx, e) in done_folds for e in EXPERIMENT_CONFIGS):
            print(f"  Fold {fold_idx:02d} ({test_subj}): ALREADY DONE — skipped")
            continue

        fold_results = run_fold(
            fold=fold,
            subject_features=subject_features,
            subject_labels=subject_labels,
            subject_acc_windows=subject_acc_windows,
            exercise_features=exercise_features,
            exercise_acc_windows=exercise_acc_windows,
            subject_exercise_windows=subject_exercise_windows,
            seed=42,
        )

        for exp_name, res in fold_results.items():
            row = {
                "fold_idx": fold_idx,
                "test_subject": test_subj,
                "experiment": exp_name,
                "macro_f1": round(res["macro_f1"], 4),
                "balanced_acc": round(res["balanced_acc"], 4),
                "f1_rest": round(res["f1_rest"], 4),
                "f1_stress": round(res["f1_stress"], 4),
                "exercise_far": round(res.get("exercise_far", float("nan")), 4) if not np.isnan(res.get("exercise_far", float("nan"))) else float("nan"),
                "n_exercise_windows": res.get("n_exercise_windows", 0),
            }
            fold_rows.append(row)

        best_f1 = max(fold_results[e]["macro_f1"] for e in fold_results)
        print(
            f"  Fold {fold_idx:02d} ({test_subj}): "
            + " | ".join(
                f"{e}={fold_results[e]['macro_f1']:.4f}"
                for e in EXPERIMENT_CONFIGS
            )
            + f"  (best={best_f1:.4f})"
        )

        # Save intermediate results
        pd.DataFrame(fold_rows).to_csv(out_folds, index=False)

    elapsed = time.time() - t0
    print(f"\n[Phase 3] Completed in {elapsed:.1f}s")

    # ---------------------------------------------------------------------------
    # Aggregate summary
    # ---------------------------------------------------------------------------
    print("\n[Phase 4] Computing per-experiment summary statistics...")
    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out_folds, index=False)

    summary_rows: List[Dict[str, Any]] = []
    for exp_name in EXPERIMENT_CONFIGS:
        sub = folds_df[folds_df["experiment"] == exp_name]
        macro_f1_mean = sub["macro_f1"].mean()
        macro_f1_std = sub["macro_f1"].std()
        ex_far_vals = sub["exercise_far"].dropna()
        ex_far_mean = ex_far_vals.mean() if len(ex_far_vals) > 0 else float("nan")

        summary_rows.append({
            "experiment": exp_name,
            "description": EXPERIMENT_CONFIGS[exp_name].description,
            "macro_f1_mean": round(macro_f1_mean, 4),
            "macro_f1_std": round(macro_f1_std, 4),
            "balanced_acc_mean": round(sub["balanced_acc"].mean(), 4),
            "f1_rest_mean": round(sub["f1_rest"].mean(), 4),
            "f1_stress_mean": round(sub["f1_stress"].mean(), 4),
            "exercise_far_mean": round(ex_far_mean, 4) if not np.isnan(ex_far_mean) else float("nan"),
            "n_folds": len(sub),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_summary, index=False)

    print("\n" + "=" * 70)
    print("RESEARCH EXPERIMENT RESULTS")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nResults saved:")
    print(f"  Summary: {out_summary}")
    print(f"  Per-fold: {out_folds}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run H1/H2 contextualization research experiments (LOSO evaluation)."
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data root directory (default: auto-detected via ProjectConfig).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_research_experiments(data_dir=args.data_dir)

