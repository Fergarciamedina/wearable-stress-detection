"""
Factorial Experiments Execution Engine (3 Models × 3 Sensor Sets)
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Executes the complete 3×3 factorial experiment matrix under nested subject-disjoint LOSO:
Factor 1 (Models): Majority Class, Random Forest, Small 1D-CNN.
Factor 2 (Sensors): EDA, PPG, EDA+PPG (Fusion).
Seeds: 42, 123, 999.
Saves:
- results/metrics/factorial_metrics.csv
- results/metrics/subject_level_metrics.csv
- results/metrics/exercise_far_metrics.csv
- results/metrics/interaction_analysis.json
- results/predictions/ (fold prediction logs)
- results/checkpoints/ (deployment model checkpoints)
"""

import os
import sys
from pathlib import Path

# Ensure root containing 'src' is in sys.path
_curr = Path(__file__).resolve()
for _p in [_curr.parent, _curr.parent.parent, _curr.parent.parent.parent]:
    if (_p / "src").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import json
import time
import pickle
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.preprocessing.signal_loader import SignalLoader
from src.preprocessing.segmentation import WindowSegmenter
from src.features.feature_extractor import FeatureExtractor
from src.models.tensor_builder import TensorBuilder
from src.models.cnn1d import Small1DCNN, CNNTrainer, set_seed
from src.evaluation.splits import generate_loso_manifest, save_splits_manifest
from src.evaluation.metrics import (
    compute_binary_metrics,
    compute_clustered_bootstrap_ci,
    compute_paired_difference_ci,
    compute_exercise_far
)

from src.config import default_config

PROJECT_ROOT = str(default_config.project_dir)
DATASET_ROOT = str(default_config.raw_data_dir)
RESULTS_DIR = str(default_config.results_dir)
METRICS_DIR = str(default_config.metrics_dir)
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
CHECKPOINTS_DIR = str(default_config.checkpoints_dir)
SPLITS_DIR = str(default_config.splits_dir)

MODELS = ["Majority", "RandomForest", "1D-CNN"]
SENSOR_CONFIGS = ["EDA", "PPG", "EDA+PPG"]
SEEDS = [42, 123, 999]


def run_full_evaluation():
    print("=" * 70)
    print("Starting COEN 6322 Factorial Research Experiment Engine")
    print("=" * 70)
    
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    loader = SignalLoader(DATASET_ROOT)
    segmenter = WindowSegmenter(window_len_s=30.0, stride_s=15.0)
    feat_extractor = FeatureExtractor(eda_fs=4.0, bvp_fs=64.0)
    tensor_builder = TensorBuilder(target_fs=16.0, window_len_s=30.0)

    # 1. Load and segment all eligible subjects
    # S02 excluded due to duplicated records; f07 excluded for PPG due to dock cover
    all_candidate_ids = [f"S{i:02d}" for i in range(1, 19) if i != 2] + [f"f{i:02d}" for i in range(1, 19)]
    
    print("\n[Phase 1] Ingesting and segmenting all subject sessions...")
    subject_stress_windows = {}
    subject_exercise_windows = {}
    eligible_paired_subjects = []

    for s_id in all_candidate_ids:
        stress_sess = loader.load_subject_stress_session(s_id)
        if stress_sess is not None:
            w_stress = segmenter.segment_stress_session(stress_sess)
            if len(w_stress) > 0:
                subject_stress_windows[s_id] = w_stress
                # Check if subject has valid BVP (exclude f07 for paired cohort)
                if stress_sess["bvp_valid"] and s_id != "f07":
                    eligible_paired_subjects.append(s_id)

        # Exercise challenge sessions
        aero_sess = loader.load_subject_exercise_session(s_id, "AEROBIC")
        anaero_sess = loader.load_subject_exercise_session(s_id, "ANAEROBIC")
        ex_wins = []
        if aero_sess is not None:
            ex_wins.extend(segmenter.segment_exercise_session(aero_sess))
        if anaero_sess is not None:
            ex_wins.extend(segmenter.segment_exercise_session(anaero_sess))
        if len(ex_wins) > 0:
            subject_exercise_windows[s_id] = ex_wins

    print(f"Loaded {len(subject_stress_windows)} eligible stress subjects.")
    print(f"Paired evaluation cohort (both EDA and PPG valid): {len(eligible_paired_subjects)} subjects.")
    print(f"Loaded exercise challenge recordings for {len(subject_exercise_windows)} subjects.")

    # 2. Generate and persist nested LOSO splits manifest
    splits_path = os.path.join(SPLITS_DIR, "loso_splits.json")
    manifest = generate_loso_manifest(eligible_paired_subjects, inner_val_ratio=0.2, seed=42)
    save_splits_manifest(manifest, splits_path)
    print(f"Persisted {len(manifest)} outer LOSO folds to {splits_path}")

    print("\n[Phase 1.5] Precomputing subject features and tensors for all sensor configurations...")
    subject_features = {cfg: {} for cfg in SENSOR_CONFIGS}
    subject_tensors = {cfg: {} for cfg in SENSOR_CONFIGS}
    subject_labels = {}
    exercise_features = {cfg: {} for cfg in SENSOR_CONFIGS}
    exercise_tensors = {cfg: {} for cfg in SENSOR_CONFIGS}

    for s_id in eligible_paired_subjects:
        wins = subject_stress_windows[s_id]
        subject_labels[s_id] = np.array([w["label"] for w in wins], dtype=int)
        for cfg in SENSOR_CONFIGS:
            subject_features[cfg][s_id] = np.array([feat_extractor.extract_features_for_sensor_config(w, cfg) for w in wins])
            subject_tensors[cfg][s_id] = np.array([tensor_builder.build_tensor(w, cfg) for w in wins])

        if s_id in subject_exercise_windows:
            ex_wins = subject_exercise_windows[s_id]
            for cfg in SENSOR_CONFIGS:
                exercise_features[cfg][s_id] = np.array([feat_extractor.extract_features_for_sensor_config(w, cfg) for w in ex_wins])
                exercise_tensors[cfg][s_id] = np.array([tensor_builder.build_tensor(w, cfg) for w in ex_wins])

    print("Precomputation completed successfully.")

    # Data structures for results with resume capability
    factorial_csv_path = os.path.join(METRICS_DIR, "factorial_metrics.csv")
    subject_csv_path = os.path.join(METRICS_DIR, "subject_level_metrics.csv")
    ex_csv_path = os.path.join(METRICS_DIR, "exercise_far_metrics.csv")

    summary_results = []
    subject_results = []
    exercise_far_results = []
    done_keys = set()

    if os.path.exists(factorial_csv_path) and os.path.exists(subject_csv_path) and os.path.exists(ex_csv_path):
        try:
            summary_results = pd.read_csv(factorial_csv_path).to_dict("records")
            subject_results = pd.read_csv(subject_csv_path).to_dict("records")
            exercise_far_results = pd.read_csv(ex_csv_path).to_dict("records")
            done_keys = {(r["model"], r["sensor_config"], int(r["seed"])) for r in summary_results}
            print(f"Found existing progress: {len(done_keys)} experiments already completed and loaded.")
        except Exception as e:
            print(f"Notice: Could not load partial results ({e}), starting clean.")

    # 3. Factorial execution loop
    total_experiments = len(MODELS) * len(SENSOR_CONFIGS) * len(SEEDS)
    exp_counter = 0

    for sensor_config in SENSOR_CONFIGS:
        for model_name in MODELS:
            for seed in SEEDS:
                exp_counter += 1
                exp_id = f"{model_name}_{sensor_config}_seed{seed}"
                
                if (model_name, sensor_config, seed) in done_keys:
                    print(f"[{exp_counter}/{total_experiments}] Skipping already completed: {exp_id}")
                    continue

                print(f"\n[{exp_counter}/{total_experiments}] Executing: Model={model_name}, Sensor={sensor_config}, Seed={seed}")
                set_seed(seed)

                y_true_all = []
                y_pred_all = []
                subject_scores = []
                exercise_pred_all = []

                t_start = time.time()

                for fold in manifest:
                    test_subj = fold["test_subject"]
                    train_subjs = fold["train_subjects"]
                    val_subjs = fold["val_subjects"]

                    # Labels
                    y_train = np.concatenate([subject_labels[s] for s in train_subjs], axis=0)
                    y_val = np.concatenate([subject_labels[s] for s in val_subjs], axis=0)
                    y_test = subject_labels[test_subj]

                    if model_name in ["Majority", "RandomForest"]:
                        X_train = np.concatenate([subject_features[sensor_config][s] for s in train_subjs], axis=0)
                        X_val = np.concatenate([subject_features[sensor_config][s] for s in val_subjs], axis=0)
                        X_test = subject_features[sensor_config][test_subj]

                        # Train-only standard scaling (strict zero leakage)
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_val_scaled = scaler.transform(X_val)
                        X_test_scaled = scaler.transform(X_test)

                        if model_name == "Majority":
                            clf = DummyClassifier(strategy="most_frequent")
                            clf.fit(X_train_scaled, y_train)
                        else:
                            clf = RandomForestClassifier(
                                n_estimators=100,
                                max_depth=8,
                                min_samples_split=4,
                                class_weight="balanced",
                                random_state=seed,
                                n_jobs=-1
                            )
                            clf.fit(X_train_scaled, y_train)

                        y_test_pred = clf.predict(X_test_scaled)

                        # Evaluate on held-out exercise challenge if available for test subject
                        if test_subj in exercise_features[sensor_config]:
                            X_ex = exercise_features[sensor_config][test_subj]
                            X_ex_scaled = scaler.transform(X_ex)
                            ex_pred = clf.predict(X_ex_scaled)
                            exercise_pred_all.extend(ex_pred)

                    elif model_name == "1D-CNN":
                        X_train_t = np.concatenate([subject_tensors[sensor_config][s] for s in train_subjs], axis=0)
                        X_val_t = np.concatenate([subject_tensors[sensor_config][s] for s in val_subjs], axis=0)
                        X_test_t = subject_tensors[sensor_config][test_subj]

                        # Standardize waveforms using training mean and std per channel (zero leakage)
                        train_mean = np.mean(X_train_t, axis=(0, 2), keepdims=True)
                        train_std = np.std(X_train_t, axis=(0, 2), keepdims=True) + 1e-6
                        
                        X_train_t = (X_train_t - train_mean) / train_std
                        X_val_t = (X_val_t - train_mean) / train_std
                        X_test_t = (X_test_t - train_mean) / train_std

                        in_channels = 1 if sensor_config in ["EDA", "PPG"] else 2
                        trainer = CNNTrainer(
                            in_channels=in_channels,
                            lr=1e-3,
                            batch_size=64,
                            max_epochs=6,
                            patience=2,
                            device="cpu"
                        )
                        trainer.fit(X_train_t, y_train, X_val_t, y_val)
                        y_test_pred = trainer.predict(X_test_t)

                        # Evaluate on held-out exercise
                        if test_subj in exercise_tensors[sensor_config]:
                            X_ex_t = exercise_tensors[sensor_config][test_subj]
                            X_ex_t = (X_ex_t - train_mean) / train_std
                            ex_pred = trainer.predict(X_ex_t)
                            exercise_pred_all.extend(ex_pred)

                    # Compute fold metrics
                    fold_metrics = compute_binary_metrics(y_test, y_test_pred)
                    subject_scores.append(fold_metrics["macro_f1"])

                    subject_results.append({
                        "experiment_id": exp_id,
                        "model": model_name,
                        "sensor_config": sensor_config,
                        "seed": seed,
                        "test_subject": test_subj,
                        "macro_f1": fold_metrics["macro_f1"],
                        "balanced_acc": fold_metrics["balanced_acc"],
                        "f1_rest": fold_metrics["f1_rest"],
                        "f1_stress": fold_metrics["f1_stress"]
                    })

                    y_true_all.extend(y_test)
                    y_pred_all.extend(y_test_pred)

                elapsed_time = time.time() - t_start

                # Overall pooled metrics
                overall_metrics = compute_binary_metrics(np.array(y_true_all), np.array(y_pred_all))
                
                # Subject-clustered bootstrap 95% CI
                mean_f1, ci_low_f1, ci_high_f1 = compute_clustered_bootstrap_ci(subject_scores, seed=seed)

                # Exercise FAR
                far_res = compute_exercise_far(np.array(exercise_pred_all))

                summary_results.append({
                    "model": model_name,
                    "sensor_config": sensor_config,
                    "seed": seed,
                    "macro_f1_pooled": overall_metrics["macro_f1"],
                    "macro_f1_subject_mean": mean_f1,
                    "macro_f1_ci_lower": ci_low_f1,
                    "macro_f1_ci_upper": ci_high_f1,
                    "balanced_acc": overall_metrics["balanced_acc"],
                    "precision_rest": overall_metrics["precision_rest"],
                    "recall_rest": overall_metrics["recall_rest"],
                    "f1_rest": overall_metrics["f1_rest"],
                    "precision_stress": overall_metrics["precision_stress"],
                    "recall_stress": overall_metrics["recall_stress"],
                    "f1_stress": overall_metrics["f1_stress"],
                    "exercise_far": far_res["exercise_far"],
                    "exercise_total_windows": far_res["total_windows"],
                    "exercise_false_stress": far_res["false_stress_windows"],
                    "runtime_seconds": round(elapsed_time, 2)
                })

                exercise_far_results.append({
                    "model": model_name,
                    "sensor_config": sensor_config,
                    "seed": seed,
                    "exercise_far": far_res["exercise_far"],
                    "total_windows": far_res["total_windows"],
                    "false_stress_windows": far_res["false_stress_windows"]
                })

                print(f"Result: Macro-F1={mean_f1:.4f} [95% CI: {ci_low_f1:.4f}, {ci_high_f1:.4f}], BalAcc={overall_metrics['balanced_acc']:.4f}, Exercise FAR={far_res['exercise_far']:.4f}")

                # Save intermediate progress immediately to disk
                pd.DataFrame(summary_results).to_csv(factorial_csv_path, index=False)
                pd.DataFrame(subject_results).to_csv(subject_csv_path, index=False)
                pd.DataFrame(exercise_far_results).to_csv(ex_csv_path, index=False)

    # 4. Save final metrics tables
    summary_df = pd.DataFrame(summary_results)
    summary_df.to_csv(os.path.join(METRICS_DIR, "factorial_metrics.csv"), index=False)

    subject_df = pd.DataFrame(subject_results)
    subject_df.to_csv(os.path.join(METRICS_DIR, "subject_level_metrics.csv"), index=False)

    ex_df = pd.DataFrame(exercise_far_results)
    ex_df.to_csv(os.path.join(METRICS_DIR, "exercise_far_metrics.csv"), index=False)

    # 5. Compute Model × Sensor Interaction Analysis and Paired Differences
    print("\n[Phase 2] Computing Model × Sensor Interaction Analysis...")
    interaction_data = {}
    
    # Aggregate across seeds
    agg_df = summary_df.groupby(["model", "sensor_config"])[["macro_f1_subject_mean", "balanced_acc", "exercise_far"]].mean().reset_index()
    
    # RF vs CNN differences per sensor configuration
    for sensor in SENSOR_CONFIGS:
        rf_score = agg_df[(agg_df["model"] == "RandomForest") & (agg_df["sensor_config"] == sensor)]["macro_f1_subject_mean"].values[0]
        cnn_score = agg_df[(agg_df["model"] == "1D-CNN") & (agg_df["sensor_config"] == sensor)]["macro_f1_subject_mean"].values[0]
        diff = cnn_score - rf_score
        interaction_data[f"RF_vs_CNN_diff_{sensor}"] = round(float(diff), 4)

    # Interaction contrast: (CNN - RF in EDA+PPG) - (CNN - RF in EDA)
    contrast = interaction_data["RF_vs_CNN_diff_EDA+PPG"] - interaction_data["RF_vs_CNN_diff_EDA"]
    interaction_data["interaction_contrast_fusion_vs_eda"] = round(float(contrast), 4)
    
    with open(os.path.join(METRICS_DIR, "interaction_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(interaction_data, f, indent=2)

    # 6. Fit and Save Final Deployable Models (Trained on entire eligible public dataset)
    print("\n[Phase 3] Fitting and serializing deployment model checkpoints...")
    all_public_wins = []
    for s_id in eligible_paired_subjects:
        all_public_wins.extend(subject_stress_windows[s_id])

    y_all_public = np.array([w["label"] for w in all_public_wins], dtype=int)

    for sensor_config in SENSOR_CONFIGS:
        # Fit deployment Random Forest
        X_all_rf = np.array([feat_extractor.extract_features_for_sensor_config(w, sensor_config) for w in all_public_wins])
        scaler_deploy = StandardScaler()
        X_all_rf_scaled = scaler_deploy.fit_transform(X_all_rf)

        rf_deploy = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=4, class_weight="balanced", random_state=42)
        rf_deploy.fit(X_all_rf_scaled, y_all_public)

        rf_ckpt_path = os.path.join(CHECKPOINTS_DIR, f"deploy_RF_{sensor_config}.pkl")
        with open(rf_ckpt_path, "wb") as f:
            pickle.dump({"model": rf_deploy, "scaler": scaler_deploy, "sensor_config": sensor_config, "features_dim": X_all_rf.shape[1]}, f)

        # Fit deployment 1D-CNN
        X_all_cnn = np.array([tensor_builder.build_tensor(w, sensor_config) for w in all_public_wins])
        mean_dep = np.mean(X_all_cnn, axis=(0, 2), keepdims=True)
        std_dep = np.std(X_all_cnn, axis=(0, 2), keepdims=True) + 1e-6
        X_all_cnn_norm = (X_all_cnn - mean_dep) / std_dep

        in_ch = 1 if sensor_config in ["EDA", "PPG"] else 2
        cnn_deploy = CNNTrainer(in_channels=in_ch, batch_size=64, max_epochs=15, device="cpu")
        cnn_deploy.fit(X_all_cnn_norm, y_all_public, X_all_cnn_norm, y_all_public)

        cnn_ckpt_path = os.path.join(CHECKPOINTS_DIR, f"deploy_CNN_{sensor_config}.pt")
        torch.save({
            "model_state_dict": cnn_deploy.model.state_dict(),
            "in_channels": in_ch,
            "mean": mean_dep,
            "std": std_dep,
            "sensor_config": sensor_config
        }, cnn_ckpt_path)

    print(f"\nAll 9 Factorial experiments and deployment checkpoints saved successfully to {RESULTS_DIR}!")


if __name__ == "__main__":
    run_full_evaluation()
