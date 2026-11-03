"""
Scientific Visualization Generator for 3×3 Factorial Experiments
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Generates publication-quality figures:
1. Macro-F1 Heatmap (3 Models × 3 Sensor Sets)
2. Balanced Accuracy Heatmap (3 Models × 3 Sensor Sets)
3. Exercise False-Stress Rate (FAR) Heatmap
4. Model-by-Sensor Interaction Plot
5. Per-Subject Performance Distribution Boxplots
6. Key Confusion Matrices (REST vs STRESS)
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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import default_config

PROJECT_ROOT = str(default_config.project_dir)
RESULTS_DIR = str(default_config.results_dir)
METRICS_DIR = str(default_config.metrics_dir)
FIGURES_DIR = str(default_config.figures_dir)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300
})


def generate_all_figures():
    factorial_csv = os.path.join(METRICS_DIR, "factorial_metrics.csv")
    subject_csv = os.path.join(METRICS_DIR, "subject_level_metrics.csv")
    interaction_json = os.path.join(METRICS_DIR, "interaction_analysis.json")

    if not os.path.exists(factorial_csv):
        print(f"⚠️ Metrics file not found at: {factorial_csv}")
        print("To generate metrics, run: make train (or python3 src/experiments/run_experiments.py)")
        print("Or set WEARABLE_DATA_DIR to the directory containing results/metrics/.")
        return

    df_factorial = pd.read_csv(factorial_csv)
    df_subject = pd.read_csv(subject_csv)

    # 1. Macro-F1 Heatmap
    pivot_f1 = df_factorial.groupby(["model", "sensor_config"])["macro_f1_subject_mean"].mean().unstack()
    # Order rows and columns explicitly
    models_order = ["Majority", "RandomForest", "1D-CNN"]
    sensors_order = ["EDA", "PPG", "EDA+PPG"]
    pivot_f1 = pivot_f1.reindex(index=models_order, columns=sensors_order)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="Blues", cbar_kws={"label": "Subject-Mean Macro-F1"}, ax=ax, vmin=0.4, vmax=0.85)
    ax.set_title("Primary Task: REST vs. STRESS Macro-F1 (LOSO)")
    ax.set_xlabel("Sensor Configuration")
    ax.set_ylabel("Classification Model")
    plt.tight_layout()
    f1_path = os.path.join(FIGURES_DIR, "heatmap_macro_f1.png")
    fig.savefig(f1_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {f1_path}")

    # 2. Balanced Accuracy Heatmap
    pivot_acc = df_factorial.groupby(["model", "sensor_config"])["balanced_acc"].mean().unstack()
    pivot_acc = pivot_acc.reindex(index=models_order, columns=sensors_order)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(pivot_acc, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Balanced Accuracy"}, ax=ax, vmin=0.45, vmax=0.85)
    ax.set_title("Primary Task: Balanced Accuracy (LOSO)")
    ax.set_xlabel("Sensor Configuration")
    ax.set_ylabel("Classification Model")
    plt.tight_layout()
    acc_path = os.path.join(FIGURES_DIR, "heatmap_balanced_acc.png")
    fig.savefig(acc_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {acc_path}")

    # 3. Exercise False-Stress Rate Heatmap
    pivot_far = df_factorial.groupby(["model", "sensor_config"])["exercise_far"].mean().unstack()
    pivot_far = pivot_far.reindex(index=models_order, columns=sensors_order)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.heatmap(pivot_far, annot=True, fmt=".3f", cmap="Reds", cbar_kws={"label": "Exercise False-Stress Rate"}, ax=ax, vmin=0.0, vmax=1.0)
    ax.set_title("Exercise Confounding: False-Stress Rate (FAR)")
    ax.set_xlabel("Sensor Configuration")
    ax.set_ylabel("Classification Model")
    plt.tight_layout()
    far_path = os.path.join(FIGURES_DIR, "heatmap_exercise_far.png")
    fig.savefig(far_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {far_path}")

    # 4. Model-by-Sensor Interaction Plot
    fig, ax = plt.subplots(figsize=(7, 4.5))
    agg_df = df_factorial.groupby(["model", "sensor_config"])[["macro_f1_subject_mean", "macro_f1_ci_lower", "macro_f1_ci_upper"]].mean().reset_index()
    
    markers = {"Majority": "o", "RandomForest": "s", "1D-CNN": "^"}
    colors = {"Majority": "#7f7f7f", "RandomForest": "#2ca02c", "1D-CNN": "#1f77b4"}

    for model in models_order:
        m_df = agg_df[agg_df["model"] == model].set_index("sensor_config").reindex(sensors_order).reset_index()
        y = m_df["macro_f1_subject_mean"].values
        yerr_low = y - m_df["macro_f1_ci_lower"].values
        yerr_high = m_df["macro_f1_ci_upper"].values - y
        ax.errorbar(
            sensors_order, y,
            yerr=[np.maximum(0, yerr_low), np.maximum(0, yerr_high)],
            label=model,
            marker=markers[model],
            color=colors[model],
            linewidth=2,
            capsize=5,
            markersize=7
        )

    ax.set_title("Model × Sensor Interaction: Macro-F1 across Sensor Sets")
    ax.set_xlabel("Available Sensor Configuration")
    ax.set_ylabel("Subject-Mean Macro-F1 (with 95% Bootstrap CI)")
    ax.legend(title="Model", frameon=True)
    ax.set_ylim(0.35, 0.90)
    plt.tight_layout()
    interaction_path = os.path.join(FIGURES_DIR, "model_sensor_interaction.png")
    fig.savefig(interaction_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {interaction_path}")

    # 5. Per-Subject Performance Distribution Boxplots
    fig, ax = plt.subplots(figsize=(9, 5))
    # Filter for non-majority models to see RF and CNN across sensors
    rf_cnn_sub = df_subject[df_subject["model"].isin(["RandomForest", "1D-CNN"])].copy()
    rf_cnn_sub["Configuration"] = rf_cnn_sub["model"] + " (" + rf_cnn_sub["sensor_config"] + ")"
    order = [
        "RandomForest (EDA)", "RandomForest (PPG)", "RandomForest (EDA+PPG)",
        "1D-CNN (EDA)", "1D-CNN (PPG)", "1D-CNN (EDA+PPG)"
    ]
    palette = ["#98df8a", "#2ca02c", "#006400", "#aec7e8", "#1f77b4", "#08306b"]
    sns.boxplot(x="Configuration", y="macro_f1", data=rf_cnn_sub, order=order, palette=palette, ax=ax)
    sns.stripplot(x="Configuration", y="macro_f1", data=rf_cnn_sub, order=order, color="black", alpha=0.35, size=4, jitter=0.2, ax=ax)
    ax.set_title("Cross-Subject Macro-F1 Distribution (LOSO Folds, N=34)")
    ax.set_xlabel("Model & Sensor Configuration")
    ax.set_ylabel("Macro-F1 per Held-out Subject")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    dist_path = os.path.join(FIGURES_DIR, "per_subject_distribution.png")
    fig.savefig(dist_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {dist_path}")

    # 6. Confusion Matrices (RF vs CNN for EDA+PPG Fusion)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    classes = ["REST", "STRESS"]
    
    # Calculate pooled normalized CM from precision/recall in df_factorial
    # RF EDA+PPG
    rf_fusion = df_factorial[(df_factorial["model"] == "RandomForest") & (df_factorial["sensor_config"] == "EDA+PPG")].iloc[0]
    rec_rest_rf = rf_fusion["recall_rest"]
    rec_stress_rf = rf_fusion["recall_stress"]
    cm_rf = np.array([[rec_rest_rf, 1.0 - rec_rest_rf], [1.0 - rec_stress_rf, rec_stress_rf]])

    sns.heatmap(cm_rf, annot=True, fmt=".2%", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=axes[0], cbar=False)
    axes[0].set_title(f"Random Forest (EDA+PPG)\nMacro-F1 = {rf_fusion['macro_f1_subject_mean']:.3f}")
    axes[0].set_xlabel("Predicted Condition")
    axes[0].set_ylabel("True Condition")

    # CNN EDA+PPG
    cnn_fusion = df_factorial[(df_factorial["model"] == "1D-CNN") & (df_factorial["sensor_config"] == "EDA+PPG")].iloc[0]
    rec_rest_cnn = cnn_fusion["recall_rest"]
    rec_stress_cnn = cnn_fusion["recall_stress"]
    cm_cnn = np.array([[rec_rest_cnn, 1.0 - rec_rest_cnn], [1.0 - rec_stress_cnn, rec_stress_cnn]])

    sns.heatmap(cm_cnn, annot=True, fmt=".2%", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axes[1], cbar=False)
    axes[1].set_title(f"1D-CNN (EDA+PPG)\nMacro-F1 = {cnn_fusion['macro_f1_subject_mean']:.3f}")
    axes[1].set_xlabel("Predicted Condition")
    axes[1].set_ylabel("True Condition")

    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {cm_path}")

    print("\nAll publication figures successfully created in results/figures/!")


if __name__ == "__main__":
    generate_all_figures()
