# Research Extension: Context-Aware Wearable Stress Detection

COEN 6322: AI for Wearable Health Applications, Fall 2026

## Overview

Wearable cognitive stress classifiers trained on autonomic arousal (EDA and PPG) exhibit severe vulnerability to **false stress alarms** during physical exercise. Because both physical activity and psychological stress trigger sympathetic nervous system activation (elevated heart rate, reduced heart rate variability, increased skin conductance), unimodal or multimodal autonomic classifiers cannot distinguish cognitive stress from physical exertion.

In the baseline benchmark on PhysioNet DS1 ($N=34$), Random Forest achieves **Macro-F1 = 0.6481**, but triggers false alarms on **61.58%** of 7,048 exercise windows.

This research package investigates two contextualization hypotheses to resolve the exercise confound under strict 34-fold Leave-One-Subject-Out (LOSO) cross-validation:

1. **H1 — ACC-Aware Movement Gating (`acc_aware/`):** Accelerometer movement features detect gross physical activity, suppressing false stress alarms when movement intensity exceeds the 25th percentile of training exercise windows.
2. **H2 — Similarity-Aware Physiological Retrieval (`similarity_aware/`):** An exact $k$-Nearest-Neighbors index ($k=7$) over historical training exemplars (REST, STRESS, EXERCISE) provides non-parametric evidence of whether current autonomic signals match exercise patterns.
3. **Fusion Policy (`fusion/`):** Sequential evaluation matrix covering E0 (Baseline), E1 (H1 only), E2 (H2 only), and E3 (Combined H1+H2).

---

## Empirical Benchmark Results (34-Fold LOSO)

| Exp | Description | Macro-F1 (Mean ± SD) | Balanced Acc | Rest F1 | Stress F1 | Exercise FAR |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **E0** | Baseline RF (EDA+PPG, uncontextualized) | 0.6481 ± 0.1570 | 68.94% | 0.8303 | 0.4660 | 61.58% |
| **E1** | **RF + H1 (ACC-Aware Movement Gate)** | **0.6528 ± 0.1528** | **68.81%** | **0.8430** | **0.4627** | **15.58%** |
| **E2** | RF + H2 (KNN Physiological Similarity) | 0.5226 ± 0.0849 | 54.18% | 0.8669 | 0.1784 | 2.79% |
| **E3** | RF + H1 + H2 (Combined Sequential) | 0.5228 ± 0.0848 | 54.21% | 0.8671 | 0.1785 | 0.98% |

### Key Scientific Insights

1. **H1 Confirmed (Pareto-Optimal):** Adding 3-axis ACC dynamic acceleration gating slashes exercise false alarms from **61.58% down to 15.58%** (a **74.7% relative reduction**) while maintaining stress detection accuracy (Macro-F1 **0.6528 vs 0.6481**).
2. **H2 Autonomic Overlap:** The $k$-NN physiological similarity index successfully eliminates almost all exercise false alarms (FAR down to 2.79%), but because autonomic features of intense cognitive stress overlap with moderate exercise, it over-suppresses true stress alarms, reducing Stress F1 from 0.466 to 0.178.
3. **Clinical Recommendation:** H1 (ACC-gating) is the recommended deployment architecture for wearable devices, providing robust motion discrimination without degrading stress sensitivity.

---

## Directory Layout

```text
src/research/
├── __init__.py
├── README.md                      # This overview & empirical synthesis
├── run_research_experiments.py     # 34-fold LOSO experiment engine
├── acc_aware/                     # H1: Accelerometer movement gating
│   ├── __init__.py
│   ├── README.md
│   ├── acc_features.py            # 7 movement features (SVM, dynamic SVM, SMA, etc.)
│   └── movement_context.py        # MovementContextualizer gate
├── similarity_aware/              # H2: Non-parametric physiological retrieval
│   ├── __init__.py
│   ├── README.md
│   ├── knn_retrieval.py           # Exact k-NN index (k=7) over REST/STRESS/EXERCISE
│   └── similarity_context.py     # SimilarityContextualizer
└── fusion/                        # Decision fusion engine
    ├── __init__.py
    ├── README.md
    └── fusion_policy.py           # E0, E1, E2, E3 configuration matrix
```

## Running the Evaluation

To reproduce the 34-fold LOSO experiments:

```bash
python3 -m src.research.run_research_experiments
```

Outputs are written to:
- `data/results/research/comparison/research_comparison.csv`
- `data/results/research/comparison/fold_predictions.csv`

