# Subsystems Architecture Deep Dives

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

This directory contains technical deep-dive architectural specifications, block diagrams, and execution flowcharts for the primary subsystems in the codebase.

---

## Subsystem Index

| Subsystem | Scope & Purpose | Primary Python Source Files | Deep Dive Document |
|---|---|---|---|
| **1. Multisensor Preprocessing** | Zero-phase Butterworth filtering (EDA low-pass, BVP band-pass), cvxEDA tonic/phasic decomposition, and protocol-aligned 30s windowing | `src/preprocessing/filters.py`<br/>`src/preprocessing/signal_loader.py`<br/>`src/preprocessing/segmentation.py` | [multisensor_preprocessing.md](multisensor_preprocessing.md) |
| **2. Feature Extraction Pipeline** | 20D physiological feature extraction (10 EDA, 10 PPG) and 7D kinematic acceleration motion intensity feature calculation | `src/features/feature_extractor.py`<br/>`src/research/acc_aware/acc_features.py` | [feature_extraction_pipeline.md](feature_extraction_pipeline.md) |
| **3. RF & CNN Training Branches** | Dual model training pathways: Tabular Random Forest with balanced subsampling vs. Temporal 1D-CNN with PyTorch early stopping | `src/models/random_forest.py`<br/>`src/models/cnn1d.py`<br/>`src/models/tensor_builder.py` | [rf_cnn_training_branches.md](rf_cnn_training_branches.md) |
| **4. ACC-Aware Contextualization** | Hypothesis 1 (H1): Motion gating via dynamic acceleration threshold calibration (25th percentile of training exercise) | `src/research/acc_aware/movement_context.py`<br/>`src/research/acc_aware/acc_features.py` | [acc_aware_contextualization.md](acc_aware_contextualization.md) |
| **5. Similarity-Aware KNN Retrieval** | Hypothesis 2 (H2): Non-parametric $k$-NN exemplar retrieval ($k=7$) over historical REST, STRESS, and EXERCISE feature vectors | `src/research/similarity_aware/knn_retrieval.py`<br/>`src/research/similarity_aware/similarity_context.py` | [similarity_aware_retrieval.md](similarity_aware_retrieval.md) |
| **6. Decision Fusion Policies** | Combinatorial evaluation matrix orchestrating E0 (baseline), E1 (H1), E2 (H2), and E3 (combined sequential) configurations | `src/research/fusion/fusion_policy.py` | [decision_fusion.md](decision_fusion.md) |
| **7. Nested LOSO Evaluation** | Outer subject isolation (34 folds), inner validation splits, zero-leakage scaling, and bootstrap confidence intervals | `src/evaluation/splits.py`<br/>`src/evaluation/metrics.py`<br/>`src/research/run_research_experiments.py` | [nested_loso_evaluation.md](nested_loso_evaluation.md) |
| **8. Streamlit Sense-Think-Act Engine** | Sliding window ring buffer, live inference, and sub-millisecond deterministic rule engine vs. offline LLM recommendations | `src/act/sensor_adapter.py`<br/>`src/act/dashboard.py`<br/>`src/act/act_evaluator.py` | [streamlit_sense_think_act.md](streamlit_sense_think_act.md) |

---

## Navigating the Subsystems

Each subsystem document provides:
1. Mathematical formulations and algorithmic specifications.
2. An engineering Block Diagram showing inputs, internal processing steps, and outputs.
3. An execution Flowchart with conventional engineering logic (decisions, loops, fallback paths).
4. Concrete code references (classes, methods, arguments).
