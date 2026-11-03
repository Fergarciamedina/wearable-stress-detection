# Wearable Stress Detection: Complete System Architecture & Design Index

**COEN 6322: AI for Wearable Health Applications, Fall 2026**  
**Repository:** `https://github.com/Fergarciamedina/wearable-stress-detection.git`

---

## 1. Architectural Overview & System Taxonomy

This documentation package provides an exhaustive, code-accurate technical walkthrough of the **Wearable Cognitive Stress Detection & Contextualization System**.

The architecture bridges physiological signal processing, statistical machine learning, kinematic contextualization, and interactive closed-loop actuation across **Five Canonical Engineering Stages**:

```
[STAGE 1: Data Preparation] ──► [STAGE 2: Feature Engineering] ──► [STAGE 3: Model Training]
                                                                            │
[STAGE 5: Model Inference]  ◄── [Serialized Checkpoints]       ◄── [STAGE 4: Model Evaluation]
```

---

## 2. Master System Navigation Hierarchy

```text
SYSTEM (Level 0)
├── 00-System Overview
│   ├── Level 0 Block Diagram (Five Stages)
│   └── Level 0 Macro Flowchart (End-to-End Execution)
│
├── STAGES (Level 1)
│   ├── Stage 1: Data Preparation & Preprocessing (Filters, Windowing, Exclusions)
│   ├── Stage 2: Feature Engineering & Representation (20D Physio, 7D Motion, Tensors)
│   ├── Stage 3: Model Training & Contextualization (RF, CNN, H1 Gate, H2 k-NN)
│   ├── Stage 4: Model Evaluation & Validation (34-Fold Nested LOSO, E0-E3 Benchmarks)
│   └── Stage 5: Model Inference & Application (Streamlit Sense-Think-Act Dashboard)
│
├── SUBSYSTEMS (Level 2 Deep Dives)
│   ├── Subsystem 1: Multisensor Preprocessing (Butterworth filters & boundary guard)
│   ├── Subsystem 2: Feature Extraction Pipeline (20 EDA+PPG & 7 ACC features)
│   ├── Subsystem 3: RF & CNN Training Branches (Tabular vs Temporal models)
│   ├── Subsystem 4: ACC-Aware Contextualization (H1 Motion gating & dynamic threshold)
│   ├── Subsystem 5: Similarity-Aware Retrieval (H2 Non-parametric k-NN index)
│   ├── Subsystem 6: Decision Fusion Policies (E0-E3 sequential orchestration)
│   ├── Subsystem 7: Nested LOSO Evaluation (Zero-leakage outer isolation)
│   └── Subsystem 8: Streamlit Sense-Think-Act Engine (Ring buffer & dual guidance)
│
└── VISUALIZATIONS (Exploratory Data Cloud)
    └── 2D PCA Vector Space Projection (20D EDA+PPG Space, k-NN Retrieval Illustration)
```

---

## 3. End-to-End Module & Function Mapping Table

| Stage | Subsystem / Component | Python Source File | Key Class / Function | Inputs | Outputs |
|:---:|:---|:---|:---|:---|:---|
| **1** | Signal Ingestion | `src/preprocessing/signal_loader.py` | `SignalLoader.parse_empatica_csv()`<br/>`load_subject_stress_session()` | Raw CSV files (`EDA.csv`, `BVP.csv`, `ACC.csv`, `tags.csv`) | In-memory time-aligned signal arrays |
| **1** | Digital Filtering | `src/preprocessing/filters.py` | `filter_eda()`<br/>`decompose_eda()`<br/>`filter_bvp()` | Raw 1D numpy arrays | Filtered EDA ($f_c=1.0\text{Hz}$), Tonic/Phasic, Bandpassed BVP ($0.5-8.0\text{Hz}$) |
| **1** | Window Slicing | `src/preprocessing/segmentation.py` | `WindowSegmenter.segment_stress_session()`<br/>`segment_exercise_session()` | Continuous filtered signals & labeled protocol tags | List of 30-second window dictionaries ($T_{\text{stride}} = 15.0\text{s}$) |
| **1** | Data Audit | `src/data_audit.py` | `audit_all_subjects()` | Raw folder structure | Audit summary CSVs in `data/results/data_audit/` |
| **2** | Physiological Features | `src/features/feature_extractor.py` | `FeatureExtractor.extract_eda_features()`<br/>`extract_bvp_features()` | 30s EDA (120) & BVP (1920) windows | 20-dimensional physiological vector (float32) |
| **2** | Kinematic Features | `src/research/acc_aware/acc_features.py` | `extract_acc_features()` | 30s ACC window ($960 \times 3$ @ 32Hz) | 7-dimensional motion vector (SVM, Dynamic SVM, Jerk) |
| **2** | Waveform Tensors | `src/models/tensor_builder.py` | `TensorBuilder.build_tensor()` | 30s EDA and BVP windows | PyTorch temporal tensor $(C \times 480)$ resampled to 16Hz |
| **3** | Random Forest Fitting | `src/models/random_forest.py` | `StressRandomForest.fit()` | 20D feature matrix $X_{\text{train}}$, labels $y_{\text{train}}$ | Trained RF model + fitted `StandardScaler` |
| **3** | 1D-CNN Training | `src/models/cnn1d.py` | `CNNTrainer.fit()`<br/>`Small1DCNN` | Waveform tensors $(N \times C \times 480)$ | Trained PyTorch weights, early stopping state |
| **3** | H1 Gate Calibration | `src/research/acc_aware/movement_context.py` | `MovementContextualizer.fit_threshold()` | Training subjects' exercise ACC windows | Dynamic acceleration threshold $\tau_{\text{motion}}$ (25th pct) |
| **3** | H2 Exemplar Indexing | `src/research/similarity_aware/knn_retrieval.py` | `PhysiologicalKNN.fit()` | Training REST (0), STRESS (1), EXERCISE (2) 20D vectors | Exact spatial Ball-Tree index ($k=7$) |
| **3** | Fusion Policy | `src/research/fusion/fusion_policy.py` | `FusionPolicy.fuse()` | Primary predictions, ACC windows, 20D features | Contextualized binary predictions (E0, E1, E2, E3) |
| **4** | Partitioning | `src/evaluation/splits.py` | `generate_loso_manifest()` | Eligible paired subjects list ($N=34$) | `loso_splits.json` outer/inner split manifest |
| **4** | Scoring Engine | `src/evaluation/metrics.py` | `compute_binary_metrics()`<br/>`compute_exercise_far()` | Ground truth labels, model predictions | Macro-F1, Stress F1, Rest F1, BalAcc, Exercise FAR |
| **4** | Research Evaluator | `src/research/run_research_experiments.py` | `run_research_experiments()` | 34 outer folds, precomputed features | `research_comparison.csv`, `fold_predictions.csv` |
| **5** | Sensor Adapter Buffer | `src/act/sensor_adapter.py` | `ReplaySensorAdapter.advance_by_step()`<br/>`get_latest_window()` | Real-time / Replayed biometric stream | 30-second sliding buffer context with memory pruning |
| **5** | Interactive Dashboard | `src/act/dashboard.py` | Streamlit entry point | Replay adapter, Checkpoints | Web UI: SENSE waveforms, THINK gauge, ACT cards |
| **5** | Deterministic Rules | `src/act/act_evaluator.py` | `DeterministicRuleAct.generate_recommendation()`| Structured state context dictionary | Clinical wellness action code ($< 0.1\text{ms}$ latency) |
| **5** | LLM Recommender | `src/act/act_evaluator.py` | `LLMActAdapter.generate_recommendation()` | Structured state context dictionary | Conversational empathetic guidance card |

---

## 4. Primary Stage Directories

1. **[00-System Overview](00-system-overview/README.md):** Complete system-wide block diagram and end-to-end execution flowchart.
   - [Level 4 Python Code Execution Map](00-system-overview/python_execution_map.md): In-depth call graphs and entry points.
2. **[01-Data Preparation & Preprocessing](01-data-preparation/README.md):** Raw file parsing, Butterworth signal conditioning, subject exclusions (`S02`, `f07`), and 30-second windowing.
3. **[02-Feature Engineering & Representation](02-feature-engineering/README.md):** Mathematical formulation of the 20D physiological features, 7D kinematic motion features, and $(C \times 480)$ tensors.
4. **[03-Model Training & Contextualization](03-model-training/README.md):** Supervised training of Random Forest and 1D-CNN, H1 motion threshold calibration, and H2 exact $k$-NN reference indexing.
5. **[04-Model Evaluation & Validation](04-model-evaluation/README.md):** 34-fold nested LOSO cross-validation, E0–E3 benchmark results, and Pareto trade-off analysis.
6. **[05-Model Inference & Application](05-model-inference/README.md):** Interactive Streamlit Sense–Think–Act demonstrator, streaming buffer engine, deterministic rules, and LLM advice.

---

## 5. Subsystems Deep Dives

- [Subsystem 1: Multisensor Preprocessing](subsystems/multisensor_preprocessing.md)
- [Subsystem 2: Feature Extraction Pipeline](subsystems/feature_extraction_pipeline.md)
- [Subsystem 3: RF & CNN Training Branches](subsystems/rf_cnn_training_branches.md)
- [Subsystem 4: ACC-Aware Contextualization (H1)](subsystems/acc_aware_contextualization.md)
- [Subsystem 5: Similarity-Aware Retrieval (H2)](subsystems/similarity_aware_retrieval.md)
- [Subsystem 6: Decision Fusion Policies (E0–E3)](subsystems/decision_fusion.md)
- [Subsystem 7: Nested Subject-Disjoint LOSO Evaluation](subsystems/nested_loso_evaluation.md)
- [Subsystem 8: Streamlit Sense–Think–Act Engine](subsystems/streamlit_sense_think_act.md)

---

## 6. Real Vector Cloud Visualization

- **[2D PCA Vector Space Visualization](visualizations/README.md):** 2D PCA projection of 10,414 authentic windows demonstrating the sympathetic overlap between acute stress and exercise, accompanied by an exact $k$-NN ($k=7$) spatial retrieval illustration.
