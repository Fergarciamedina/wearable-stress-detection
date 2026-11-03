# Level 0: Complete System Overview

**COEN 6322: AI for Wearable Health Applications, Fall 2026**  
*Project: Stress vs. Exercise in Wrist Wearables: Does the Model Ranking Depend on the Sensor Set?*

---

## 1. Executive System Description

The system implements a standalone, reproducible machine learning pipeline designed to detect cognitive stress from non-invasive wearable sensors (Electrodermal Activity, Photoplethysmography, and Tri-Axial Accelerometry) while explicitly addressing the severe confounding effect of physical exercise.

The system is organized into **Five Main Engineering Stages**:

1. **Stage 1 — Data Preparation & Preprocessing:** Ingestion of raw physiological streams (EDA 4 Hz, BVP 64 Hz, ACC 32 Hz), zero-phase Butterworth filtering, and strict boundary-safe 30-second window segmentation.
2. **Stage 2 — Feature Engineering & Representation:** Domain-guided extraction of 20 physiological features (10 EDA, 10 PPG), 7 tri-axial motion intensity features, and multi-channel temporal tensor resampling (16 Hz, $C \times 480$).
3. **Stage 3 — Model Training & Contextualization:** Supervised training of Random Forest and 1D-CNN architectures, dynamic acceleration threshold calibration (H1), and non-parametric physiological $k$-NN reference-index construction (H2).
4. **Stage 4 — Model Evaluation & Validation:** Strict 34-fold nested Leave-One-Subject-Out (LOSO) cross-validation evaluating baseline (E0), ACC-aware (E1), similarity-aware (E2), and combined (E3) configurations across Stress F1, Macro-F1, and Exercise False Alarm Rate (FAR).
5. **Stage 5 — Model Inference & Application:** Streaming sensor adapter buffer, real-time model inference, deterministic sub-millisecond rule actuation vs. offline LLM recommendations, and an interactive Streamlit demonstrator.

---

## 2. Macro Block Diagram (Level 0)

```mermaid
graph TB
    subgraph S1["STAGE 1: DATA PREPARATION & PREPROCESSING"]
        RAW["Raw Multimodal Data<br/>EDA (4Hz), BVP (64Hz), ACC (32Hz)<br/>PhysioNet DS1 (N=34 Cohort)"] --> LOAD["Signal Loading & Preprocessing<br/>signal_loader.py | filters.py"]
        LOAD --> SEG["30-Second Sliding Windows<br/>stride=15s (Zero Cross-Subject Leakage)<br/>segmentation.py"]
    end

    subgraph S2["STAGE 2: FEATURE ENGINEERING & REPRESENTATION"]
        SEG --> FEAT["Handcrafted Physiological Extraction<br/>10 EDA + 10 PPG = 20D Feature Vectors<br/>feature_extractor.py"]
        SEG --> ACC_FEAT["Tri-Axial Motion Extraction<br/>7 Movement Features (SVM, Jerk, SMA)<br/>acc_features.py"]
        SEG --> TENSOR["Temporal Tensor Resampling<br/>16Hz Common Grid (Channels × 480)<br/>tensor_builder.py"]
    end

    subgraph S3["STAGE 3: MODEL TRAINING & CONTEXTUALIZATION"]
        FEAT --> RF_TRAIN["Random Forest Fitting<br/>StandardScaler + 100 Trees (depth=8)<br/>random_forest.py"]
        TENSOR --> CNN_TRAIN["PyTorch 1D-CNN Fitting<br/>2 Conv Layers + GAP + Linear<br/>cnn1d.py"]
        ACC_FEAT --> H1_CALIB["H1: Movement Gate Calibration<br/>25th Percentile Dynamic SVM<br/>movement_context.py"]
        FEAT --> H2_INDEX["H2: Exact k-NN Exemplar Index<br/>k=7 (REST=0, STRESS=1, EXERCISE=2)<br/>knn_retrieval.py"]
    end

    subgraph S4["STAGE 4: MODEL EVALUATION & VALIDATION"]
        RF_TRAIN & H1_CALIB & H2_INDEX --> LOSO["34-Fold Nested LOSO Engine<br/>Strict Outer Test Isolation<br/>run_research_experiments.py"]
        LOSO --> EVAL["Evaluation Matrix E0–E3<br/>Macro-F1, Stress F1, Exercise FAR<br/>metrics.py"]
        EVAL --> RES_STORE[("Persistence<br/>research_comparison.csv<br/>fold_predictions.csv")]
    end

    subgraph S5["STAGE 5: MODEL INFERENCE & APPLICATION"]
        RES_STORE -.-> CKPT["Deployable Checkpoints<br/>deploy_RF_EDA+PPG.pkl"]
        CKPT --> DASH["Streamlit Sense–Think–Act Replay<br/>Chronological Replay / Synthetic Demo<br/>dashboard.py"]
        DASH --> SENSE["SENSE: Sliding Window Buffer<br/>sensor_adapter.py"]
        SENSE --> THINK["THINK: Inference + Contextualization<br/>RF (20D) + H1 Motion Gate"]
        THINK --> ACT["ACT: Adaptive Closed-Loop Guidance<br/>Rules Engine (0.01ms) vs LLM Adapter<br/>act_evaluator.py"]
    end

    style S1 fill:#f8f9fa,stroke:#495057,stroke-width:2px
    style S2 fill:#e9ecef,stroke:#495057,stroke-width:2px
    style S3 fill:#e8f4f8,stroke:#0c5460,stroke-width:2px
    style S4 fill:#d1ecf1,stroke:#0c5460,stroke-width:2px
    style S5 fill:#d4edda,stroke:#155724,stroke-width:2px
    style RAW fill:#ffffff,stroke:#6c757d
    style RES_STORE fill:#fff3cd,stroke:#856404
    style CKPT fill:#fff3cd,stroke:#856404
```

---

## 3. Macro Execution Flowchart (Level 0)

```mermaid
flowchart TD
    START(["Program Entry: CLI / Runner / Dashboard"]):::terminator --> CONF["Load Configuration & Environment<br/>config.py: ProjectConfig"]:::process
    CONF --> ROUTE{"Select Primary Execution Mode"}:::decision

    ROUTE -->|"CLI: make audit"| AUDIT["Run Physical Signal Audit<br/>data_audit.py"]:::process
    AUDIT --> AUDIT_OUT[("Save Audit CSVs<br/>results/data_audit/")]:::storage

    ROUTE -->|"CLI: make train"| EXP_BASE["Run 3x3 Factorial LOSO<br/>run_experiments.py"]:::process
    EXP_BASE --> SPLITS["Generate/Load LOSO Splits<br/>splits.py"]:::subproc
    SPLITS --> FACT_LOOP["Iterate Models x Sensors x Seeds"]:::subproc
    FACT_LOOP --> FACT_METRICS[("Save Factorial Metrics<br/>results/metrics/")]:::storage

    ROUTE -->|"CLI: research runner"| EXP_RES["Run Contextualized LOSO (E0-E3)<br/>run_research_experiments.py"]:::process
    EXP_RES --> RES_INGEST["Ingest EDA, PPG, ACC Sessions<br/>signal_loader.py"]:::subproc
    RES_INGEST --> RES_SEG["Segment 30s Temporal Windows<br/>segmentation.py"]:::subproc
    RES_SEG --> RES_FEAT["Compute 20D Physio + 7D Motion<br/>feature_extractor.py | acc_features.py"]:::subproc
    RES_FEAT --> RES_FOLDS{"Iterate 34 LOSO Folds"}:::decision

    RES_FOLDS -->|"Fold i"| FOLD_FIT["1. Train RF on N-1 Subjects<br/>2. Fit H1 ACC Gate (25th pct DynSVM)<br/>3. Build H2 KNN Index (k=7)"]:::process
    FOLD_FIT --> FOLD_EVAL["Evaluate Held-Out Test Subject<br/>E0: Baseline RF<br/>E1: RF + H1 (ACC Gate)<br/>E2: RF + H2 (KNN Physio)<br/>E3: RF + H1 + H2 (Combined)"]:::process
    FOLD_EVAL --> RES_FOLDS
    RES_FOLDS -->|"Completed 34 Folds"| RES_OUT[("Save Research CSVs<br/>research_comparison.csv<br/>fold_predictions.csv")]:::storage

    ROUTE -->|"CLI: streamlit run"| DASH_INIT["Start Streamlit UI<br/>dashboard.py"]:::process
    DASH_INIT --> MODE_SEL{"Input Mode?"}:::decision
    MODE_SEL -->|"Mode A"| REPLAY["PhysioNet Dataset Replay<br/>sensor_adapter.py"]:::subproc
    MODE_SEL -->|"Mode B"| SYNTH["Synthetic Demo Generator"]:::subproc
    MODE_SEL -->|"Mode C/D"| EMOTIBIT["EmotiBit CSV / UDP<br/>[Hardware Pending]"]:::subproc

    REPLAY & SYNTH & EMOTIBIT --> SLIDE["Buffer 30s Window (stride=5s)"]:::process
    SLIDE --> INFER["Load Checkpoint & Predict<br/>deploy_RF_EDA+PPG.pkl"]:::subproc
    INFER --> ACT_ENGINE["Select Recommendation Engine"]:::decision
    ACT_ENGINE -->|"Deterministic"| RULE["Rule Engine (< 0.1ms)<br/>act_evaluator.py"]:::process
    ACT_ENGINE -->|"Offline LLM"| LLM["LLM Recommender Adapter<br/>act_evaluator.py"]:::process
    RULE & LLM --> DISP["Render Live UI Dashboard<br/>SENSE + THINK + ACT Cards"]:::output
    DISP --> REPLAY_LOOP{"Next Timestep?"}:::decision
    REPLAY_LOOP -->|"Yes"| SLIDE
    REPLAY_LOOP -->|"User Stop"| END(["End Execution"]):::terminator

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
    classDef storage fill:#f8d7da,stroke:#dc3545,stroke-width:1.5px,color:#721c24;
    classDef output fill:#e2e3e5,stroke:#383d41,stroke-width:1.5px,color:#383d41;
```

---

## 4. Stage Summary & File Reference Table

| Stage | Name | Key Modules | Primary Functions / Classes | Inputs | Outputs |
|---|---|---|---|---|---|
| **1** | **Data Preparation** | `src/preprocessing/signal_loader.py`<br/>`src/preprocessing/segmentation.py`<br/>`src/preprocessing/filters.py`<br/>`src/data_audit.py` | `SignalLoader`<br/>`WindowSegmenter`<br/>`filter_eda()`, `filter_bvp()` | Raw Empatica CSVs (`EDA.csv`, `BVP.csv`, `ACC.csv`, `tags.csv`) | 30s Window Dicts (`eda_window`, `bvp_window`, `acc_window`, `label`) |
| **2** | **Feature Engineering** | `src/features/feature_extractor.py`<br/>`src/research/acc_aware/acc_features.py`<br/>`src/models/tensor_builder.py` | `FeatureExtractor`<br/>`extract_acc_features()`<br/>`TensorBuilder` | 30s Window Dicts | 20D Physio Vectors, 7D ACC Vectors, $(C \times 480)$ Tensors |
| **3** | **Model Training** | `src/models/random_forest.py`<br/>`src/models/cnn1d.py`<br/>`src/research/acc_aware/movement_context.py`<br/>`src/research/similarity_aware/knn_retrieval.py`<br/>`src/research/fusion/fusion_policy.py` | `RandomForestClassifier`<br/>`Small1DCNN`, `CNNTrainer`<br/>`MovementContextualizer`<br/>`PhysiologicalKNN`<br/>`FusionPolicy` | Training Matrices ($X_{\text{train}}, y_{\text{train}}$), Exercise Windows | Trained Classifiers, Gate Thresholds, Fitted $k$-NN Index |
| **4** | **Model Evaluation** | `src/evaluation/splits.py`<br/>`src/evaluation/metrics.py`<br/>`src/experiments/run_experiments.py`<br/>`src/research/run_research_experiments.py` | `generate_loso_manifest()`<br/>`compute_binary_metrics()`<br/>`compute_exercise_far()`<br/>`run_research_experiments()` | Outer Folds Manifest, Precomputed Features | `factorial_metrics.csv`, `research_comparison.csv`, `fold_predictions.csv` |
| **5** | **Model Inference** | `src/act/sensor_adapter.py`<br/>`src/act/dashboard.py`<br/>`src/act/act_evaluator.py`<br/>`src/act/run_act_eval.py` | `ReplaySensorAdapter`<br/>`Streamlit UI`<br/>`DeterministicRuleAct`<br/>`LLMActAdapter` | Live stream / Replay buffer, Checkpoints | Stress predictions, Rule action codes, LLM advice, Session logs |

---

## 5. Navigation Links

- [Level 4: Python Code Execution Map & Call Graphs](python_execution_map.md)
- [Stage 1: Data Preparation & Preprocessing](../01-data-preparation/README.md)
- [Stage 2: Feature Engineering & Representation](../02-feature-engineering/README.md)
- [Stage 3: Model Training & Contextualization](../03-model-training/README.md)
- [Stage 4: Model Evaluation & Validation](../04-model-evaluation/README.md)
- [Stage 5: Model Inference & Application](../05-model-inference/README.md)
- [Subsystems Architecture Deep Dives](../subsystems/README.md)
- [Vector Data Cloud Visualization](../visualizations/README.md)
