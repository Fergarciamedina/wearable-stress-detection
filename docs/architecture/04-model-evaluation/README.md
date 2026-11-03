# Stage 4: Model Evaluation & Validation

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Stage Scope & Responsibilities

Stage 4 conducts the scientific validation of all model configurations under subject-disjoint partitions. It enforces zero data leakage between development and evaluation participants, computes multi-class classification and false-alarm metrics, and saves reproducible benchmarks.

### Primary Modules
- [`src/evaluation/splits.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/evaluation/splits.py): Partition manifest generator for outer LOSO and inner grouped validation.
- [`src/evaluation/metrics.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/evaluation/metrics.py): Binary classification metrics, clustered bootstrap confidence intervals, and Exercise False Alarm Rate (FAR).
- [`src/research/run_research_experiments.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/research/run_research_experiments.py): 34-fold nested evaluation runner for configurations E0–E3.
- [`src/experiments/run_experiments.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/experiments/run_experiments.py): Historical 3×3 factorial runner (Majority, RF, 1D-CNN across EDA, PPG, Fusion).

---

## 2. Experimental Configurations (E0–E3)

All research configurations evaluate the best-performing multimodal physiological feature representation (EDA + PPG, 20 features) under Leave-One-Subject-Out:

- **E0 (Baseline Random Forest):** Uncontextualized RF predictions.
- **E1 (RF + H1 Movement Gate):** Dynamic acceleration gating. If primary RF predicts STRESS (1) and `DYNAMIC_SVM_MEAN` $\ge \tau_{\text{motion}}$, the prediction is overridden to REST (0).
- **E2 (RF + H2 Physiological Similarity):** Non-parametric $k$-NN retrieval ($k=7$). If primary RF predicts STRESS (1) and exercise neighbor fraction $\ge 0.40$, the prediction is overridden to REST (0).
- **E3 (Combined H1 + H2):** Sequential fusion. H1 motion gate is applied first; remaining STRESS predictions are subsequently screened by H2 similarity.

---

## 3. Verified Empirical Benchmark Results (34-Fold LOSO)

From [`data/results/research/comparison/research_comparison.csv`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/data/results/research/comparison/research_comparison.csv):

| Config | Description | Macro-F1 (Mean ± SD) | Balanced Acc | Rest F1 | Stress F1 | Exercise FAR |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **E0** | Baseline Random Forest | 0.6481 ± 0.1570 | 68.94% | 0.8303 | 0.4660 | **61.58%** |
| **E1** | **RF + H1 ACC-aware movement gate** | **0.6528 ± 0.1528** | **68.81%** | **0.8430** | **0.4627** | **15.58%** |
| **E2** | RF + H2 KNN physiological similarity | 0.5226 ± 0.0849 | 54.18% | 0.8669 | 0.1784 | **2.79%** |
| **E3** | RF + H1 + H2 (Combined) | 0.5228 ± 0.0848 | 54.21% | 0.8671 | 0.1785 | **0.98%** |

### Nuanced Pareto Analysis
- **H1 Performance Trade-off:** H1 provides an extraordinary **74.7% relative reduction** in Exercise FAR (from 61.58% down to 15.58%) while increasing overall Macro-F1 (0.6481 to 0.6528) and Rest F1 (0.8303 to 0.8430). However, it is not strictly Pareto-dominant across every single metric: Stress F1 exhibits a slight decline (0.4660 to 0.4627) and Balanced Accuracy drops by 0.13% (68.94% to 68.81%) due to occasional small wrist gestures during cognitive tasks tripping the 25th percentile gate.
- **H2 Autonomic Overlap:** H2 demonstrates that physiological similarity alone can reduce false alarms to 2.79%, but because sympathetic hyperarousal during exercise mimics cognitive stress, it over-suppresses true stress alarms, lowering Stress F1 to 0.1784.

---

## 4. Stage 4 Block Diagram

```mermaid
graph LR
    subgraph SPLIT_ENGINE["LOSO Partitioning (src/evaluation/splits.py)"]
        MANIFEST["loso_splits.json<br/>34 Outer Folds"]
        FOLD_I["Fold i: Test Subject s_test<br/>Train Subjects s_train (N=33)"]
        MANIFEST --> FOLD_I
    end

    subgraph ISOLATION_BOUNDARY["Strict Data Isolation Boundary (Zero Leakage)"]
        direction TB
        TRAIN_ENV["Training Environment (s_train)<br/>- StandardScaler.fit()<br/>- RF.fit()<br/>- H1 Percentile.fit()<br/>- H2 KNN.fit()"]
        TEST_ENV["Held-Out Test Environment (s_test)<br/>- Stress Session: X_test_rs, y_test_rs, acc_test_rs<br/>- Exercise Challenge: X_test_ex, acc_test_ex"]
    end

    subgraph FUSION_EVAL["Policy Evaluation Engine (run_research_experiments.py)"]
        direction TB
        E0["E0: Baseline RF Prediction<br/>y_pred_base = rf.predict(X_test)"]
        E1["E1: H1 Gated Prediction<br/>y_e1 = acc_ctx.contextualize(y_pred_base, acc_test)"]
        E2["E2: H2 Similarity Gated<br/>y_e2 = sim_ctx.contextualize(y_pred_base, X_test)"]
        E3["E3: Sequential Combined<br/>y_e3 = sim_ctx.contextualize(y_e1, X_test)"]
        E0 --> E1 & E2 --> E3
    end

    subgraph METRICS_CALC["Metrics Calculation (src/evaluation/metrics.py)"]
        STRESS_EVAL["Stress Session Scoring:<br/>compute_binary_metrics(y_true, y_pred)<br/>- Macro-F1<br/>- Stress Recall & F1<br/>- Rest Recall & F1<br/>- Balanced Accuracy"]
        EX_EVAL["Exercise Challenge Scoring:<br/>compute_exercise_far(y_ex_pred)<br/>- Exercise FAR = False Stress / Total Ex Windows"]
    end

    subgraph PERSIST["Results Persistence"]
        CSV_SUMMARY[("research_comparison.csv<br/>Summary: Macro-F1, FAR, Acc")]
        CSV_FOLDS[("fold_predictions.csv<br/>34 Folds x 4 Configurations")]
    end

    FOLD_I --> TRAIN_ENV & TEST_ENV
    TRAIN_ENV & TEST_ENV --> FUSION_EVAL
    FUSION_EVAL --> STRESS_EVAL & EX_EVAL
    STRESS_EVAL & EX_EVAL --> CSV_SUMMARY & CSV_FOLDS

    style SPLIT_ENGINE fill:#f8f9fa,stroke:#6c757d
    style ISOLATION_BOUNDARY fill:#fff3cd,stroke:#856404,stroke-width:2px,stroke-dasharray: 5 5
    style FUSION_EVAL fill:#e8f4f8,stroke:#0c5460
    style METRICS_CALC fill:#d1ecf1,stroke:#0c5460
    style PERSIST fill:#d4edda,stroke:#155724,stroke-width:2px
```

---

## 5. Stage 4 Execution Flowchart

```mermaid
flowchart TD
    START(["Start Research Evaluation: run_research_experiments()"]):::terminator --> LOAD_SPLITS["Load manifest: loso_splits.json (34 folds)"]:::process
    LOAD_SPLITS --> CHECK_RESUME{"Check fold_predictions.csv exists?"}:::decision
    CHECK_RESUME -->|"Yes"| LOAD_DONE["Load done_folds set to skip completed"]:::process
    CHECK_RESUME -->|"No"| INIT_EMPTY["Initialize fold_rows = []"]:::process

    LOAD_DONE & INIT_EMPTY --> FOLD_LOOP{"For each fold in manifest (34 total)"}:::decision
    FOLD_LOOP -->|"Fold i already done"| SKIP_FOLD["Skip fold"]:::process
    SKIP_FOLD --> FOLD_LOOP

    FOLD_LOOP -->|"Execute Fold i"| ISOLATE["Isolate s_test, s_train, s_val"]:::process
    ISOLATE --> FIT_ALL["Fit Stage 3 Models on s_train:<br/>1. Scaler + Random Forest<br/>2. H1 MovementContextualizer (25th pct)<br/>3. H2 SimilarityContextualizer (k=7)"]:::subproc

    FIT_ALL --> TEST_PRED["Transform Held-Out Test Features:<br/>X_test_scaled = scaler.transform(X_test_rs)<br/>y_pred_base = rf.predict(X_test_scaled)"]:::process

    TEST_PRED --> POLICY_EVAL{"For exp in [E0, E1, E2, E3]"}:::decision

    POLICY_EVAL -->|"E0"| RUN_E0["y_ctx = y_pred_base"]:::process
    POLICY_EVAL -->|"E1"| RUN_E1["y_ctx = acc_ctx.contextualize(y_pred_base, acc_test_rs)"]:::process
    POLICY_EVAL -->|"E2"| RUN_E2["y_ctx = sim_ctx.contextualize(y_pred_base, X_test_rs)"]:::process
    POLICY_EVAL -->|"E3"| RUN_E3["y_inter = acc_ctx.contextualize(y_pred_base, acc_test_rs)<br/>y_ctx = sim_ctx.contextualize(y_inter, X_test_rs)"]:::process

    RUN_E0 & RUN_E1 & RUN_E2 & RUN_E3 --> SCORE_STRESS["Compute Binary Metrics on Stress Session:<br/>m = compute_binary_metrics(y_test_rs, y_ctx)<br/>- macro_f1, balanced_acc, f1_rest, f1_stress"]:::subproc

    SCORE_STRESS --> HAS_EX{"s_test has exercise challenge windows?"}:::decision
    HAS_EX -->|"Yes"| EVAL_EX["Predict on Exercise Windows:<br/>y_ex_base = rf.predict(scaler.transform(X_test_ex))<br/>y_ex_ctx = policy.fuse(y_ex_base, acc_test_ex, X_test_ex)<br/>far = compute_exercise_far(y_ex_ctx)"]:::subproc
    HAS_EX -->|"No"| SET_NAN_FAR["far = NaN"]:::process

    EVAL_EX & SET_NAN_FAR --> APPEND_RES["Record fold row:<br/>{fold_idx, test_subject, experiment, macro_f1, f1_stress, far}"]:::process
    APPEND_RES --> POLICY_EVAL

    POLICY_EVAL -->|"All 4 exps done for Fold i"| FLUSH_CSV["Save progress to fold_predictions.csv"]:::storage
    FLUSH_CSV --> FOLD_LOOP

    FOLD_LOOP -->|"All 34 Folds Complete"| AGGREGATE["Compute Macro Aggregate Summary:<br/>Mean, Std, CIs per Experiment Configuration"]:::process
    AGGREGATE --> SAVE_SUMMARY[("Save research_comparison.csv")]:::storage
    SAVE_SUMMARY --> END_EVAL(["Evaluation Complete"]):::terminator

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
    classDef storage fill:#f8d7da,stroke:#dc3545,stroke-width:1.5px,color:#721c24;
```
