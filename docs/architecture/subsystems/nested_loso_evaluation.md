# Subsystem 7: Nested Subject-Disjoint LOSO Evaluation

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Zero Data-Leakage Protocol

In physiological machine learning, random window-level cross-validation leads to severe optimistic bias because neighboring sliding windows from the same subject share baseline physiological morphology. The evaluation subsystem implements a strict **Nested Leave-One-Subject-Out (LOSO)** cross-validation strategy:

1. **Outer Loop (34 Folds):**
   - In each fold $i \in \{1, \dots, 34\}$, exactly one subject $s_{\text{test}}$ is held out exclusively for test scoring.
   - The remaining 33 participants form the development cohort $\mathcal{S}_{\text{dev}}$.
2. **Inner Loop (Zero-Leakage Model Selection):**
   - For hyperparameter tuning and early stopping (1D-CNN), 20% of $\mathcal{S}_{\text{dev}}$ is reserved as an inner validation cohort $\mathcal{S}_{\text{val}}$, while the remainder $\mathcal{S}_{\text{train}}$ trains the weights.
3. **Strict Normalization Isolation:**
   - All standardizers (`sklearn.preprocessing.StandardScaler`) are fit strictly on $\mathcal{S}_{\text{train}}$. The held-out test subject is transformed using frozen parameters:
     $$X_{\text{test, scaled}} = \frac{X_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
4. **Contextualization Isolation:**
   - The motion threshold $\tau_{\text{motion}}$ is computed strictly from $\mathcal{S}_{\text{train}}$ exercise sessions.
   - The $k$-NN reference index is populated strictly from $\mathcal{S}_{\text{train}}$ feature vectors.

---

## 2. Statistical Confidence Estimation

Implemented in [`src/evaluation/metrics.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/evaluation/metrics.py):
- **Subject-Clustered Bootstrap:** Rather than resampling individual windows (which violates independence), the bootstrap resamples entire subject-level Macro-F1 scores with replacement ($B = 1000$ iterations) to construct 95% confidence intervals:
  $$\text{CI}_{95\%} = [\text{Percentile}_{2.5}(\{\bar{\theta}_b\}), \text{Percentile}_{97.5}(\{\bar{\theta}_b\})]$$

---

## 3. Subsystem Block Diagram

```mermaid
graph TD
    subgraph COHORT["34 Eligible Paired Subjects"]
        ALL_S["Subjects: S01-S18 (ex S02), f01-f18 (ex f07)"]
    end

    subgraph OUTER_LOOP["Outer LOSO Loop (Fold i = 1..34)"]
        TEST_SUBJ["Test Subject: s_test<br/>[Completely Isolated]"]
        DEV_SUBJS["Development Subjects: S_dev (33 subjects)"]
    end

    subgraph INNER_SPLIT["Inner Partitioning (splits.py)"]
        VAL_SUBJS["Inner Validation (20%): S_val<br/>[Hyperparameters & Early Stop]"]
        TRAIN_SUBJS["Inner Training (80%): S_train<br/>[Fit Weights, Scaler, H1, H2]"]
    end

    subgraph FIT_STAGE["Train-Only Fitting"]
        FIT_SCALER["Fit StandardScaler(S_train)"]
        FIT_RF["Fit Random Forest(S_train)"]
        FIT_H1["Calibrate H1 Gate(S_train Exercise)"]
        FIT_H2["Build H2 k-NN Index(S_train)"]
        TRAIN_SUBJS --> FIT_SCALER & FIT_RF & FIT_H1 & FIT_H2
    end

    subgraph TEST_STAGE["Evaluation on Isolated s_test"]
        TRANS_TEST["Transform: X_test_scaled = scaler.transform(X_test)"]
        EVAL_E0["Evaluate E0 (Baseline)"]
        EVAL_E1["Evaluate E1 (ACC Gate)"]
        EVAL_E2["Evaluate E2 (k-NN Similarity)"]
        EVAL_E3["Evaluate E3 (Combined)"]
        TEST_SUBJ --> TRANS_TEST --> EVAL_E0 & EVAL_E1 & EVAL_E2 & EVAL_E3
    end

    subgraph SCORING["Fold Metrics Persistence"]
        OUT_CSV[("fold_predictions.csv<br/>34 Folds x 4 Configurations")]
        EVAL_E0 & EVAL_E1 & EVAL_E2 & EVAL_E3 --> OUT_CSV
    end

    ALL_S --> TEST_SUBJ & DEV_SUBJS
    DEV_SUBJS --> VAL_SUBJS & TRAIN_SUBJS
    FIT_SCALER & FIT_RF & FIT_H1 & FIT_H2 -.->|Frozen Parameters| TEST_STAGE

    style COHORT fill:#f8f9fa,stroke:#6c757d
    style OUTER_LOOP fill:#e8f4f8,stroke:#0c5460
    style INNER_SPLIT fill:#d1ecf1,stroke:#0c5460
    style FIT_STAGE fill:#fff3cd,stroke:#856404
    style TEST_STAGE fill:#d4edda,stroke:#155724,stroke-width:2px
    style SCORING fill:#fff3cd,stroke:#856404
```

---

## 4. Subsystem Execution Flowchart

```mermaid
flowchart TD
    START(["Start 34-Fold LOSO Evaluation"]):::terminator --> LOAD_SPLITS["Load / Generate loso_splits.json"]:::process
    LOAD_SPLITS --> LOOP_FOLDS{"For fold i in 0..33"}:::decision

    LOOP_FOLDS -->|"Next Fold i"| EXTRACT_PARTITIONS["Extract s_test and s_train list"]:::process
    EXTRACT_PARTITIONS --> TRAIN_PIPE["Train pipeline on s_train strictly:<br/>1. Scaler.fit()<br/>2. RandomForest.fit()<br/>3. MovementContextualizer.fit()<br/>4. PhysiologicalKNN.fit()"]:::subproc

    TRAIN_PIPE --> EVAL_STRESS["Evaluate s_test on Stress Session:<br/>Transform features -> Predict E0, E1, E2, E3<br/>Compute Macro-F1, Stress F1, Rest F1"]:::subproc

    EVAL_STRESS --> CHK_EX{"s_test has Exercise Session?"}:::decision
    CHK_EX -->|"Yes"| EVAL_EX["Predict on Exercise Windows:<br/>Compute Exercise FAR for E0, E1, E2, E3"]:::subproc
    CHK_EX -->|"No"| NAN_EX["FAR = NaN"]:::process

    EVAL_EX & NAN_EX --> SAVE_ROW["Append fold metrics to fold_predictions.csv"]:::storage
    SAVE_ROW --> LOOP_FOLDS

    LOOP_FOLDS -->|"All 34 Folds Done"| BOOTSTRAP["Compute Clustered Bootstrap 95% CIs:<br/>Resample subject scores (B=1000)"]:::subproc
    BOOTSTRAP --> FINAL_CSV[("Save research_comparison.csv")]:::storage
    FINAL_CSV --> END_LOSO(["LOSO Evaluation Completed"]):::terminator

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
    classDef storage fill:#f8d7da,stroke:#dc3545,stroke-width:1.5px,color:#721c24;
```
