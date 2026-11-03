# Subsystem 6: Decision Fusion Policies (E0–E3)

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Specification of the Combinatorial Matrix

The decision fusion engine ([`src/research/fusion/fusion_policy.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/research/fusion/fusion_policy.py)) orchestrates the interaction between the primary statistical Random Forest classifier and the two contextualization mechanisms (H1 Motion Gating and H2 Exemplar Retrieval).

| Experiment Configuration | ACC Motion Gate (H1) | Physiological k-NN (H2) | Primary Hypothesis Under Test |
|:---:|:---:|:---:|:---|
| **E0** | Disabled | Disabled | **Baseline:** Can autonomic features alone (EDA + PPG) separate cognitive stress from physical exercise without context? |
| **E1** | **Enabled** | Disabled | **H1:** Does wrist kinematic evidence suppress false alarms during physical exercise without degrading cognitive stress recall? |
| **E2** | Disabled | **Enabled** | **H2:** Can non-parametric neighborhood voting in 20D autonomic space identify exercise confounding without motion sensors? |
| **E3** | **Enabled** | **Enabled** | **Combined:** Does sequential multimodal fusion provide complementary artifact rejection? |

---

## 2. Sequential Execution Order in E3

When both contextualizers are active (`E3`):
1. **Stage 1 — Kinematic Motion Screening (H1):** Fast, rule-based check on tri-axial accelerometry. Any prediction where `DYNAMIC_SVM_MEAN` $\ge \tau_{\text{motion}}$ is immediately overridden from STRESS (1) to REST (0).
2. **Stage 2 — Autonomic Similarity Screening (H2):** The surviving STRESS predictions are queried against the exact $k$-NN index in 20D feature space. If the neighborhood exercise fraction $\ge 0.40$, the prediction is overridden to REST (0).

---

## 3. Subsystem Block Diagram

```mermaid
graph TD
    subgraph IN["Input Windows"]
        FEATS["20D Physiological Features<br/>X_test in R^(N x 20)"]
        ACC_WINS["Tri-Axial ACC Windows<br/>acc_windows (N x 960 x 3)"]
    end

    subgraph PRIMARY["Primary Classification"]
        RF_MODEL["Trained Random Forest<br/>StandardScaler + 100 Trees"]
        Y_PRED["y_pred_base in {0, 1}^N"]
        FEATS --> RF_MODEL --> Y_PRED
    end

    subgraph FUSION_ROUTER["FusionPolicy Router (src/research/fusion/fusion_policy.py)"]
        CFG_DEC{"Active Experiment Config?"}

        subgraph E0_PATH["E0: Baseline Path"]
            OUT_E0["y_out = y_pred_base"]
        end

        subgraph E1_PATH["E1: ACC Gate Path"]
            H1_GATE["MovementContextualizer.contextualize()<br/>If dyn_svm >= tau: STRESS -> REST"]
        end

        subgraph E2_PATH["E2: Similarity Path"]
            H2_GATE["SimilarityContextualizer.contextualize()<br/>If ex_frac >= 0.40: STRESS -> REST"]
        end

        subgraph E3_PATH["E3: Sequential Combined Path"]
            STEP1["Step 1: Apply H1 Motion Gate<br/>y_inter = H1(y_pred_base, acc)"]
            STEP2["Step 2: Apply H2 KNN Gate<br/>y_out = H2(y_inter, X_test)"]
            STEP1 --> STEP2
        end

        CFG_DEC -->|"E0"| OUT_E0
        CFG_DEC -->|"E1"| H1_GATE
        CFG_DEC -->|"E2"| H2_GATE
        CFG_DEC -->|"E3"| STEP1
    end

    subgraph FINAL["Contextualized Predictions"]
        Y_FINAL["y_fused in {0, 1}^N"]
        OUT_E0 & H1_GATE & H2_GATE & STEP2 --> Y_FINAL
    end

    Y_PRED --> CFG_DEC
    ACC_WINS --> H1_GATE & STEP1
    FEATS --> H2_GATE & STEP2

    style IN fill:#f8f9fa,stroke:#6c757d
    style PRIMARY fill:#e8f4f8,stroke:#0c5460
    style FUSION_ROUTER fill:#d1ecf1,stroke:#0c5460
    style FINAL fill:#d4edda,stroke:#155724,stroke-width:2px
```

---

## 4. Subsystem Execution Flowchart

```mermaid
flowchart TD
    START(["Start: FusionPolicy.fuse(y_pred, acc_windows, X_features)"]):::terminator --> INIT["y_out = np.array(y_pred, dtype=int)"]:::process

    INIT --> CHK_H1{"config.use_h1_acc == True?"}:::decision
    CHK_H1 -->|"No"| CHK_H2{"config.use_h2_knn == True?"}:::decision
    CHK_H1 -->|"Yes"| RUN_H1["y_out = acc_ctx.contextualize(y_out, acc_windows)"]:::subproc

    RUN_H1 --> CHK_H2
    CHK_H2 -->|"No"| END_FUSE(["Return y_out"]):::terminator
    CHK_H2 -->|"Yes"| RUN_H2["y_out = sim_ctx.contextualize(y_out, X_features)"]:::subproc

    RUN_H2 --> END_FUSE

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
```
