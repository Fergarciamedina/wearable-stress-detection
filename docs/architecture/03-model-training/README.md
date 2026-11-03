# Stage 3: Model Training & Contextualization

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Stage Scope & Responsibilities

Stage 3 coordinates the model fitting and contextualization layer setup strictly on the $N-1$ development subjects within each cross-validation fold. It guarantees strict zero-leakage boundaries by distinguishing three distinct training procedures:
1. **Supervised Classifier Fitting:** Standardized Random Forest and PyTorch 1D-CNN trained on labeled stress protocol data (REST vs. STRESS).
2. **Kinematic Threshold Calibration (H1):** Non-parametric calibration of the dynamic acceleration gate fitted strictly on training exercise sessions.
3. **Exemplar Indexing (H2):** Construction of a 3-class ($k=7$) exact spatial ball-tree index over training physiological features including exercise exemplars.
4. **Policy Formulation:** Assembling the decision fusion policies for configurations E0, E1, E2, and E3.

---

## 2. Model Architectures & Calibration Parameters

### Primary Classifiers
- **Random Forest (`src/models/random_forest.py`):**
  - Architecture: Scikit-learn ensemble of 100 decision trees.
  - Regularization: `max_depth=8`, `min_samples_split=4`.
  - Class Balancing: `class_weight='balanced'` to offset natural class imbalance (REST ~80%, STRESS ~20%).
  - Preprocessing: `StandardScaler` fitted strictly on training fold matrices.
- **1D-CNN (`src/models/cnn1d.py`):**
  - Architecture: `Conv1D(in_channels, 16, k=7, s=2, p=3) -> ReLU -> Conv1D(16, 32, k=5, s=2, p=2) -> ReLU -> GlobalAveragePooling1D -> Linear(32, 2)`.
  - Optimization: Adam ($\text{lr} = 10^{-3}$, batch size 64). Early stopping with patience 2 on inner validation set.

### H1 Movement Gate Calibration (`src/research/acc_aware/movement_context.py`)
- **Metric:** `DYNAMIC_SVM_MEAN` (mean of gravity-subtracted signal vector magnitude).
- **Calibration Set:** All valid ACC windows from training subjects' AEROBIC and ANAEROBIC sessions.
- **Decision Boundary:** 25th percentile of dynamic acceleration:
  $$\tau_{\text{motion}} = \text{Percentile}_{25}(\{\text{DYNAMIC\_SVM\_MEAN}_i\}_{i \in \text{TrainExercise}})$$
- **Logic:** Any test window classified as STRESS (1) whose `DYNAMIC_SVM_MEAN` $\ge \tau_{\text{motion}}$ is overridden to REST (0).

### H2 Similarity Index Construction (`src/research/similarity_aware/knn_retrieval.py`)
- **Algorithm:** Scikit-learn `NearestNeighbors(n_neighbors=7, metric='euclidean', algorithm='ball_tree')`.
- **Reference Pool:**
  - REST ($y=0$): All training stress session baseline and recovery windows.
  - STRESS ($y=1$): All training cognitive stress task windows.
  - EXERCISE ($y=2$): All training aerobic and anaerobic exercise challenge windows.
- **Decision Boundary:** Exercise fraction threshold $\theta_{\text{ex}} = 0.40$ ($\ge 3$ of 7 nearest neighbors are labeled exercise).
- **Logic:** Any test window classified as STRESS (1) where `exercise_frac` $\ge 0.40$ is overridden to REST (0).

---

## 3. Stage 3 Block Diagram

```mermaid
graph LR
    subgraph TRAIN_PARTITION["Fold Training Partition (N-1 Subjects)"]
        TR_STRESS["Training Stress Sessions<br/>REST (y=0) & STRESS (y=1)<br/>Physiological: (N_tr, 20)"]
        TR_EXERCISE["Training Exercise Sessions<br/>Held-out Challenge (y=2)<br/>Physiological: (N_ex, 20)<br/>ACC Motion: (N_ex, 7)"]
        TR_TENSORS["Training Waveform Tensors<br/>(N_tr, C, 480)"]
    end

    subgraph BASE_TRAIN["1. Base Supervised Model Training"]
        SCALER_RF["StandardScaler (Train-only)<br/>Fit on TR_STRESS 20D"]
        RF_MODEL["Random Forest Classifier<br/>n_estimators=100, max_depth=8<br/>min_samples_split=4, class_weight='balanced'"]
        SCALER_CNN["Channel Standardization<br/>mean & std over (0, 2)"]
        CNN_MODEL["PyTorch Small1DCNN<br/>2 Conv Layers + GAP + Linear<br/>Adam lr=1e-3, early stopping"]
        TR_STRESS --> SCALER_RF --> RF_MODEL
        TR_TENSORS --> SCALER_CNN --> CNN_MODEL
    end

    subgraph H1_CALIBRATION["2. H1: Movement Gate Calibration (acc_aware)"]
        EXT_DYN["Extract DYNAMIC_SVM_MEAN<br/>from training exercise ACC windows"]
        CALC_PCT["Compute 25th Percentile<br/>threshold_ = np.percentile(dyn_means, 25.0)"]
        H1_GATE["MovementContextualizer Object<br/>is_high_movement(win) -> dyn >= threshold_<br/>contextualize(y_pred, acc_wins)"]
        TR_EXERCISE --> EXT_DYN --> CALC_PCT --> H1_GATE
    end

    subgraph H2_INDEXING["3. H2: Physiological k-NN Index (similarity_aware)"]
        CONCAT_REF["Reference Pool Assembly:<br/>- REST (label 0)<br/>- STRESS (label 1)<br/>- EXERCISE (label 2)"]
        SCALER_KNN["StandardScaler (Ref-only)<br/>Fit on concatenated 20D features"]
        KNN_BALLTREE["Exact NearestNeighbors Index<br/>k=7, metric='euclidean', algorithm='ball_tree'"]
        H2_CTX["SimilarityContextualizer Object<br/>ex_threshold = 0.40 (>= 3/7 neighbors)"]
        TR_STRESS & TR_EXERCISE --> CONCAT_REF --> SCALER_KNN --> KNN_BALLTREE --> H2_CTX
    end

    subgraph FUSION_CFG["4. Decision Fusion Policy (fusion/fusion_policy.py)"]
        POL_E0["E0: Baseline RF<br/>(No contextualization)"]
        POL_E1["E1: RF + H1 Gate<br/>(Override STRESS->REST if high motion)"]
        POL_E2["E2: RF + H2 Gate<br/>(Override STRESS->REST if ex_frac >= 0.4)"]
        POL_E3["E3: RF + H1 + H2<br/>(Sequential: H1 Motion -> H2 Similarity)"]
    end

    RF_MODEL --> POL_E0
    RF_MODEL & H1_GATE --> POL_E1
    RF_MODEL & H2_CTX --> POL_E2
    RF_MODEL & H1_GATE & H2_CTX --> POL_E3

    style TRAIN_PARTITION fill:#f8f9fa,stroke:#6c757d
    style BASE_TRAIN fill:#e8f4f8,stroke:#0c5460
    style H1_CALIBRATION fill:#d1ecf1,stroke:#0c5460
    style H2_INDEXING fill:#e8f4f8,stroke:#0c5460
    style FUSION_CFG fill:#d4edda,stroke:#155724,stroke-width:2px
```

---

## 4. Stage 3 Execution Flowchart

```mermaid
flowchart TD
    START(["Start Fold Training Loop: Train Partition {s_train}"]):::terminator --> SPLIT_DATA["Assemble Training Sets from N-1 Subjects:<br/>- X_train_rs: Stress session 20D features<br/>- y_train_rs: Labels 0 (REST) and 1 (STRESS)<br/>- X_train_ex: Exercise session 20D features<br/>- acc_train_ex: Exercise session ACC windows"]:::process

    SPLIT_DATA --> FIT_SCALER["scaler = StandardScaler()<br/>X_train_scaled = scaler.fit_transform(X_train_rs)"]:::process
    FIT_SCALER --> FIT_RF["Fit RandomForestClassifier:<br/>n_estimators=100, max_depth=8, min_samples_split=4<br/>class_weight='balanced', random_state=42<br/>rf.fit(X_train_scaled, y_train_rs)"]:::subproc

    SPLIT_DATA --> COLLECT_DYN["Collect DYNAMIC_SVM_MEAN from all<br/>training exercise ACC windows: acc_train_ex"]:::subproc
    COLLECT_DYN --> CHECK_EX_ACC{"acc_train_ex empty?"}:::decision
    CHECK_EX_ACC -->|"Yes (Fallback)"| SET_INF["threshold_ = +inf (gate disabled)"]:::process
    CHECK_EX_ACC -->|"No"| COMP_PCT["threshold_ = np.percentile(dyn_means, 25.0)"]:::process
    SET_INF & COMP_PCT --> CREATE_H1["Initialize MovementContextualizer(threshold_)"]:::process

    SPLIT_DATA --> ASSEMBLE_REF["Assemble Reference Pool:<br/>X_all = [X_train_rs; X_train_ex]<br/>y_all = [y_train_rs; ones(len(X_train_ex)) * 2]"]:::subproc
    ASSEMBLE_REF --> FIT_KNN_SCALER["knn_scaler = StandardScaler()<br/>X_all_scaled = knn_scaler.fit_transform(X_all)"]:::process
    FIT_KNN_SCALER --> FIT_KNN["Fit NearestNeighbors Index:<br/>nn = NearestNeighbors(k=7, metric='euclidean', algorithm='ball_tree')<br/>nn.fit(X_all_scaled)"]:::subproc
    FIT_KNN --> CREATE_H2["Initialize SimilarityContextualizer(nn, ex_threshold=0.40)"]:::process

    FIT_RF & CREATE_H1 & CREATE_H2 --> ASSEMBLE_POLICIES["Construct Four FusionPolicy Instances:<br/>- E0: FusionPolicy(use_h1=False, use_h2=False)<br/>- E1: FusionPolicy(use_h1=True,  use_h2=False)<br/>- E2: FusionPolicy(use_h1=False, use_h2=True)<br/>- E3: FusionPolicy(use_h1=True,  use_h2=True)"]:::process

    ASSEMBLE_POLICIES --> END_TRAIN(["Ready for Fold Inference & Evaluation"]):::terminator

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
```
