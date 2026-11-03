# Subsystem 3: Random Forest & CNN Training Branches

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Architectural Divergence: Tabular vs. Temporal

The project benchmarks two fundamentally distinct machine learning paradigms on wearable physiological signals:

1. **Tabular Feature-Engineered Branch (Random Forest):**
   - **Inductive Bias:** Uses domain-guided statistical moments and physiological peak intervals (20 features).
   - **Invariance:** Highly invariant to phase shifts, sensor baseline drifts, and minor temporal misalignments.
   - **Empirical Performance:** Consistently outperforms raw neural architectures on small cohorts ($N=34$), achieving **Macro-F1 = 0.6528** vs. 0.4782 for 1D-CNN.
2. **End-to-End Temporal Waveform Branch (1D-CNN):**
   - **Inductive Bias:** Hierarchical 1D convolutional kernel feature extractors operating directly on resampled waveforms ($(C \times 480)$ at 16 Hz).
   - **Risk on Small Cohorts:** Prone to overfitting on idiosyncratic subject morphologies despite aggressive regularization and early stopping.

---

## 2. Model Specifications

### Random Forest (`src/models/random_forest.py`)
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_split=4,
    class_weight="balanced",
    random_state=seed,
    n_jobs=-1
)
```
- Integrated with `sklearn.preprocessing.StandardScaler` fitted strictly on training fold vectors.
- Saved deployment artifact (`deploy_RF_EDA+PPG.pkl`) dictionary structure:
  ```python
  {
      "model": rf_model,
      "scaler": scaler,
      "sensor_config": "EDA+PPG",
      "features_dim": 20
  }
  ```

### PyTorch 1D-CNN (`src/models/cnn1d.py`)
- **Layer 1:** `Conv1D(in_channels=C, out_channels=16, kernel_size=7, stride=2, padding=3)` + `ReLU()`.
- **Layer 2:** `Conv1D(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)` + `ReLU()`.
- **Global Average Pooling:** `GAP1D` reducing temporal dimension $120 \to 1$.
- **Classifier:** `Linear(in_features=32, out_features=2)` with Cross-Entropy Loss.
- **Training Harness (`CNNTrainer`):** Adam optimizer ($\text{lr} = 10^{-3}$), batch size 64, maximum 6 epochs, early stopping patience 2 evaluated on inner validation partition.

---

## 3. Subsystem Block Diagram

```mermaid
graph TD
    subgraph INPUT_DATA["Input: 30s Window Dataset"]
        WINS["Training Subject Windows<br/>(N_tr subjects)"]
    end

    subgraph RF_BRANCH["Branch A: Tabular Random Forest"]
        FEAT_20D["FeatureExtractor<br/>-> 20D Physiological Vector"]
        SCALE_TRAIN["StandardScaler.fit_transform()<br/>(Zero Leakage)"]
        RF_FIT["RandomForestClassifier<br/>100 Trees, Depth=8, Balanced"]
        CKPT_RF[("Checkpoint<br/>deploy_RF_EDA+PPG.pkl")]
        FEAT_20D --> SCALE_TRAIN --> RF_FIT --> CKPT_RF
    end

    subgraph CNN_BRANCH["Branch B: Temporal PyTorch 1D-CNN"]
        RESAMP["TensorBuilder<br/>16Hz Common Grid (C, 480)"]
        NORM_WAVE["Channel-wise Standardize<br/>(X - mean) / std"]
        CNN_ARCH["Small1DCNN PyTorch Model<br/>Conv1D(16) -> ReLU -> Conv1D(32) -> GAP -> Linear(2)"]
        EARLY_STOP["CNNTrainer (Adam lr=1e-3)<br/>Early Stopping (patience=2) on Val"]
        CKPT_CNN[("Checkpoint<br/>deploy_CNN_EDA+PPG.pt")]
        RESAMP --> NORM_WAVE --> CNN_ARCH --> EARLY_STOP --> CKPT_CNN
    end

    WINS --> FEAT_20D
    WINS --> RESAMP

    style INPUT_DATA fill:#f8f9fa,stroke:#6c757d
    style RF_BRANCH fill:#e8f4f8,stroke:#0c5460
    style CNN_BRANCH fill:#d1ecf1,stroke:#0c5460
    style CKPT_RF fill:#fff3cd,stroke:#856404
    style CKPT_CNN fill:#fff3cd,stroke:#856404
```

---

## 4. Subsystem Execution Flowchart

```mermaid
flowchart TD
    START(["Start Model Training Branch"]):::terminator --> MODEL_TYPE{"Select Architecture"}:::decision

    %% Random Forest Flow
    MODEL_TYPE -->|"Random Forest"| GET_RF_FEATS["Compute 20D feature matrix X_train (N, 20)"]:::process
    GET_RF_FEATS --> FIT_RF_SCALER["scaler = StandardScaler()<br/>X_scaled = scaler.fit_transform(X_train)"]:::process
    FIT_RF_SCALER --> FIT_RF_INST["Instantiate RandomForestClassifier:<br/>n_estimators=100, max_depth=8, class_weight='balanced'<br/>rf.fit(X_scaled, y_train)"]:::subproc
    FIT_RF_INST --> SAVE_RF_CKPT[("Serialize to deploy_RF_{config}.pkl")]:::storage

    %% 1D-CNN Flow
    MODEL_TYPE -->|"PyTorch 1D-CNN"| BUILD_TENSORS["Build tensor matrix X_tensors (N, C, 480)"]:::process
    BUILD_TENSORS --> SPLIT_VAL["Split inner validation partition (20% subjects)"]:::process
    BUILD_TENSORS --> COMP_STATS["Compute train_mean & train_std per channel"]:::process
    COMP_STATS --> NORM_TENSORS["Normalize: (X - train_mean) / train_std"]:::process
    NORM_TENSORS --> INIT_CNN["Instantiate Small1DCNN(in_channels=C, num_classes=2)"]:::process
    INIT_CNN --> TRAIN_LOOP{"For epoch in range(max_epochs)"}:::decision
    TRAIN_LOOP --> BATCH_STEP["Batch gradient descent (Adam lr=1e-3, batch=64)"]:::subproc
    BATCH_STEP --> VAL_CHECK{"Validation loss improved?"}:::decision
    VAL_CHECK -->|"Yes"| SAVE_BEST["Save best model state weights"]:::process
    VAL_CHECK -->|"No"| CHK_PATIENCE{"Patience exceeded (2 epochs)?"}:::decision
    CHK_PATIENCE -->|"No"| TRAIN_LOOP
    CHK_PATIENCE -->|"Yes (Early Stop)"| STOP_CNN["Halt training loop"]:::process
    SAVE_BEST & STOP_CNN --> SAVE_CNN_CKPT[("Serialize to deploy_CNN_{config}.pt")]:::storage

    SAVE_RF_CKPT & SAVE_CNN_CKPT --> END_TRAIN(["Training Branch Complete"]):::terminator

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
    classDef storage fill:#f8d7da,stroke:#dc3545,stroke-width:1.5px,color:#721c24;
```
