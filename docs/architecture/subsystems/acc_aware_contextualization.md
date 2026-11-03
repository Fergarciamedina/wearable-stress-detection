# Subsystem 4: ACC-Aware Contextualization (Hypothesis 1)

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Scientific Hypothesis & Problem Formulation

### The Problem
Autonomic physiological models (EDA + PPG) suffer from severe false alarms during physical exercise (~61.6% false stress alarms on PhysioNet challenge data). Physical movement triggers sympathetic nervous system excitation, causing heart rate elevation, heart rate variability depression, and increased sweat gland conductance identical to acute psychological threat.

### Hypothesis 1 (H1)
Tri-axial wrist accelerometry ($f_s = 32\text{ Hz}$) provides direct kinetic evidence of gross physical activity. If an autonomic window is classified as STRESS (1), but concurrent wrist motion intensity exceeds a calibrated threshold, the prediction can be safely overridden to REST (0).

---

## 2. Mathematical Formulation & Algorithmic Design

Implemented in [`src/research/acc_aware/movement_context.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/research/acc_aware/movement_context.py) (`MovementContextualizer`):

### 1. Motion Intensity Metric
From tri-axial acceleration $[x_t, y_t, z_t]$:
1. Signal Vector Magnitude: $SVM_t = \sqrt{x_t^2 + y_t^2 + z_t^2}$.
2. Dynamic Acceleration: $d_t = SVM_t - (SVM * \mathbf{1}_{32}/32)_t$.
3. Feature: $m = \frac{1}{N} \sum_{t=1}^N |d_t|$ (`DYNAMIC_SVM_MEAN`).

### 2. Threshold Calibration (Training Time)
To prevent hand-tuning and eliminate data leakage, the threshold $\tau_{\text{motion}}$ is fitted dynamically per fold strictly on the training subjects' exercise sessions:
$$\tau_{\text{motion}} = \text{Percentile}_{25}\left(\left\{m_j\right\}_{j \in \text{TrainExerciseWindows}}\right)$$
Choosing the **25th percentile** ensures the gate activates for 75% of vigorous exercise patterns while remaining high enough to avoid false triggers during sedentary desk work or small wrist fidgets.

### 3. Contextualization Rule (Inference Time)
For primary model prediction $\hat{y}_i \in \{0, 1\}$ and ACC window $W_{\text{acc}, i}$:
$$\hat{y}_{i, \text{H1}} = \begin{cases} 
0, & \text{if } \hat{y}_i = 1 \text{ and } \text{DYNAMIC\_SVM\_MEAN}(W_{\text{acc}, i}) \ge \tau_{\text{motion}} \\
\hat{y}_i, & \text{otherwise}
\end{cases}$$
**Safety Property:** The gate only overrides STRESS $\to$ REST; it never flips REST $\to$ STRESS.

---

## 3. Subsystem Block Diagram

```mermaid
graph LR
    subgraph CALIBRATION_STAGE["Calibration Phase (Training Partition Only)"]
        TR_EX_ACC["Training Exercise ACC Windows<br/>(AEROBIC + ANAEROBIC)"]
        FEAT_CALC["Extract DYNAMIC_SVM_MEAN<br/>acc_features.py (Index 2)"]
        PCT_FIT["25th Percentile Fit:<br/>tau_motion = np.percentile(dyn_means, 25.0)"]
        TR_EX_ACC --> FEAT_CALC --> PCT_FIT
    end

    subgraph INFERENCE_STAGE["Inference Phase (Held-Out Evaluation)"]
        Y_PRED["Primary RF Prediction<br/>y_pred in {0, 1}"]
        TEST_ACC["Test ACC Window<br/>(960 x 3 samples @ 32Hz)"]
        DYN_EVAL["Compute Dynamic SVM Mean<br/>m_test"]
        GATE_DEC{"m_test >= tau_motion<br/>AND y_pred == 1?"}
        OVERRIDE["Override: STRESS -> REST (0)<br/>[Exercise False Alarm Blocked]"]
        PASSTHROUGH["Pass-Through: y_out = y_pred"]
        
        TEST_ACC --> DYN_EVAL --> GATE_DEC
        Y_PRED --> GATE_DEC
        PCT_FIT -.->|Threshold Parameter| GATE_DEC
        GATE_DEC -->|True| OVERRIDE
        GATE_DEC -->|False| PASSTHROUGH
    end

    subgraph OUTPUT["Final Contextualized Output"]
        Y_OUT["y_contextualized in {0, 1}"]
        OVERRIDE & PASSTHROUGH --> Y_OUT
    end

    style CALIBRATION_STAGE fill:#f8f9fa,stroke:#6c757d
    style INFERENCE_STAGE fill:#e8f4f8,stroke:#0c5460
    style GATE_DEC fill:#fff3cd,stroke:#856404
    style OVERRIDE fill:#d4edda,stroke:#28a745
    style OUTPUT fill:#d4edda,stroke:#155724,stroke-width:2px
```

---

## 4. Subsystem Execution Flowchart

```mermaid
flowchart TD
    START(["Start: MovementContextualizer.contextualize(y_pred, acc_windows)"]):::terminator --> CHK_FIT{"is_fitted == True?"}:::decision
    CHK_FIT -->|"No"| ERR["Raise RuntimeError('Call fit_threshold() first')"]:::process
    CHK_FIT -->|"Yes"| INIT_OUT["y_ctx = np.array(y_pred, dtype=int)"]:::process

    INIT_OUT --> LOOP_WINS{"For i in range(len(y_pred))"}:::decision
    LOOP_WINS -->|"Next Window i"| CHK_PRED{"y_ctx[i] == 1 (STRESS)?"}:::decision

    CHK_PRED -->|"No (REST)"| NEXT_I["Leave unchanged: y_ctx[i] = 0"]:::process
    NEXT_I --> LOOP_WINS

    CHK_PRED -->|"Yes (STRESS)"| CHK_ACC_PRES{"acc_windows[i] is valid?"}:::decision
    CHK_ACC_PRES -->|"No"| NEXT_I_LEAVE["Leave STRESS unchanged (No motion data)"]:::process
    NEXT_I_LEAVE --> LOOP_WINS

    CHK_ACC_PRES -->|"Yes"| EXT_DYN["dyn_mean = extract_acc_features(acc_windows[i])[2]"]:::subproc
    EXT_DYN --> CHK_THRESH{"dyn_mean >= threshold_?"}:::decision
    CHK_THRESH -->|"Yes (Motion Confirmed)"| SET_REST["Override: y_ctx[i] = 0<br/>(Exercise false alarm suppressed)"]:::process
    CHK_THRESH -->|"No (Sedentary)"| KEEP_STRESS["Retain: y_ctx[i] = 1<br/>(True cognitive stress preserved)"]:::process

    SET_REST & KEEP_STRESS --> LOOP_WINS
    LOOP_WINS -->|"All Windows Processed"| END_CTX(["Return Contextualized Predictions y_ctx"]):::terminator

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
```
