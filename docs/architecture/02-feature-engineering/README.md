# Stage 2: Feature Engineering & Representation

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Stage Scope & Responsibilities

Stage 2 transforms raw continuous 30-second physiological and kinematic window arrays into standardized numerical representations appropriate for downstream machine learning models:
1. **20-Dimensional Handcrafted Physiological Features:** Engineered for tabular Random Forest classification and non-parametric $k$-NN exemplar retrieval.
2. **7-Dimensional Kinematic Motion Features:** Derived from 3-axis accelerometry to measure dynamic physical exertion for H1 movement gating.
3. **Temporal Waveform Tensors ($(C \times 480)$):** Resampled to a uniform 16 Hz temporal grid for end-to-end 1D Convolutional Neural Network training.

---

## 2. 20-Dimensional Physiological Feature Set

Implemented in [`src/features/feature_extractor.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/features/feature_extractor.py) (`FeatureExtractor`).

### EDA Sub-Vector (Features 1 to 10)
| Index | Feature Name | Description | Mathematical Formulation |
|---|---|---|---|
| **0** | `EDA_MEAN` | Arithmetic mean of window | $\mu = \frac{1}{N} \sum_{i=1}^N x_i$ |
| **1** | `EDA_STD` | Standard deviation | $\sigma = \sqrt{\frac{1}{N} \sum (x_i - \mu)^2}$ |
| **2** | `EDA_MIN` | Minimum skin conductance value | $\min(x)$ |
| **3** | `EDA_MAX` | Peak skin conductance value | $\max(x)$ |
| **4** | `EDA_RANGE` | Conductance span | $\max(x) - \min(x)$ |
| **5** | `EDA_SLOPE` | Linear drift rate | First-order polynomial coefficient $\beta_1$ via `np.polyfit(t, x, 1)` |
| **6** | `EDA_TONIC_MEAN` | Mean of tonic component (SCL) | Moving average filter (5-second kernel, 20 samples) |
| **7** | `EDA_SCR_NUM_PEAKS` | Phasic peak count (SCR) | Peaks detected on phasic signal with height $\ge 0.01\,\mu\text{S}$, distance $\ge 1.0\,\text{s}$ |
| **8** | `EDA_SCR_MAX_AMP` | Maximum SCR peak amplitude | $\max(\text{peak\_heights})$ |
| **9** | `EDA_SCR_SUM_AMP` | Cumulative phasic amplitude | $\sum \text{peak\_heights}$ |

### BVP/PPG Sub-Vector (Features 11 to 20)
| Index | Feature Name | Description | Mathematical Formulation |
|---|---|---|---|
| **10** | `BVP_MEAN` | Arithmetic mean | $\mu = \frac{1}{N} \sum y_i$ |
| **11** | `BVP_STD` | Signal dispersion | $\sigma = \sqrt{\frac{1}{N} \sum (y_i - \mu)^2}$ |
| **12** | `BVP_RANGE` | Pulse excursion | $\text{ptp}(y) = \max(y) - \min(y)$ |
| **13** | `BVP_ENERGY` | Mean signal energy | $\frac{1}{N} \sum y_i^2$ |
| **14** | `BVP_PULSE_RATE_BPM` | Estimated instantaneous pulse | Systolic peaks detected with distance $\ge 0.4\,\text{s}$ ($150\,\text{BPM}$ max); $\text{BPM} = 60 / \overline{\text{PPI}}$ |
| **15** | `BVP_MEAN_PPI` | Mean peak-to-peak interval | $\overline{\text{PPI}} = \frac{1}{K} \sum \Delta t_{\text{peaks}}$ (seconds) |
| **16** | `BVP_SDRR` | Standard deviation of intervals | $\sigma_{\text{PPI}}$ (Heart Rate Variability measure) |
| **17** | `BVP_RMSSD` | Root mean square successive diffs | $\sqrt{\frac{1}{K-1} \sum (\text{PPI}_{k+1} - \text{PPI}_k)^2}$ |
| **18** | `BVP_LF_POWER` | Low-frequency spectral power | Spectral density in $0.5 - 2.0\,\text{Hz}$ via Welch periodogram |
| **19** | `BVP_HF_POWER` | High-frequency spectral power | Spectral density in $2.0 - 5.0\,\text{Hz}$ via Welch periodogram |

---

## 3. 7-Dimensional Tri-Axial ACC Motion Feature Set

Implemented in [`src/research/acc_aware/acc_features.py`](file:///Users/fernando/Concordia/2026_2_Fall/COEN_6322/project/code/src/research/acc_aware/acc_features.py) (`extract_acc_features`).

From a 30-second window at 32 Hz ($960 \times 3$ integer acceleration values in Empatica units where $\sim 64 \approx 1g$):

1. **`SVM_MEAN`:** Mean Signal Vector Magnitude ($SVM = \sqrt{x^2 + y^2 + z^2}$).
2. **`SVM_STD`:** Standard deviation of total acceleration magnitude.
3. **`DYNAMIC_SVM_MEAN`:** Dynamic motion intensity isolated by subtracting a 1-second (32-sample) moving average gravity estimate: $\text{Dynamic } SVM = SVM - \text{Convolve}(SVM, \mathbf{1}_{32}/32)$.
4. **`DYNAMIC_SVM_STD`:** Variability of dynamic acceleration.
5. **`SMA`:** Signal Magnitude Area: $\frac{1}{N} \sum (|x| + |y| + |z|)$.
6. **`ENERGY`:** Mean squared magnitude: $\frac{1}{N} \sum SVM^2$.
7. **`JERK_MEAN`:** Rate of change of acceleration magnitude: $\frac{1}{N-1} \sum |\Delta SVM| \times f_s$.

---

## 4. Stage 2 Block Diagram

```mermaid
graph LR
    subgraph INPUT["Input: 30-Second Window Object"]
        EDA_WIN["eda_window<br/>120 samples @ 4 Hz"]
        BVP_WIN["bvp_window<br/>1920 samples @ 64 Hz"]
        ACC_WIN["acc_window<br/>960 x 3 samples @ 32 Hz"]
    end

    subgraph FEAT_ENG["Handcrafted Feature Extractor (src/features/feature_extractor.py)"]
        subgraph EDA_FEATS["10 EDA Statistical & Phasic Features"]
            E1["1. Mean<br/>2. Std Dev<br/>3. Min<br/>4. Max<br/>5. Range (Max-Min)"]
            E2["6. Linear Slope (polyfit)<br/>7. Tonic Mean (5s kernel)<br/>8. SCR Peak Count (h>=0.01)<br/>9. Max SCR Peak Amplitude<br/>10. Sum of SCR Amplitudes"]
        end
        subgraph BVP_FEATS["10 BVP/PPG Morphological & Spectral Features"]
            P1["1. Mean<br/>2. Std Dev<br/>3. Range<br/>4. Energy (mean x^2)<br/>5. Pulse Rate BPM (peaks)"]
            P2["6. Mean PPI (Peak-to-Peak)<br/>7. SDRR (Std PPI)<br/>8. RMSSD (diff PPI)<br/>9. LF Power (0.5-2.0 Hz)<br/>10. HF Power (2.0-5.0 Hz)"]
        end
    end

    subgraph ACC_ENG["Motion Feature Extractor (src/research/acc_aware/acc_features.py)"]
        subgraph ACC_7D["7 Tri-Axial ACC Features"]
            A1["1. SVM_MEAN (Signal Vector Mag)<br/>2. SVM_STD<br/>3. DYNAMIC_SVM_MEAN (Gravity removed)<br/>4. DYNAMIC_SVM_STD"]
            A2["5. SMA (Signal Magnitude Area)<br/>6. ENERGY (Mean SVM^2)<br/>7. JERK_MEAN (d(SVM)/dt * fs)"]
        end
    end

    subgraph TENSOR_ENG["Temporal Tensor Builder (src/models/tensor_builder.py)"]
        RESAMP_EDA["Resample EDA: 4 Hz -> 16 Hz<br/>(480 samples)"]
        RESAMP_BVP["Resample BVP: 64 Hz -> 16 Hz<br/>(480 samples)"]
        STACK["Stack Channels: (C, 480)<br/>C=1 (EDA/PPG), C=2 (Fusion)"]
    end

    subgraph OUTPUT_REPS["Output Mathematical Representations"]
        REP_20D["20D Physiological Vector (Tabular)<br/>Used by Random Forest & KNN<br/>shape: (20,) float32"]
        REP_7D["7D Motion Vector (Kinematic)<br/>Used by H1 Movement Gate<br/>shape: (7,) float32"]
        REP_TENSOR["Temporal Waveform Tensor<br/>Used by PyTorch 1D-CNN<br/>shape: (C, 480) float32"]
    end

    EDA_WIN --> E1 & E2
    BVP_WIN --> P1 & P2
    ACC_WIN --> A1 & A2

    EDA_WIN --> RESAMP_EDA
    BVP_WIN --> RESAMP_BVP
    RESAMP_EDA & RESAMP_BVP --> STACK

    E1 & E2 & P1 & P2 --> REP_20D
    A1 & A2 --> REP_7D
    STACK --> REP_TENSOR

    style INPUT fill:#f8f9fa,stroke:#6c757d
    style FEAT_ENG fill:#e8f4f8,stroke:#0c5460
    style ACC_ENG fill:#e8f4f8,stroke:#0c5460
    style TENSOR_ENG fill:#e9ecef,stroke:#495057
    style OUTPUT_REPS fill:#d4edda,stroke:#155724,stroke-width:2px
```

---

## 5. Stage 2 Execution Flowchart

```mermaid
flowchart TD
    START(["Start: Extract Features for Window"]):::terminator --> ROUTE{"Sensor Set / Model Branch"}:::decision

    ROUTE -->|"Tabular (RF / KNN)"| IN_WIN["Input Window Dict: eda_window, bvp_window"]:::process
    IN_WIN --> EXTRACT_EDA["extract_eda_features(eda_window):<br/>1. Moments: mean, std, min, max, range<br/>2. Slope: polyfit(x, eda, 1)[0]<br/>3. Tonic: 5s moving avg kernel conv<br/>4. Phasic: eda - tonic<br/>5. Peaks: find_peaks(phasic, h>=0.01, d=4)<br/>6. Output: 10 EDA features"]:::subproc

    EXTRACT_EDA --> BVP_CHECK{"bvp_window is None?"}:::decision
    BVP_CHECK -->|"Yes"| ZERO_BVP["bvp_features = zeros(10)"]:::process
    BVP_CHECK -->|"No"| EXTRACT_BVP["extract_bvp_features(bvp_window):<br/>1. Moments: mean, std, range, energy<br/>2. Peaks: find_peaks(bvp, dist=25, prom=std*0.5)<br/>3. PPI: diff(peaks) / 64.0<br/>4. Pulse BPM: 60.0 / mean(PPI)<br/>5. SDRR: std(PPI), RMSSD: sqrt(mean(diff(PPI)^2))<br/>6. PSD: welch() -> LF (0.5-2.0Hz), HF (2.0-5.0Hz)<br/>7. Output: 10 BVP features"]:::subproc

    ZERO_BVP & EXTRACT_BVP --> CONCAT_PHYSIO["Concatenate Physiological Vector:<br/>EDA only: (10,)<br/>PPG only: (10,)<br/>EDA+PPG Fusion: (20,) float32"]:::process
    CONCAT_PHYSIO --> NAN_GUARD["np.nan_to_num(features, nan=0.0)"]:::process
    NAN_GUARD --> OUT_PHYSIO(["Return 20D Feature Vector"]):::terminator

    ROUTE -->|"Kinematics (H1 Gate)"| IN_ACC["Input: acc_window (N, 3)"]:::process
    IN_ACC --> CHECK_ACC_SHAPE{"len(acc) >= 2 and shape[1] == 3?"}:::decision
    CHECK_ACC_SHAPE -->|"No"| ZERO_ACC["Return zeros(7)"]:::process
    CHECK_ACC_SHAPE -->|"Yes"| COMP_SVM["SVM = sqrt(x^2 + y^2 + z^2)"]:::process
    COMP_SVM --> SUB_GRAV["Gravity Removal:<br/>kernel = ones(32) / 32<br/>gravity = convolve(SVM, kernel, 'same')<br/>dynamic_svm = SVM - gravity"]:::subproc
    SUB_GRAV --> COMP_ACC_FEATS["Compute 7 ACC Features:<br/>1. mean(SVM), 2. std(SVM)<br/>3. mean(|dynamic_svm|), 4. std(dynamic_svm)<br/>5. mean(sum(|x|+|y|+|z|)) (SMA)<br/>6. mean(SVM^2) (Energy)<br/>7. mean(|diff(SVM)| * fs) (Jerk)"]:::process
    COMP_ACC_FEATS --> OUT_ACC(["Return 7D Motion Vector"]):::terminator

    ROUTE -->|"Waveform (1D-CNN)"| IN_TENSOR["Input: eda_window, bvp_window"]:::process
    IN_TENSOR --> RES_EDA["Resample EDA:<br/>scipy.signal.resample(eda, 480)"]:::subproc
    RES_EDA --> RES_BVP{"BVP valid?"}:::decision
    RES_BVP -->|"No"| ZERO_TENSOR["bvp_resampled = zeros(480)"]:::process
    RES_BVP -->|"Yes"| RESAMPLE_BVP["Resample BVP:<br/>scipy.signal.resample(bvp, 480)"]:::subproc
    ZERO_TENSOR & RESAMPLE_BVP --> STACK_CHANNELS{"sensor_config?"}:::decision
    STACK_CHANNELS -->|"EDA"| STACK_1A["shape: (1, 480)"]:::process
    STACK_CHANNELS -->|"PPG"| STACK_1B["shape: (1, 480)"]:::process
    STACK_CHANNELS -->|"EDA+PPG"| STACK_2["shape: (2, 480)"]:::process
    STACK_1A & STACK_1B & STACK_2 --> OUT_TENSOR(["Return PyTorch Input Tensor"]):::terminator

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
```
