# Subsystem 1: Multisensor Preprocessing

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Technical Specification & Signal Conditioning

The preprocessing subsystem ingests three heterogeneous asynchronous sensor streams from the Empatica E4 wristband and prepares clean, boundary-safe 30-second windows.

### Digital Filtering Specifications
1. **Electrodermal Activity (EDA, 4 Hz):**
   - **Primary Low-Pass Filter:** 4th-order zero-phase Butterworth filter with cutoff $f_c = 1.0\text{ Hz}$ (`scipy.signal.filtfilt(b, a, eda_signal)`). Removes high-frequency electrical interference while preserving sudomotor nerve impulses ($0.05 - 0.5\text{ Hz}$).
   - **Tonic/Phasic Decomposition:** 2nd-order low-pass filter with cutoff $f_c = 0.05\text{ Hz}$ extracts slowly varying tonic Skin Conductance Level (SCL). The phasic Skin Conductance Response (SCR) is obtained via subtraction:
     $$\text{Phasic}(t) = \max\left(0, \text{EDA}_{\text{filtered}}(t) - \text{Tonic}(t)\right)$$
2. **Photoplethysmography (BVP, 64 Hz):**
   - **Band-Pass Filter:** 4th-order zero-phase Butterworth filter with passband $[0.5\text{ Hz}, 8.0\text{ Hz}]$. Lower cutoff ($0.5\text{ Hz}$, corresponding to $30\text{ BPM}$) suppresses baseline respiratory wander; upper cutoff ($8.0\text{ Hz}$, corresponding to $480\text{ BPM}$) eliminates high-frequency sensor motion jitter while preserving dicrotic notches.
3. **Tri-Axial Accelerometer (ACC, 32 Hz):**
   - **Signal Vector Magnitude:** $SVM(t) = \sqrt{x(t)^2 + y(t)^2 + z(t)^2}$.
   - **Gravity Removal:** 1-second rolling boxcar average (32 samples) captures the static gravitational acceleration vector ($\sim 64$ units in E4 scale):
     $$\text{Dynamic } SVM(t) = SVM(t) - (SVM * \mathbf{1}_{32}/32)(t)$$

---

## 2. Window Segmentation Algorithm

Implemented in `src/preprocessing/segmentation.py` (`WindowSegmenter`):
- **Window Length:** $T_{\text{win}} = 30.0\text{ s}$ (120 EDA samples, 1920 BVP samples, 960 ACC samples).
- **Temporal Stride:** $T_{\text{stride}} = 15.0\text{ s}$ ($50\%$ overlap).
- **Boundary Safety Constraint:** Slicing is constrained strictly within each protocol interval $[t_{\text{start}}, t_{\text{end}}]$. If $t_{\text{curr}} + 30.0\text{ s} > t_{\text{end}}$, the loop terminates immediately for that phase. Windows never cross protocol phases or participant sessions.

---

## 3. Subsystem Block Diagram

```mermaid
graph LR
    subgraph RAW["Raw Stream Inputs"]
        R_EDA["Raw EDA<br/>4 Hz, float32"]
        R_BVP["Raw BVP<br/>64 Hz, float32"]
        R_ACC["Raw ACC (x,y,z)<br/>32 Hz, int16"]
    end

    subgraph FILTERS["Digital Conditioning (src/preprocessing/filters.py)"]
        BUTTER_EDA["Butterworth Low-pass<br/>4th Order, fc = 1.0 Hz<br/>Zero-phase filtfilt()"]
        DECOMP_EDA["Tonic Decomposition<br/>2nd Order Low-pass, fc = 0.05 Hz<br/>Phasic = clip(EDA - Tonic, min=0)"]
        BUTTER_BVP["Butterworth Band-pass<br/>4th Order, [0.5 - 8.0 Hz]<br/>Zero-phase filtfilt()"]
        ACC_GRAV["Kinematic Gravity Filter<br/>SVM = sqrt(x^2 + y^2 + z^2)<br/>DynSVM = SVM - Convolve(1/32)"]
    end

    subgraph SEGMENTATION["Window Segmenter (src/preprocessing/segmentation.py)"]
        GUARD["Interval Boundary Guard<br/>curr_start + 30s <= interval_end"]
        SLICE["Fixed Index Slicers:<br/>- EDA: [start*4 : end*4] (120)<br/>- BVP: [start*64 : end*64] (1920)<br/>- ACC: [start*32 : end*32] (960x3)"]
        STRIDE["Stride Step: +15.0s"]
    end

    subgraph OUTPUT["Preprocessed Window Tuple"]
        WIN["Segmented Window Dict:<br/>{subject_id, phase, label, eda_window, bvp_window, acc_window}"]
    end

    R_EDA --> BUTTER_EDA --> DECOMP_EDA
    R_BVP --> BUTTER_BVP
    R_ACC --> ACC_GRAV

    DECOMP_EDA & BUTTER_BVP & ACC_GRAV --> GUARD --> SLICE --> WIN
    SLICE --> STRIDE --> GUARD

    style RAW fill:#f8f9fa,stroke:#6c757d
    style FILTERS fill:#e8f4f8,stroke:#0c5460
    style SEGMENTATION fill:#d1ecf1,stroke:#0c5460
    style OUTPUT fill:#d4edda,stroke:#155724,stroke-width:2px
```

---

## 4. Subsystem Execution Flowchart

```mermaid
flowchart TD
    START(["Start Preprocessing Pipeline"]):::terminator --> IN_SIGS["Input: Raw continuous EDA, BVP, ACC arrays & labeled intervals"]:::process
    IN_SIGS --> FILT_EDA["filter_eda(eda_signal, fs=4.0, cutoff=1.0, order=4)"]:::subproc
    FILT_EDA --> DECOMP["decompose_eda(eda_signal, fs=4.0) -> tonic, phasic"]:::subproc
    DECOMP --> FILT_BVP["filter_bvp(bvp_signal, fs=64.0, lowcut=0.5, highcut=8.0, order=4)"]:::subproc

    FILT_BVP --> LOOP_INTERVALS{"For interval in session.labeled_intervals"}:::decision
    LOOP_INTERVALS -->|"Next Interval"| CHECK_DUR{"interval_end - interval_start >= 30.0s?"}:::decision
    CHECK_DUR -->|"No"| SKIP_INT["Skip short interval"]:::process
    SKIP_INT --> LOOP_INTERVALS

    CHECK_DUR -->|"Yes"| INIT_STRIDE["curr_start = interval_start"]:::process
    INIT_STRIDE --> STRIDE_COND{"curr_start + 30.0s <= interval_end?"}:::decision
    STRIDE_COND -->|"No"| LOOP_INTERVALS
    STRIDE_COND -->|"Yes"| SLICE_DATA["Slice raw segments:<br/>eda_win = eda[curr_start*4 : (curr_start+30)*4]<br/>bvp_win = bvp[curr_start*64 : (curr_start+30)*64]<br/>acc_win = acc[curr_start*32 : (curr_start+30)*32]"]:::process

    SLICE_DATA --> BUILD_WIN["Build Window Dict and Append to Output"]:::process
    BUILD_WIN --> ADVANCE["curr_start += 15.0s"]:::process
    ADVANCE --> STRIDE_COND

    LOOP_INTERVALS -->|"All Intervals Processed"| END_PREP(["Return List of Segmented Windows"]):::terminator

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
```
