# Subsystem 2: Feature Extraction Pipeline

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Algorithmic Formulations

The feature extraction pipeline converts continuous 30-second physiological and motion waveforms into compact, noise-tolerant feature vectors.

### EDA Feature Extraction (`extract_eda_features`)
Inputs: 30-second EDA window (120 samples at 4 Hz).
1. **Moments (Mean, Std, Min, Max, Range):** Standard empirical statistical moments capturing baseline level and variance.
2. **Linear Drift Slope:** $\beta_1$ calculated by fitting a 1st-degree polynomial $x_t = \beta_1 t + \beta_0$ across the 120 time indices using `np.polyfit`.
3. **Tonic Conductance:** Moving average over 5.0 seconds (kernel length $K = 20$ samples):
   $$\text{Tonic}_t = \frac{1}{K} \sum_{j=-K/2}^{K/2} \text{EDA}_{t+j}$$
4. **Phasic SCR Peak Count & Amplitudes:** Phasic signal isolated by $\max(0, \text{EDA} - \text{Tonic})$. Peaks detected via `scipy.signal.find_peaks` with parameters:
   - Minimum height: $0.01\,\mu\text{S}$.
   - Minimum temporal distance: $1.0\,\text{s}$ (4 samples).
   - Metrics extracted: Total peak count, maximum peak amplitude, and cumulative sum of peak amplitudes.

### BVP/PPG Feature Extraction (`extract_bvp_features`)
Inputs: 30-second BVP window (1920 samples at 64 Hz).
1. **Moments & Energy:** Mean, standard deviation, peak-to-peak range, and mean squared signal energy $\frac{1}{N}\sum y_i^2$.
2. **Systolic Peak Detection & Heart Rate:** Peaks located using `scipy.signal.find_peaks` with:
   - Minimum distance: $0.4\,\text{s}$ (25 samples at 64 Hz, enforcing a physiological ceiling of $150\,\text{BPM}$).
   - Prominence threshold: $0.5 \times \sigma_{\text{BVP}}$.
3. **Peak-to-Peak Intervals (PPI) & Heart Rate Variability:**
   - Interval vector in seconds: $\text{PPI}_k = \frac{\text{peak}_{k+1} - \text{peak}_k}{64.0}$.
   - Pulse rate: $\text{BPM} = 60.0 / \overline{\text{PPI}}$.
   - SDRR (standard deviation of intervals): $\sigma_{\text{PPI}}$.
   - RMSSD: Root mean square of successive interval differences:
     $$\text{RMSSD} = \sqrt{\frac{1}{K-1} \sum_{k=1}^{K-1} (\text{PPI}_{k+1} - \text{PPI}_k)^2}$$
4. **Spectral Power (Welch Periodogram):** Welch PSD computed with Hamming window segment length $8.0\,\text{s}$ (512 samples). Low-frequency power integrated across $0.5 - 2.0\,\text{Hz}$; high-frequency power across $2.0 - 5.0\,\text{Hz}$.

### Accelerometer Feature Extraction (`extract_acc_features`)
Inputs: 30-second ACC window ($960 \times 3$ samples at 32 Hz).
1. Total acceleration magnitude: $SVM_i = \sqrt{x_i^2 + y_i^2 + z_i^2}$.
2. Dynamic acceleration: $SVM_i - \text{Convolve}(SVM, \mathbf{1}_{32}/32)_i$.
3. 7 Kinematic features: `SVM_MEAN`, `SVM_STD`, `DYNAMIC_SVM_MEAN`, `DYNAMIC_SVM_STD`, `SMA`, `ENERGY`, `JERK_MEAN`.

---

## 2. Subsystem Block Diagram

```mermaid
graph TD
    subgraph IN["Input 30-Second Window Arrays"]
        EDA["EDA Window<br/>120 samples @ 4 Hz"]
        BVP["BVP Window<br/>1920 samples @ 64 Hz"]
        ACC["ACC Window<br/>960x3 samples @ 32 Hz"]
    end

    subgraph EDA_PIPE["EDA Pipeline (10 Features)"]
        E_STAT["Statistical Moments<br/>Mean, Std, Min, Max, Range"]
        E_SLOPE["Linear Trend<br/>np.polyfit(t, eda, 1)"]
        E_TONIC["Tonic Extraction<br/>convolve(ones(20)/20)"]
        E_PHASIC["Phasic SCR Peaks<br/>find_peaks(h>=0.01, d=4)"]
        E_STAT & E_SLOPE & E_TONIC & E_PHASIC --> E_VEC["10D EDA Vector"]
    end

    subgraph BVP_PIPE["BVP Pipeline (10 Features)"]
        B_STAT["Morphological Stats<br/>Mean, Std, Range, Energy"]
        B_PEAKS["Systolic Peak Detection<br/>find_peaks(dist=25, prom=0.5*std)"]
        B_HRV["HRV Dynamics<br/>BPM, PPI, SDRR, RMSSD"]
        B_SPEC["Welch Periodogram<br/>LF (0.5-2Hz), HF (2-5Hz)"]
        B_STAT & B_PEAKS & B_HRV & B_SPEC --> B_VEC["10D BVP Vector"]
    end

    subgraph ACC_PIPE["ACC Pipeline (7 Features)"]
        A_SVM["Magnitude & Grav Removal<br/>SVM = sqrt(x^2+y^2+z^2)<br/>Dyn = SVM - Boxcar(32)"]
        A_FEATS["Kinematic Metrics<br/>DynMean, DynStd, SMA, Jerk, Energy"]
        A_SVM --> A_FEATS --> A_VEC["7D ACC Vector"]
    end

    subgraph OUT["Formatted Feature Representations"]
        PHYSIO["20D Physiological Vector<br/>[EDA_10, BVP_10]<br/>Random Forest / k-NN"]
        MOTION["7D Motion Vector<br/>H1 Movement Gate"]
    end

    EDA --> EDA_PIPE
    BVP --> BVP_PIPE
    ACC --> ACC_PIPE

    E_VEC & B_VEC --> PHYSIO
    A_VEC --> MOTION

    style IN fill:#f8f9fa,stroke:#6c757d
    style EDA_PIPE fill:#e8f4f8,stroke:#0c5460
    style BVP_PIPE fill:#e8f4f8,stroke:#0c5460
    style ACC_PIPE fill:#d1ecf1,stroke:#0c5460
    style OUT fill:#d4edda,stroke:#155724,stroke-width:2px
```

---

## 3. Subsystem Execution Flowchart

```mermaid
flowchart TD
    START(["Start Feature Extraction"]):::terminator --> IN_DATA["Input: 30s Window Object"]:::process

    %% EDA Sub-routine
    IN_DATA --> CALC_EDA["Compute 10 EDA Features:<br/>- mean, std, min, max, range<br/>- slope = polyfit(t, eda, 1)[0]<br/>- tonic = convolve(eda, ones(20)/20, 'same')<br/>- phasic = clip(eda - tonic, 0, inf)<br/>- peaks = find_peaks(phasic, h>=0.01, d=4)<br/>- count, max_amp, sum_amp"]:::subproc

    %% BVP Sub-routine
    IN_DATA --> CHK_BVP{"Window has valid BVP?"}:::decision
    CHK_BVP -->|"No"| ZERO_BVP["bvp_vec = zeros(10)"]:::process
    CHK_BVP -->|"Yes"| CALC_BVP["Compute 10 BVP Features:<br/>- mean, std, range, energy<br/>- peaks = find_peaks(bvp, d=25, prom=0.5*std)<br/>- PPI = diff(peaks) / 64.0<br/>- BPM = 60.0 / mean(PPI)<br/>- SDRR = std(PPI)<br/>- RMSSD = sqrt(mean(diff(PPI)^2))<br/>- LF/HF power via scipy.signal.welch"]:::subproc

    CALC_EDA & ZERO_BVP & CALC_BVP --> CONCAT_20D["Concatenate 20D Physiological Vector:<br/>physio_20d = [eda_vec (10,), bvp_vec (10,)]"]:::process

    %% ACC Sub-routine
    IN_DATA --> CHK_ACC{"Window has valid ACC?"}:::decision
    CHK_ACC -->|"No"| ZERO_ACC["acc_vec = zeros(7)"]:::process
    CHK_ACC -->|"Yes"| CALC_ACC["Compute 7 ACC Features:<br/>- SVM = sqrt(x^2 + y^2 + z^2)<br/>- DynSVM = SVM - convolve(SVM, ones(32)/32)<br/>- SVM_MEAN, SVM_STD<br/>- DYNAMIC_SVM_MEAN, DYNAMIC_SVM_STD<br/>- SMA = mean(sum(|x|+|y|+|z|))<br/>- ENERGY = mean(SVM^2)<br/>- JERK = mean(|diff(SVM)| * 32.0)"]:::subproc

    CONCAT_20D & ZERO_ACC & CALC_ACC --> NAN_CLEAN["np.nan_to_num(vectors, nan=0.0, posinf=0.0)"]:::process
    NAN_CLEAN --> END_FEAT(["Return (physio_20d, acc_vec)"]):::terminator

    classDef terminator fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef process fill:#e8f4f8,stroke:#17a2b8,stroke-width:1.5px,color:#0c5460;
    classDef subproc fill:#d1ecf1,stroke:#0c5460,stroke-width:2px,color:#0c5460;
    classDef decision fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:#856404;
```
