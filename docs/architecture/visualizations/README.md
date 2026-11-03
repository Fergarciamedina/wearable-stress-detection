# Vector Data Cloud Visualization: 2D PCA of 20D Physiological Space

**COEN 6322: AI for Wearable Health Applications, Fall 2026**

---

## 1. Overview & Purpose

In this research project, each 30-second window is mapped to a **20-dimensional physiological vector** capturing 10 Electrodermal Activity (EDA) features (tonic conductance, phasic SCR peaks, linear drift) and 10 Photoplethysmography (BVP/PPG) features (heart rate, HRV RMSSD/SDRR, Welch low/high-frequency power).

To provide visual intuition into:
1. The severe **autonomic overlap** between acute cognitive stress and physical exercise, and
2. The exact spatial mechanism of the **$k$-NN ($k=7$) exemplar retrieval** in Hypothesis 2,

we generated a verified 2-dimensional Principal Component Analysis (PCA) projection directly from the authentic PhysioNet dataset cohort.

---

## 2. Real Dataset Statistics & PCA Decomposition

Computed by [`generate_pca_visualization.py`](generate_pca_visualization.py):

| Metric | Empirical Value |
|---|---|
| **Total Windows Extracted** | **10,414 windows** across 34 subjects |
| **Verified REST Windows (Label 0)** | 2,741 windows (Baseline & Recovery) |
| **Verified STRESS Windows (Label 1)** | 625 windows (Stroop, TMCT, Speech, Mental Arithmetic) |
| **Verified EXERCISE Windows (Label 2)** | 7,048 windows (AEROBIC & ANAEROBIC challenge protocols) |
| **Sampled Points for Visualization** | 1,250 balanced windows (to prevent visual clutter) |
| **PC1 Explained Variance** | **36.27%** |
| **PC2 Explained Variance** | **20.83%** |
| **Cumulative Explained Variance (PC1 + PC2)** | **57.10%** |

---

## 3. Visual Representation

![2D PCA Vector Space Projection](pca_vector_space.png)

*(A scalable vector version is available in [pca_vector_space.svg](pca_vector_space.svg))*

---

## 4. Scientific Interpretation of the Feature Space

### 1. The Autonomic Overlap Zone
As visualized in the upper-right quadrant, **STRESS (orange)** and **EXERCISE (purple)** points intermingle significantly along PC1. Both conditions represent profound sympathetic nervous system excitation:
- Heart rate surges ($\text{BPM} > 95$).
- Heart rate variability collapses ($\text{RMSSD} < 20\,\text{ms}$).
- Skin conductance level dramatically rises ($\text{EDA} > 3.5\,\mu\text{S}$).

Because these two physiological states occupy adjacent or overlapping manifolds in autonomic space, a model trained purely on autonomic data cannot reliably differentiate cognitive stress from physical exercise without an auxiliary kinematic sensor (tri-axial ACC).

### 2. The $k$-NN (k=7) Exemplar Query Mechanism
The pink star indicates an illustrative test window extracted during a physical exercise protocol:
- The exact $k$-NN algorithm queries the 7 nearest neighbors in 20D standardized space.
- The retrieved neighbors consist of:
  - **4 EXERCISE exemplars** (57.1%)
  - **2 REST exemplars** (28.6%)
  - **1 STRESS exemplar** (14.3%)
- **H2 Decision:** Since the exercise neighbor fraction $f_{\text{ex}} = 57.1\% \ge 40.0\%$, the contextualizer flags this window as physical movement and successfully **suppresses the false stress alarm**.
- **The Trade-off:** When a true cognitive stress window lands in the dense autonomic overlap zone, its neighborhood is often contaminated with exercise exemplars, leading H2 to erroneously suppress true stress alarms (reducing Stress F1 to 0.1784). This empirical visual confirms why **H1 (ACC motion gating)** is the superior, Pareto-optimal approach for practical wearable devices.
