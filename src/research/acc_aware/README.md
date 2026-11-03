# H1: ACC-Aware Contextualization

COEN 6322: AI for Wearable Health Applications, Fall 2026

## Objective

Cognitive stress detection models using autonomic physiological signals (EDA, PPG) frequently produce false positives during physical exercise because movement triggers sympathetic nervous system excitation. The ACC-Aware Contextualization module (H1) uses tri-axial accelerometer movement signals (32 Hz, Empatica E4) to verify whether autonomic arousal is accompanied by gross physical motion.

## Feature Extraction (`acc_features.py`)

Extracts 7 domain features from each 30-second ACC window ($960 \times 3$ samples):

1. **`SVM_MEAN`**: Mean Signal Vector Magnitude ($SVM = \sqrt{x^2 + y^2 + z^2}$).
2. **`SVM_STD`**: Standard deviation of SVM.
3. **`DYNAMIC_SVM_MEAN`**: Mean dynamic acceleration magnitude after removing gravity via 1-second (32-sample) rolling mean subtraction.
4. **`DYNAMIC_SVM_STD`**: Standard deviation of dynamic acceleration.
5. **`SMA`**: Signal Magnitude Area ($\frac{1}{N} \sum (|x| + |y| + |z|)$).
6. **`ENERGY`**: Mean squared magnitude ($\frac{1}{N} \sum SVM^2$).
7. **`JERK_MEAN`**: Mean of discrete first-order derivative of SVM ($\frac{\Delta SVM}{\Delta t}$).

## Movement Gate (`movement_context.py`)

The `MovementContextualizer` implements a zero-leakage thresholding gate:

- **Calibration:** Threshold is fitted per fold strictly on **training subjects' exercise sessions** at the 25th percentile of `DYNAMIC_SVM_MEAN`.
- **Inference Override:** If a window is classified by the primary RF model as `STRESS (1)`, but `DYNAMIC_SVM_MEAN >= threshold`, the prediction is overridden to `REST (0)`.
- **Preservation:** `REST (0)` predictions are never altered to `STRESS`.

