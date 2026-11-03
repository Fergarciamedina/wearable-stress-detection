# 🧪 Automated Test Suite (`tests/`)

This directory contains automated unit and regression tests ensuring reproducibility, signal stability, data sanity, and mathematical correctness across the entire pipeline.

---

## 🏃 Running Tests

To run the complete test suite:

```bash
cd code
python3 -m unittest discover tests
# Or: make test
```

*Expected output:*
```text
................
----------------------------------------------------------------------
Ran 16 tests in 0.05s

OK
```

---

## 📋 Test Catalog

- `test_01_filter_stability_and_shapes`: Validates Butterworth lowpass and bandpass filter stability, ensuring frequency attenuation curves match physiological bounds (EDA: 0.05–1.0 Hz, PPG: 0.5–8.0 Hz).
- `test_02_segmentation_boundaries_and_labels`: Confirms that sliding windows do not leak samples between distinct experimental blocks or across participants.
- `test_03_feature_extractor_dimensions_and_fusion`: Validates dimensional consistency for 10 EDA features, 10 PPG features, and 20 multimodal fusion features.
- `test_04_tensor_builder_shapes`: Checks resampling to 16 Hz and raw tensor output formatting `(C, 480)`.
- `test_05_1d_cnn_architecture_forward_pass`: Validates forward-pass shapes and output logits for PyTorch 1D-CNN across all sensor channels.
- `test_06_loso_zero_leakage_assertion`: Verifies strict Leave-One-Subject-Out participant disjointness across all cross-validation folds.
- `test_07_baseline_classifier_behavior`: Confirms majority class baseline prediction behavior.
- `test_08_checkpoint_loading_and_inference`: Loads serialized Random Forest checkpoints and verifies live inference output.
- `test_09_metrics_and_far_calculation`: Validates Macro-F1, Balanced Accuracy, and Exercise False Alarm Rate (FAR) calculations.
- `test_10_deterministic_act_recommendation`: Confirms deterministic Sense–Think–Act rule engine recommendations and low latency.

