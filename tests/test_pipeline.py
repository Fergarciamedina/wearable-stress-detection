"""
Comprehensive Automated Test Suite for Wearable Stress Detection
COEN 6322: AI for Wearable Health Applications

Covers:
1. Signal preprocessing & Butterworth filters
2. Non-leaking window segmentation
3. Feature extraction dimensions & feature names
4. 1D-CNN tensor builder shapes & PyTorch forward pass
5. LOSO subject-disjoint validation split assertions
6. Pre-trained model checkpoint loading & inference
7. Classification metrics & Exercise False Alarm Rate (FAR) calculation
8. Deterministic rule-based ACT engine
"""

import os
import unittest
import numpy as np
import torch
import pickle
from pathlib import Path

from src.config import default_config
from src.preprocessing.filters import filter_eda, decompose_eda, filter_bvp
from src.preprocessing.segmentation import WindowSegmenter
from src.features.feature_extractor import FeatureExtractor
from src.models.tensor_builder import TensorBuilder
from src.models.cnn1d import Small1DCNN
from src.models.baseline import MajorityBaselineClassifier
from src.models.random_forest import StressRandomForest
from src.evaluation.splits import generate_loso_manifest
from src.evaluation.metrics import compute_binary_metrics, compute_exercise_far
from src.act.act_evaluator import DeterministicRuleAct


class TestWearablePipelineComprehensive(unittest.TestCase):

    def setUp(self):
        self.eda_fs = 4.0
        self.bvp_fs = 64.0
        self.target_fs = 16.0
        self.window_len_s = 30.0
        self.stride_s = 15.0

        # Synthetic 30s window data
        self.eda_samples = int(self.window_len_s * self.eda_fs)  # 120
        self.bvp_samples = int(self.window_len_s * self.bvp_fs)  # 1920

        self.mock_window = {
            "subject_id": "S99",
            "eda_window": np.random.uniform(0.1, 2.5, self.eda_samples).astype(np.float32),
            "bvp_window": np.random.randn(self.bvp_samples).astype(np.float32),
            "label": 1
        }

    def test_01_filter_stability_and_shapes(self):
        """Verifies Butterworth filter outputs preserve array dimensions and produce finite values."""
        raw_eda = np.random.uniform(0.1, 1.5, size=self.eda_samples)
        filtered_eda = filter_eda(raw_eda, fs=self.eda_fs)
        self.assertEqual(len(filtered_eda), self.eda_samples)
        self.assertTrue(np.all(np.isfinite(filtered_eda)))

        tonic, phasic = decompose_eda(filtered_eda, fs=self.eda_fs)
        self.assertEqual(len(tonic), self.eda_samples)
        self.assertEqual(len(phasic), self.eda_samples)
        self.assertTrue(np.all(np.isfinite(tonic)))
        self.assertTrue(np.all(np.isfinite(phasic)))

        raw_bvp = np.random.randn(self.bvp_samples)
        filtered_bvp = filter_bvp(raw_bvp, fs=self.bvp_fs)
        self.assertEqual(len(filtered_bvp), self.bvp_samples)
        self.assertTrue(np.all(np.isfinite(filtered_bvp)))

    def test_02_segmentation_boundaries_and_labels(self):
        """Verifies sliding windows respect boundary lengths and ground truth intervals."""
        segmenter = WindowSegmenter(window_len_s=self.window_len_s, stride_s=self.stride_s)
        # 180s Rest (0) + 120s Stress (1) = 300s total
        total_s = 300
        eda_sig = np.random.uniform(0.1, 1.0, int(total_s * self.eda_fs))
        bvp_sig = np.random.randn(int(total_s * self.bvp_fs))

        mock_session = {
            "subject_id": "S01",
            "session_type": "STRESS",
            "protocol_version": "V1",
            "eda_fs": self.eda_fs,
            "eda_signal": eda_sig,
            "bvp_fs": self.bvp_fs,
            "bvp_signal": bvp_sig,
            "bvp_valid": True,
            "labeled_intervals": [
                {"start_s": 0.0, "end_s": 180.0, "label": 0, "phase": "Baseline"},
                {"start_s": 180.0, "end_s": 300.0, "label": 1, "phase": "Stroop"}
            ]
        }
        windows = segmenter.segment_stress_session(mock_session)
        self.assertTrue(len(windows) > 0)
        for w in windows:
            self.assertEqual(len(w["eda_window"]), self.eda_samples)
            self.assertEqual(len(w["bvp_window"]), self.bvp_samples)
            self.assertIn(w["label"], [0, 1])
            self.assertEqual(w["subject_id"], "S01")

    def test_03_feature_extractor_dimensions_and_fusion(self):
        """Verifies 10 EDA features, 10 PPG features, and 20 fused features."""
        extractor = FeatureExtractor(eda_fs=self.eda_fs, bvp_fs=self.bvp_fs)
        
        eda_feat = extractor.extract_features_for_sensor_config(self.mock_window, "EDA")
        self.assertEqual(eda_feat.shape, (10,))
        self.assertTrue(np.all(np.isfinite(eda_feat)))

        ppg_feat = extractor.extract_features_for_sensor_config(self.mock_window, "PPG")
        self.assertEqual(ppg_feat.shape, (10,))
        self.assertTrue(np.all(np.isfinite(ppg_feat)))

        fusion_feat = extractor.extract_features_for_sensor_config(self.mock_window, "EDA+PPG")
        self.assertEqual(fusion_feat.shape, (20,))
        self.assertTrue(np.all(np.isfinite(fusion_feat)))

        # Consistency check: fusion should be exact concatenation
        np.testing.assert_array_almost_equal(fusion_feat[:10], eda_feat)
        np.testing.assert_array_almost_equal(fusion_feat[10:], ppg_feat)

    def test_04_tensor_builder_shapes(self):
        """Verifies (C, 480) tensor formatting at 16 Hz."""
        builder = TensorBuilder(target_fs=self.target_fs, window_len_s=self.window_len_s)

        t_eda = builder.build_tensor(self.mock_window, "EDA")
        self.assertEqual(t_eda.shape, (1, 480))

        t_ppg = builder.build_tensor(self.mock_window, "PPG")
        self.assertEqual(t_ppg.shape, (1, 480))

        t_fusion = builder.build_tensor(self.mock_window, "EDA+PPG")
        self.assertEqual(t_fusion.shape, (2, 480))

    def test_05_1d_cnn_architecture_forward_pass(self):
        """Verifies 1D-CNN handles single and dual-channel inputs yielding valid logits."""
        batch_size = 4
        time_steps = 480

        # 1-channel (EDA or PPG)
        x1 = torch.randn(batch_size, 1, time_steps)
        model1 = Small1DCNN(in_channels=1, num_classes=2)
        out1 = model1(x1)
        self.assertEqual(out1.shape, (batch_size, 2))

        # 2-channel (Fusion)
        x2 = torch.randn(batch_size, 2, time_steps)
        model2 = Small1DCNN(in_channels=2, num_classes=2)
        out2 = model2(x2)
        self.assertEqual(out2.shape, (batch_size, 2))

    def test_06_loso_zero_leakage_assertion(self):
        """Verifies participant separation strictly holds across folds."""
        subjects = [f"S{i:02d}" for i in range(1, 11)]
        manifest = generate_loso_manifest(subjects, inner_val_ratio=0.2, seed=42)
        self.assertEqual(len(manifest), len(subjects))

        for fold in manifest:
            test_s = fold["test_subject"]
            train_s = fold["train_subjects"]
            val_s = fold["val_subjects"]

            self.assertNotIn(test_s, train_s)
            self.assertNotIn(test_s, val_s)
            self.assertEqual(len(set(train_s).intersection(set(val_s))), 0)

    def test_07_baseline_classifier_behavior(self):
        """Verifies MajorityBaselineClassifier predicts most frequent class."""
        X_train = np.random.randn(10, 20)
        y_train = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])  # 70% Rest (0)
        
        baseline = MajorityBaselineClassifier()
        baseline.fit(X_train, y_train)

        X_test = np.random.randn(5, 20)
        preds = baseline.predict(X_test)
        self.assertTrue(np.all(preds == 0))

    def test_08_checkpoint_loading_and_inference(self):
        """Verifies existing trained RF checkpoint loads and executes correctly."""
        ckpt_path = default_config.checkpoints_dir / "deploy_RF_EDA+PPG.pkl"
        if ckpt_path.exists():
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)
            self.assertIn("model", ckpt)
            self.assertIn("scaler", ckpt)

            rf_model = ckpt["model"]
            scaler = ckpt["scaler"]

            extractor = FeatureExtractor(eda_fs=self.eda_fs, bvp_fs=self.bvp_fs)
            feats = extractor.extract_features_for_sensor_config(self.mock_window, "EDA+PPG").reshape(1, -1)
            feats_scaled = scaler.transform(feats)

            pred = rf_model.predict(feats_scaled)
            probs = rf_model.predict_proba(feats_scaled)

            self.assertIn(pred[0], [0, 1])
            self.assertEqual(probs.shape, (1, 2))
            self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=4)

    def test_09_metrics_and_far_calculation(self):
        """Verifies Macro-F1, Balanced Accuracy, and Exercise FAR calculations."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 0, 0])

        metrics = compute_binary_metrics(y_true, y_pred)
        self.assertIn("macro_f1", metrics)
        self.assertIn("balanced_acc", metrics)
        self.assertTrue(0.0 <= metrics["macro_f1"] <= 1.0)
        self.assertTrue(0.0 <= metrics["balanced_acc"] <= 1.0)

        # FAR test: in exercise, any prediction of STRESS (1) is a false alarm
        exercise_preds = np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0])  # 3 stress out of 10
        far = compute_exercise_far(exercise_preds)
        self.assertEqual(far["exercise_far"], 0.30)
        self.assertEqual(far["false_stress_windows"], 3)

    def test_10_deterministic_act_recommendation(self):
        """Verifies deterministic rule engine generates expected wellness actions."""
        # Scenario: Sedentary Stress
        ctx_stress = {
            "model_prediction": 1,
            "confidence": 0.88,
            "eda_mean_us": 1.45,
            "pulse_rate_bpm": 88.0,
            "motion_acc_magnitude": 1.02,
            "quality_status": "GOOD"
        }
        res_stress = DeterministicRuleAct.generate_recommendation(ctx_stress)
        self.assertEqual(res_stress["action_code"], "STRESS_BREATHING_BREAK")
        self.assertIn("seated break", res_stress["recommendation_text"])

        # Scenario: Exercise Confounding
        ctx_exercise = {
            "model_prediction": 1,
            "confidence": 0.90,
            "eda_mean_us": 2.10,
            "pulse_rate_bpm": 135.0,
            "motion_acc_magnitude": 2.20,
            "quality_status": "GOOD"
        }
        res_exercise = DeterministicRuleAct.generate_recommendation(ctx_exercise)
        self.assertEqual(res_exercise["action_code"], "EXERCISE_AROUSAL_RECOGNIZED")


if __name__ == "__main__":
    unittest.main()
