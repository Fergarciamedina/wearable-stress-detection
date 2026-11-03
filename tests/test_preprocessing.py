"""
Unit Tests for Preprocessing, Windowing, Feature Extraction, and Model Input Shapes
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor
"""

import os
import unittest
import numpy as np
import torch

from src.preprocessing.filters import filter_eda, decompose_eda, filter_bvp
from src.preprocessing.segmentation import WindowSegmenter
from src.features.feature_extractor import FeatureExtractor
from src.models.tensor_builder import TensorBuilder
from src.models.cnn1d import Small1DCNN, CNNTrainer
from src.evaluation.splits import generate_loso_manifest
from src.evaluation.metrics import compute_binary_metrics, compute_clustered_bootstrap_ci


class TestWearablePipeline(unittest.TestCase):
    
    def test_filter_shapes(self):
        # 30 seconds of EDA at 4 Hz = 120 samples
        raw_eda = np.random.uniform(0.1, 1.5, size=120)
        filtered_eda = filter_eda(raw_eda, fs=4.0)
        self.assertEqual(len(filtered_eda), 120)

        tonic, phasic = decompose_eda(filtered_eda, fs=4.0)
        self.assertEqual(len(tonic), 120)
        self.assertEqual(len(phasic), 120)

        # 30 seconds of BVP at 64 Hz = 1920 samples
        raw_bvp = np.random.randn(1920)
        filtered_bvp = filter_bvp(raw_bvp, fs=64.0)
        self.assertEqual(len(filtered_bvp), 1920)

    def test_window_segmentation_boundaries(self):
        segmenter = WindowSegmenter(window_len_s=30.0, stride_s=15.0)
        # 180 seconds of baseline rest -> at 4Hz = 720 samples
        eda_sig = np.random.uniform(0.1, 1.0, 720)
        bvp_sig = np.random.randn(720 * 16)
        
        mock_session = {
            "subject_id": "S01",
            "session_type": "STRESS",
            "protocol_version": "V1",
            "eda_fs": 4.0,
            "eda_signal": eda_sig,
            "bvp_fs": 64.0,
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
            # Check length: 30s * 4Hz = 120 samples
            self.assertEqual(len(w["eda_window"]), 120)
            self.assertIn(w["label"], [0, 1])
            self.assertEqual(w["subject_id"], "S01")

    def test_feature_extractor_dimensions(self):
        extractor = FeatureExtractor(eda_fs=4.0, bvp_fs=64.0)
        eda_win = np.random.uniform(0.1, 1.0, 120)
        bvp_win = np.random.randn(1920)

        mock_window = {
            "eda_window": eda_win,
            "bvp_window": bvp_win
        }

        eda_feat = extractor.extract_features_for_sensor_config(mock_window, "EDA")
        self.assertEqual(eda_feat.shape, (10,))

        ppg_feat = extractor.extract_features_for_sensor_config(mock_window, "PPG")
        self.assertEqual(ppg_feat.shape, (10,))

        fusion_feat = extractor.extract_features_for_sensor_config(mock_window, "EDA+PPG")
        self.assertEqual(fusion_feat.shape, (20,))

    def test_tensor_builder_shapes(self):
        builder = TensorBuilder(target_fs=16.0, window_len_s=30.0)
        eda_win = np.random.uniform(0.1, 1.0, 120)
        bvp_win = np.random.randn(1920)

        mock_window = {
            "eda_window": eda_win,
            "bvp_window": bvp_win
        }

        t_eda = builder.build_tensor(mock_window, "EDA")
        self.assertEqual(t_eda.shape, (1, 480))

        t_ppg = builder.build_tensor(mock_window, "PPG")
        self.assertEqual(t_ppg.shape, (1, 480))

        t_fusion = builder.build_tensor(mock_window, "EDA+PPG")
        self.assertEqual(t_fusion.shape, (2, 480))

    def test_1d_cnn_forward_pass(self):
        batch_size = 4
        in_channels = 1
        time_steps = 480
        x = torch.randn(batch_size, in_channels, time_steps)
        model = Small1DCNN(in_channels=1, num_classes=2)
        out = model(x)
        self.assertEqual(out.shape, (batch_size, 2))

        # 2-channel Fusion tensor batch
        x2 = torch.randn(batch_size, 2, time_steps)
        model2 = Small1DCNN(in_channels=2, num_classes=2)
        out2 = model2(x2)
        self.assertEqual(out2.shape, (batch_size, 2))

    def test_subject_disjoint_splits(self):
        subjects = [f"S{i:02d}" for i in range(1, 19)]
        manifest = generate_loso_manifest(subjects, inner_val_ratio=0.2, seed=42)
        self.assertEqual(len(manifest), len(subjects))
        
        for fold in manifest:
            test_s = fold["test_subject"]
            train_s = fold["train_subjects"]
            val_s = fold["val_subjects"]
            
            # Zero leakage checks
            self.assertNotIn(test_s, train_s)
            self.assertNotIn(test_s, val_s)
            self.assertEqual(len(set(train_s).intersection(set(val_s))), 0)


if __name__ == "__main__":
    unittest.main()
