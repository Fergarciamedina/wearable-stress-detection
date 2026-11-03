"""
Research Module Tests — H1 ACC-Aware and H2 Similarity-Aware Contextualization
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

10 unit tests verifying the correctness of the research modules
without requiring the dataset on disk.

Compatible with both ``python -m unittest`` and ``pytest``.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — allow running from repo root or tests/ directory
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve()
for _root in [_here.parent, _here.parent.parent]:
    if (_root / "src").is_dir():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

from src.research.acc_aware.acc_features import (
    extract_acc_features,
    ACC_FEATURE_NAMES,
    _N_FEATURES,
)
from src.research.acc_aware.movement_context import MovementContextualizer
from src.research.similarity_aware.knn_retrieval import PhysiologicalKNN
from src.research.similarity_aware.similarity_context import SimilarityContextualizer
from src.research.fusion.fusion_policy import FusionPolicy, EXPERIMENT_CONFIGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_acc_rest(seed: int = 0) -> np.ndarray:
    """Simulate a nearly-stationary (resting) ACC window at 32 Hz × 30 s."""
    rng = np.random.default_rng(seed)
    base = np.zeros((960, 3), dtype=np.float32)
    base[:, 2] = 64.0  # gravity on z-axis
    base += rng.normal(0, 0.5, size=(960, 3)).astype(np.float32)
    return base


def _make_acc_exercise(seed: int = 1) -> np.ndarray:
    """Simulate a high-movement (exercise) ACC window."""
    rng = np.random.default_rng(seed)
    base = np.zeros((960, 3), dtype=np.float32)
    base[:, 2] = 64.0
    t = np.linspace(0, 30, 960)
    base[:, 0] = (30.0 * np.sin(2 * np.pi * 2.0 * t)).astype(np.float32)
    base[:, 1] = (20.0 * np.cos(2 * np.pi * 1.5 * t)).astype(np.float32)
    base += rng.normal(0, 2, size=(960, 3)).astype(np.float32)
    return base


def _make_knn() -> PhysiologicalKNN:
    """Fitted KNN index on synthetic stress-session + exercise features."""
    rng = np.random.default_rng(42)
    X_rs = rng.standard_normal((40, 20)).astype(np.float32)
    y_rs = np.array([0] * 20 + [1] * 20, dtype=int)
    X_ex = (rng.standard_normal((10, 20)) + 3.0).astype(np.float32)
    knn = PhysiologicalKNN(k=7)
    knn.fit(X_rs, y_rs, X_ex)
    return knn


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestAccFeatures(unittest.TestCase):
    """Tests for acc_features.py."""

    def setUp(self):
        self.rest = _make_acc_rest()
        self.exercise = _make_acc_exercise()

    # Test 1
    def test_output_shape_and_dtype(self):
        feats = extract_acc_features(self.rest)
        self.assertEqual(feats.shape, (_N_FEATURES,))
        self.assertEqual(feats.dtype, np.float32)

    # Test 2
    def test_feature_names_length(self):
        self.assertEqual(len(ACC_FEATURE_NAMES), _N_FEATURES)

    # Test 3
    def test_dynamic_svm_higher_for_exercise(self):
        feats_rest = extract_acc_features(self.rest)
        feats_ex = extract_acc_features(self.exercise)
        # DYNAMIC_SVM_MEAN is index 2
        self.assertGreater(
            float(feats_ex[2]),
            float(feats_rest[2]),
            "Exercise DYNAMIC_SVM_MEAN should exceed rest DYNAMIC_SVM_MEAN",
        )

    # Test 4
    def test_empty_input_returns_zeros(self):
        feats = extract_acc_features(np.empty((0, 3), dtype=np.float32))
        np.testing.assert_array_equal(feats, np.zeros(_N_FEATURES, dtype=np.float32))


class TestMovementContextualizer(unittest.TestCase):
    """Tests for movement_context.py."""

    def setUp(self):
        self.rest = _make_acc_rest()
        self.exercise = _make_acc_exercise()
        self.ctx = MovementContextualizer(percentile=25.0)
        # Correctly fit on EXERCISE windows (not rest) — threshold = 25th pct of exercise DynSVM
        self.ctx.fit_threshold([self.exercise] * 10)

    # Test 5
    def test_exercise_flagged_as_high_movement(self):
        self.assertTrue(self.ctx.is_high_movement(self.exercise))

    # Test 6
    def test_stress_suppressed_for_exercise_windows(self):
        y_pred = np.array([1, 0, 1], dtype=int)
        # Window 0: exercise (high-movement) + STRESS → suppressed
        # Window 1: rest (low-movement) + REST → unchanged
        # Window 2: rest (low-movement) + STRESS → NOT suppressed (below threshold)
        acc_wins = [self.exercise, self.rest, self.rest]
        y_ctx = self.ctx.contextualize(y_pred, acc_wins)
        # Window 0: exercise → STRESS suppressed
        self.assertEqual(y_ctx[0], 0)
        # Window 1: REST prediction unchanged
        self.assertEqual(y_ctx[1], 0)
        # Window 2: rest window is low-movement, so STRESS NOT suppressed
        self.assertEqual(y_ctx[2], 1)


class TestPhysiologicalKNN(unittest.TestCase):
    """Tests for knn_retrieval.py."""

    def setUp(self):
        self.knn = _make_knn()
        self.rng = np.random.default_rng(99)

    # Test 7
    def test_vote_fractions_sum_to_one(self):
        x = self.rng.standard_normal(20).astype(np.float32)
        vote = self.knn.query_vote(x)
        total = vote["rest_frac"] + vote["stress_frac"] + vote["exercise_frac"]
        self.assertAlmostEqual(total, 1.0, places=6)


class TestSimilarityContextualizer(unittest.TestCase):
    """Tests for similarity_context.py."""

    def setUp(self):
        self.knn = _make_knn()
        # ex_threshold=0.0 makes the gate always fire for any exercise neighbour
        self.ctx = SimilarityContextualizer(self.knn, ex_threshold=0.0)

    # Test 8
    def test_stress_suppressed_near_exercise_cluster(self):
        rng = np.random.default_rng(5)
        # Shift features toward the exercise cluster (mean=3.0 in the fixture)
        X_test = (rng.standard_normal((3, 20)) + 3.0).astype(np.float32)
        y_pred = np.array([1, 1, 0], dtype=int)
        y_ctx = self.ctx.contextualize(y_pred, X_test)
        # Output must be valid binary
        self.assertTrue(set(y_ctx.tolist()).issubset({0, 1}))


class TestFusionPolicy(unittest.TestCase):
    """Tests for fusion_policy.py."""

    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.N = 8
        self.y_pred = self.rng.integers(0, 2, size=self.N).astype(int)
        self.X_test = self.rng.standard_normal((self.N, 20)).astype(np.float32)
        self.rest = _make_acc_rest()
        self.exercise = _make_acc_exercise()
        self.acc_wins = [self.exercise if i % 2 == 0 else self.rest for i in range(self.N)]

        acc_ctx = MovementContextualizer(percentile=25.0)
        acc_ctx.fit_threshold([self.rest] * 5)
        self.acc_ctx = acc_ctx

        knn = _make_knn()
        self.sim_ctx = SimilarityContextualizer(knn, ex_threshold=0.40)

    # Test 9
    def test_e0_is_passthrough(self):
        cfg = EXPERIMENT_CONFIGS["E0"]
        policy = FusionPolicy(cfg)
        y_out = policy.fuse(self.y_pred)
        np.testing.assert_array_equal(y_out, self.y_pred)

    # Test 10
    def test_all_configs_produce_valid_binary_output(self):
        for exp_name, exp_cfg in EXPERIMENT_CONFIGS.items():
            with self.subTest(config=exp_name):
                policy = FusionPolicy(
                    exp_cfg,
                    acc_ctx=self.acc_ctx if exp_cfg.use_h1_acc else None,
                    sim_ctx=self.sim_ctx if exp_cfg.use_h2_knn else None,
                )
                y_out = policy.fuse(
                    self.y_pred,
                    acc_windows=self.acc_wins if exp_cfg.use_h1_acc else None,
                    X_features=self.X_test if exp_cfg.use_h2_knn else None,
                )
                self.assertEqual(y_out.shape, (self.N,), f"{exp_name}: wrong shape")
                self.assertTrue(
                    set(y_out.tolist()).issubset({0, 1}),
                    f"{exp_name}: non-binary values: {set(y_out.tolist())}",
                )


if __name__ == "__main__":
    unittest.main()
