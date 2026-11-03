"""
Fusion Policy — Combined H1 + H2 Contextualization
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Defines four experiment configurations (E0–E3) that combine the base RF
classifier with ACC-aware (H1) and similarity-aware (H2) contextualizers:

  E0  — Baseline RF only                         (no contextualization)
  E1  — RF + H1 (ACC gate only)
  E2  — RF + H2 (KNN similarity only)
  E3  — RF + H1 + H2 (both, applied sequentially)

Also defines C_3CLASS as a 3-class control that encodes exercise windows
as a dedicated label during training, giving the RF direct knowledge of the
EXERCISE class.

Application order for E3:
  1. H1 (ACC gate) is applied first — fast, rule-based.
  2. H2 (KNN similarity) is applied second — can further suppress alarms
     that H1 missed (e.g., sedentary exercise or missing ACC).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from src.research.acc_aware.movement_context import MovementContextualizer
from src.research.similarity_aware.similarity_context import SimilarityContextualizer


@dataclass
class ExperimentConfig:
    """Metadata for a single fusion experiment."""

    name: str
    use_h1_acc: bool = False
    use_h2_knn: bool = False
    description: str = ""


# Canonical experiment configurations
EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "E0": ExperimentConfig(
        name="E0",
        use_h1_acc=False,
        use_h2_knn=False,
        description="Baseline Random Forest (no contextualization)",
    ),
    "E1": ExperimentConfig(
        name="E1",
        use_h1_acc=True,
        use_h2_knn=False,
        description="RF + H1 ACC-aware movement gate",
    ),
    "E2": ExperimentConfig(
        name="E2",
        use_h1_acc=False,
        use_h2_knn=True,
        description="RF + H2 KNN physiological similarity",
    ),
    "E3": ExperimentConfig(
        name="E3",
        use_h1_acc=True,
        use_h2_knn=True,
        description="RF + H1 ACC-gate + H2 KNN similarity (combined)",
    ),
}


class FusionPolicy:
    """
    Applies the configured combination of contextualizers to RF predictions.

    Parameters
    ----------
    config : ExperimentConfig
        Which contextualizers to activate.
    acc_ctx : MovementContextualizer or None
        Fitted H1 contextualizer (required when ``config.use_h1_acc`` is True).
    sim_ctx : SimilarityContextualizer or None
        Fitted H2 contextualizer (required when ``config.use_h2_knn`` is True).
    """

    def __init__(
        self,
        config: ExperimentConfig,
        acc_ctx: Optional[MovementContextualizer] = None,
        sim_ctx: Optional[SimilarityContextualizer] = None,
    ) -> None:
        self.config = config
        self.acc_ctx = acc_ctx
        self.sim_ctx = sim_ctx

    def fuse(
        self,
        y_pred: np.ndarray,
        acc_windows: Optional[List[np.ndarray]] = None,
        X_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Applies the fusion policy to a batch of RF predictions.

        Parameters
        ----------
        y_pred : np.ndarray, shape (N,)
            Base RF predictions (0 = REST, 1 = STRESS).
        acc_windows : list of np.ndarray (N×3) or None
            ACC windows aligned with each prediction.
            Required when ``config.use_h1_acc`` is True.
        X_features : np.ndarray, shape (N, D) or None
            EDA+PPG feature matrix for KNN query.
            Required when ``config.use_h2_knn`` is True.

        Returns
        -------
        np.ndarray, shape (N,), dtype int
            Contextualized predictions.
        """
        y_out = np.array(y_pred, dtype=int)

        # Step 1: H1 — ACC movement gate
        if self.config.use_h1_acc:
            if self.acc_ctx is None:
                raise ValueError("acc_ctx required for H1 but not provided.")
            if acc_windows is None:
                raise ValueError("acc_windows required for H1 but not provided.")
            y_out = self.acc_ctx.contextualize(y_out, acc_windows)

        # Step 2: H2 — KNN similarity
        if self.config.use_h2_knn:
            if self.sim_ctx is None:
                raise ValueError("sim_ctx required for H2 but not provided.")
            if X_features is None:
                raise ValueError("X_features required for H2 but not provided.")
            y_out = self.sim_ctx.contextualize(y_out, X_features)

        return y_out

