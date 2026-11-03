"""
Similarity Contextualizer — H2 Similarity-Aware Contextualization
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Adjusts RF predictions using KNN neighbour-vote fractions:

  * If exercise_frac ≥ ex_threshold  → suppress STRESS → REST  (exercise suppression)
  * If stress_frac   ≥ stress_threshold → reinforce STRESS → STRESS (no-op by default)
  * Otherwise, pass the RF prediction through unchanged.

Thresholds are fixed rather than data-driven to keep the contextualizer
a stateless wrapper on top of the already-trained KNN:

  ex_threshold    = 0.40  (40 % of k=7 neighbours are exercise → 3/7 votes)
  stress_threshold= 0.15  (conservative reinforcement; rarely overrides REST)

Both thresholds can be overridden at construction time to support ablation.
"""

from typing import Dict, List, Optional
import numpy as np

from src.research.similarity_aware.knn_retrieval import PhysiologicalKNN


class SimilarityContextualizer:
    """
    Applies KNN similarity votes to contextualise binary RF predictions.

    Usage
    -----
    >>> ctx = SimilarityContextualizer(knn)
    >>> y_ctx = ctx.contextualize(y_pred, X_test)
    """

    def __init__(
        self,
        knn: PhysiologicalKNN,
        ex_threshold: float = 0.40,
        stress_threshold: float = 0.15,
    ) -> None:
        """
        Parameters
        ----------
        knn : PhysiologicalKNN
            A fitted KNN index (call ``knn.fit()`` before passing here).
        ex_threshold : float
            Minimum fraction of exercise neighbours to trigger suppression.
        stress_threshold : float
            Minimum fraction of stress neighbours to reinforce a STRESS prediction.
            (Currently used only for logging; no prediction is changed from REST→STRESS.)
        """
        self.knn = knn
        self.ex_threshold = ex_threshold
        self.stress_threshold = stress_threshold

    # ------------------------------------------------------------------
    # Inference-time interface
    # ------------------------------------------------------------------

    def contextualize(
        self,
        y_pred: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        """
        Applies similarity-aware contextualization to a batch of predictions.

        Parameters
        ----------
        y_pred : np.ndarray, shape (N,)
            Binary RF predictions (0 = REST, 1 = STRESS).
        X_test : np.ndarray, shape (N, D)
            Feature matrix for the same N windows.

        Returns
        -------
        np.ndarray, shape (N,), dtype int
            Contextualized predictions.  Only STRESS→REST overrides are applied.
        """
        votes: List[Dict[str, float]] = self.knn.query_votes_batch(X_test)
        y_ctx = np.array(y_pred, dtype=int)

        for i, vote in enumerate(votes):
            if y_ctx[i] == 1 and vote["exercise_frac"] >= self.ex_threshold:
                # Neighbour evidence suggests exercise context → suppress alarm
                y_ctx[i] = 0

        return y_ctx

    def contextualize_with_votes(
        self,
        y_pred: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple:
        """
        Same as ``contextualize`` but also returns the per-window vote dicts.

        Returns
        -------
        y_ctx : np.ndarray, shape (N,)
        votes : list of dict, length N
        """
        votes: List[Dict[str, float]] = self.knn.query_votes_batch(X_test)
        y_ctx = np.array(y_pred, dtype=int)

        for i, vote in enumerate(votes):
            if y_ctx[i] == 1 and vote["exercise_frac"] >= self.ex_threshold:
                y_ctx[i] = 0

        return y_ctx, votes

