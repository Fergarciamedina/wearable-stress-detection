"""
Physiological KNN Retrieval — H2 Similarity-Aware Contextualization
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Builds an exact k-Nearest-Neighbours index over training physiological features
(EDA+PPG, 20 dimensions) and returns the label distribution among the k=7
nearest training windows for each test window.

Label convention
----------------
  0 = REST
  1 = STRESS
  2 = EXERCISE  (exercise windows are included in the reference index)

The KNN is trained on:
  - All labeled stress-session windows (REST and STRESS) from training subjects.
  - All exercise-session windows from training subjects (assigned label 2).

Exercise windows are included in the reference set so that the similarity
vote can signal "this looks like exercise", which H2 uses to suppress
false STRESS predictions.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class PhysiologicalKNN:
    """
    Exact k-NN physiological similarity index (scikit-learn backend).

    Usage
    -----
    >>> knn = PhysiologicalKNN(k=7)
    >>> knn.fit(X_rest_stress, y_rest_stress, X_exercise)
    >>> vote = knn.query_vote(x_test_window)
    """

    def __init__(self, k: int = 7, metric: str = "euclidean") -> None:
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbours to retrieve (default 7).
        metric : str
            Distance metric passed to ``NearestNeighbors`` (default 'euclidean').
        """
        self.k = k
        self.metric = metric
        self._nn: Optional[NearestNeighbors] = None
        self._labels: Optional[np.ndarray] = None
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Training-time interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_stress_session: np.ndarray,
        y_stress_session: np.ndarray,
        X_exercise: Optional[np.ndarray] = None,
    ) -> "PhysiologicalKNN":
        """
        Builds the reference index from training windows.

        Parameters
        ----------
        X_stress_session : np.ndarray, shape (N_rs, D)
            Feature matrix of REST (label 0) and STRESS (label 1) windows.
        y_stress_session : np.ndarray, shape (N_rs,)
            Labels (0 or 1) for each stress-session window.
        X_exercise : np.ndarray or None, shape (N_ex, D)
            Feature matrix of exercise windows.  If provided, all are assigned
            label 2 and appended to the reference index.

        Returns
        -------
        self
        """
        X_rs = np.asarray(X_stress_session, dtype=np.float32)
        y_rs = np.asarray(y_stress_session, dtype=int)

        if X_exercise is not None and len(X_exercise) > 0:
            X_ex = np.asarray(X_exercise, dtype=np.float32)
            y_ex = np.full(len(X_ex), fill_value=2, dtype=int)
            X_all = np.concatenate([X_rs, X_ex], axis=0)
            y_all = np.concatenate([y_rs, y_ex], axis=0)
        else:
            X_all = X_rs
            y_all = y_rs

        # Standardise in the reference space using the training distribution
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_all)

        self._nn = NearestNeighbors(n_neighbors=min(self.k, len(X_scaled)), metric=self.metric, algorithm="ball_tree")
        self._nn.fit(X_scaled)
        self._labels = y_all
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference-time interface
    # ------------------------------------------------------------------

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self._scaler is None:
            raise RuntimeError("Call fit() first.")
        return self._scaler.transform(np.atleast_2d(X))

    def query_neighbours(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (distances, neighbour_labels) for a single feature vector.

        Parameters
        ----------
        x : np.ndarray, shape (D,)

        Returns
        -------
        distances : np.ndarray, shape (k,)
        labels : np.ndarray, shape (k,)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before query_neighbours().")
        x_scaled = self._scale(x.reshape(1, -1))
        dists, indices = self._nn.kneighbors(x_scaled)
        return dists.ravel(), self._labels[indices.ravel()]

    def query_vote(self, x: np.ndarray) -> Dict[str, float]:
        """
        Returns the fraction of k neighbours in each class (0, 1, 2).

        Parameters
        ----------
        x : np.ndarray, shape (D,)

        Returns
        -------
        dict with keys 'rest_frac', 'stress_frac', 'exercise_frac'
        """
        _, labels = self.query_neighbours(x)
        k_actual = len(labels)
        rest_frac = float(np.sum(labels == 0)) / k_actual
        stress_frac = float(np.sum(labels == 1)) / k_actual
        exercise_frac = float(np.sum(labels == 2)) / k_actual
        return {
            "rest_frac": rest_frac,
            "stress_frac": stress_frac,
            "exercise_frac": exercise_frac,
        }

    def query_votes_batch(self, X: np.ndarray) -> List[Dict[str, float]]:
        """
        Vectorised query over a batch of feature vectors.

        Parameters
        ----------
        X : np.ndarray, shape (N, D)

        Returns
        -------
        list of dicts, length N — same format as ``query_vote``
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before query_votes_batch().")
        X_scaled = self._scale(X)
        _, all_indices = self._nn.kneighbors(X_scaled)
        votes: List[Dict[str, float]] = []
        k_actual = all_indices.shape[1]
        for row in all_indices:
            lbls = self._labels[row]
            votes.append({
                "rest_frac": float(np.sum(lbls == 0)) / k_actual,
                "stress_frac": float(np.sum(lbls == 1)) / k_actual,
                "exercise_frac": float(np.sum(lbls == 2)) / k_actual,
            })
        return votes

