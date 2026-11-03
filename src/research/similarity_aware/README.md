# H2: Similarity-Aware Contextualization

COEN 6322: AI for Wearable Health Applications, Fall 2026

## Objective

The Similarity-Aware Contextualization module (H2) investigates whether non-parametric exemplar retrieval in autonomic feature space (EDA + PPG, 20 dimensions) can identify exercise context without relying on accelerometer signals.

## Physiological KNN Retrieval (`knn_retrieval.py`)

- **Backend:** Exact $k$-Nearest-Neighbors index ($k=7$) using Euclidean distance on standard-scaled features.
- **Exemplar Indexing:**
  - Training subjects' labeled stress sessions: `REST (0)` and `STRESS (1)`.
  - Training subjects' exercise sessions: `EXERCISE (2)`.
- **Query Output:** For any candidate window, returns the proportion of nearest neighbors belonging to each class (`rest_frac`, `stress_frac`, `exercise_frac`).

## Contextualizer (`similarity_context.py`)

- **Exercise Suppression Gate:** If the primary RF model predicts `STRESS (1)`, but `exercise_frac >= 0.40` ($\ge 3$ of 7 nearest neighbors are exercise windows), the prediction is overridden to `REST (0)`.
- **Zero Leakage:** The KNN index is fitted strictly on the $N-1$ training subjects per fold.

