# Fusion Policy: Combined Contextualization (E0–E3)

COEN 6322: AI for Wearable Health Applications, Fall 2026

## Overview

The `FusionPolicy` orchestrates contextualization layers over the base Random Forest classifier (EDA + PPG, 20 features) under a structured experimental matrix:

| Configuration | ACC Gate (H1) | KNN Similarity (H2) | Description |
|:---:|:---:|:---:|:---|
| **E0** | No | No | Uncontextualized base Random Forest baseline |
| **E1** | Yes | No | RF + H1 ACC-aware movement gate |
| **E2** | No | Yes | RF + H2 KNN physiological similarity |
| **E3** | Yes | Yes | Combined: H1 gate applied first, followed by H2 similarity |

## Execution Order in E3

When both contextualizers are active (`E3`):
1. **Stage 1 (H1 — ACC Gate):** High-speed rule check. If dynamic acceleration exceeds the movement threshold, `STRESS (1) -> REST (0)`.
2. **Stage 2 (H2 — KNN Similarity):** Non-parametric check. If the remaining `STRESS` predictions exhibit high similarity to exercise exemplars (`exercise_frac >= 0.40`), `STRESS (1) -> REST (0)`.

