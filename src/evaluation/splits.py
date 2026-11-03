"""
Subject-Disjoint Cross-Validation Splitting Module
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Generates nested subject-disjoint partitions:
- Outer Loop: Leave-One-Subject-Out (LOSO) holding out one entire participant.
- Inner Loop: Grouped validation partitioning strictly within the N-1 development subjects.
Persists exact split manifests to dataset/splits/loso_splits.json.
"""

import os
import json
import random
from typing import Dict, List, Any


def generate_loso_manifest(eligible_subjects: List[str], inner_val_ratio: float = 0.2, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Creates nested LOSO folds.
    For each test subject:
        - Remaining subjects are development subjects.
        - A subset of development subjects is reserved as inner validation.
        - Remainder are inner training subjects.
    Zero data leakage guaranteed.
    """
    random.seed(seed)
    manifest = []
    
    for i, test_subj in enumerate(eligible_subjects):
        dev_subjects = [s for s in eligible_subjects if s != test_subj]
        
        # Deterministic pseudo-random inner validation split per fold
        rng = random.Random(seed + i)
        shuffled_dev = list(dev_subjects)
        rng.shuffle(shuffled_dev)
        
        num_val = max(1, int(round(len(dev_subjects) * inner_val_ratio)))
        val_subjects = sorted(shuffled_dev[:num_val])
        train_subjects = sorted(shuffled_dev[num_val:])

        manifest.append({
            "fold_idx": i,
            "test_subject": test_subj,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "all_dev_subjects": sorted(dev_subjects)
        })

    return manifest


def save_splits_manifest(manifest: List[Dict[str, Any]], filepath: str):
    """Saves fold manifest to a persistent JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_splits_manifest(filepath: str) -> List[Dict[str, Any]]:
    """Loads fold manifest from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _curr = Path(__file__).resolve()
    for _p in [_curr.parent, _curr.parent.parent, _curr.parent.parent.parent]:
        if (_p / "src").is_dir():
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
            break

    from src.config import default_config

    subjects = [f"S{i:02d}" for i in range(1, 19) if i != 2] + [f"f{i:02d}" for i in range(1, 19) if i != 7]
    manifest = generate_loso_manifest(subjects, seed=42)
    out_path = default_config.splits_dir / "loso_splits.json"
    save_splits_manifest(manifest, str(out_path))
    print(f"Generated {len(manifest)} LOSO splits at: {out_path}")
