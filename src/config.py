"""
Centralized Configuration & Path Resolution System
COEN 6322: AI for Wearable Health Applications

Resolves paths dynamically across environments:
1. Environment variables (WEARABLE_DATA_DIR, WEARABLE_RESULTS_DIR)
2. CLI arguments passed by entry points
3. Fallback to canonical repository layout (parent data/ folder)
"""

import os
from pathlib import Path
from typing import Optional


class ProjectConfig:
    """Manages paths and runtime parameters for training, evaluation, and inference."""

    def __init__(self, data_root: Optional[str] = None):
        # 1. Determine base directories
        self.code_dir = Path(__file__).resolve().parent.parent  # .../code
        self.project_dir = self.code_dir.parent                # .../project

        # 2. Determine data root
        if data_root:
            self.data_dir = Path(data_root).resolve()
        elif os.environ.get("WEARABLE_DATA_DIR"):
            self.data_dir = Path(os.environ["WEARABLE_DATA_DIR"]).resolve()
        elif (self.project_dir / "data").exists():
            self.data_dir = self.project_dir / "data"
        elif (self.code_dir / "data").exists():
            self.data_dir = self.code_dir / "data"
        else:
            # Fallback relative to project
            self.data_dir = self.project_dir / "data"

        # 3. Canonical Subdirectories
        self.raw_data_dir = self.data_dir / "public" / "raw" / "DS1_Hongn_Stress_Exercise"
        self.splits_dir = self.data_dir / "public" / "splits"
        self.results_dir = self.data_dir / "results"
        self.metrics_dir = self.results_dir / "metrics"
        self.checkpoints_dir = self.results_dir / "checkpoints"
        self.figures_dir = self.results_dir / "figures"
        self.data_audit_dir = self.results_dir / "data_audit"
        self.llm_logs_dir = self.data_dir / "llm_logs"

        # 4. Standard sampling rates (Empatica E4)
        self.fs_eda = 4.0
        self.fs_bvp = 64.0
        self.fs_acc = 32.0
        self.fs_temp = 4.0
        self.target_fs_cnn = 16.0

        # 5. Window parameters
        self.window_len_s = 30.0
        self.stride_s = 15.0

    def ensure_directories(self) -> None:
        """Ensures all necessary output directories exist."""
        for p in [self.metrics_dir, self.checkpoints_dir, self.figures_dir, self.data_audit_dir, self.llm_logs_dir]:
            p.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"<ProjectConfig data_dir={self.data_dir}>"


# Global default configuration instance
default_config = ProjectConfig()
