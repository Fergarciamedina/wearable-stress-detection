"""
Temporal Tensor Builder for 1D-CNN
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Prepares synchronized temporal waveform tensors (Channels × TimeSteps) for 1D-CNN:
- Common target sampling rate: 16 Hz (30 seconds = 480 time steps).
- EDA resampled from 4 Hz to 16 Hz.
- BVP/PPG resampled from 64 Hz to 16 Hz.
- Channels: 1 for EDA, 1 for PPG, 2 for EDA+PPG fusion.
"""

from typing import Dict, Any, Optional
import numpy as np
from scipy import signal


class TensorBuilder:
    """Builds synchronized fixed-length 1D-CNN input tensors from 30-second windows."""

    def __init__(self, target_fs: float = 16.0, window_len_s: float = 30.0):
        self.target_fs = target_fs
        self.window_len_s = window_len_s
        self.num_time_steps = int(round(target_fs * window_len_s))  # 16 * 30 = 480

    def resample_signal(self, sig: np.ndarray, orig_fs: float) -> np.ndarray:
        """Resamples signal to common target frequency using Fourier method or linear interpolation."""
        if len(sig) == 0:
            return np.zeros(self.num_time_steps, dtype=np.float32)
        
        num_target_samples = self.num_time_steps
        if orig_fs == self.target_fs and len(sig) == num_target_samples:
            return sig.astype(np.float32)

        resampled = signal.resample(sig, num_target_samples)
        return resampled.astype(np.float32)

    def build_tensor(self, window_data: Dict[str, Any], sensor_config: str) -> np.ndarray:
        """
        Builds temporal tensor array of shape (Channels, TimeSteps):
            - 'EDA': (1, 480)
            - 'PPG': (1, 480)
            - 'EDA+PPG': (2, 480)
        """
        eda_raw = window_data["eda_window"]
        eda_fs = 4.0
        eda_resampled = self.resample_signal(eda_raw, eda_fs)

        bvp_raw = window_data.get("bvp_window")
        bvp_fs = 64.0
        if bvp_raw is not None and len(bvp_raw) > 0:
            bvp_resampled = self.resample_signal(bvp_raw, bvp_fs)
        else:
            bvp_resampled = np.zeros(self.num_time_steps, dtype=np.float32)

        if sensor_config == "EDA":
            # Shape (1, 480)
            return np.expand_dims(eda_resampled, axis=0)
        elif sensor_config == "PPG":
            # Shape (1, 480)
            return np.expand_dims(bvp_resampled, axis=0)
        elif sensor_config in ["EDA+PPG", "FUSION"]:
            # Shape (2, 480)
            return np.stack([eda_resampled, bvp_resampled], axis=0)
        else:
            raise ValueError(f"Unknown sensor configuration: {sensor_config}")
