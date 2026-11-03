"""
Handcrafted Physiological Feature Extraction Module (for Random Forest)
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Extracts domain-guided features from 30-second temporal windows:
- EDA: Statistical moments, tonic conductance level, phasic SCR peak metrics.
- BVP/PPG: Statistical moments, peak-to-peak interval dynamics, RMSSD, pulse rate, spectral energy.
- Fusion: Synchronized multimodal concatenation.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import signal


class FeatureExtractor:
    """Extracts handcrafted statistical and physiological features for classical models."""

    def __init__(self, eda_fs: float = 4.0, bvp_fs: float = 64.0):
        self.eda_fs = eda_fs
        self.bvp_fs = bvp_fs

    def extract_eda_features(self, eda_window: np.ndarray) -> np.ndarray:
        """
        Extracts 10 domain features from a 30-second EDA window.
        1. Mean
        2. Standard Deviation
        3. Minimum
        4. Maximum
        5. Range (Max - Min)
        6. Slope (linear trend over window)
        7. Tonic mean level
        8. SCR peak count
        9. Max SCR peak amplitude
        10. Sum of SCR peak amplitudes
        """
        if len(eda_window) == 0:
            return np.zeros(10, dtype=np.float32)

        mean_val = float(np.mean(eda_window))
        std_val = float(np.std(eda_window))
        min_val = float(np.min(eda_window))
        max_val = float(np.max(eda_window))
        range_val = max_val - min_val

        # Linear slope
        x = np.arange(len(eda_window))
        if len(eda_window) > 1:
            slope = float(np.polyfit(x, eda_window, 1)[0])
        else:
            slope = 0.0

        # Phasic decomposition approximation on window
        # Tonic baseline via moving average
        kernel_size = max(1, int(self.eda_fs * 5))
        tonic = np.convolve(eda_window, np.ones(kernel_size) / kernel_size, mode='same')
        phasic = np.clip(eda_window - tonic, a_min=0.0, a_max=None)
        tonic_mean = float(np.mean(tonic))

        # Peak detection on phasic signal (SCR peaks with min height 0.02 uS)
        peaks, properties = signal.find_peaks(phasic, height=0.01, distance=int(self.eda_fs * 1.0))
        peak_count = float(len(peaks))
        if len(peaks) > 0:
            peak_heights = properties.get("peak_heights", np.array([0.0]))
            max_amp = float(np.max(peak_heights))
            sum_amp = float(np.sum(peak_heights))
        else:
            max_amp = 0.0
            sum_amp = 0.0

        features = np.array([
            mean_val, std_val, min_val, max_val, range_val,
            slope, tonic_mean, peak_count, max_amp, sum_amp
        ], dtype=np.float32)

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_bvp_features(self, bvp_window: np.ndarray) -> np.ndarray:
        """
        Extracts 10 domain features from a 30-second BVP window.
        1. Mean
        2. Standard Deviation
        3. Range
        4. Energy (mean squared signal)
        5. Estimated Pulse Rate (BPM from systolic peaks)
        6. Mean Peak-to-Peak Interval (PPI in seconds)
        7. Std of Peak-to-Peak Intervals (SDRR)
        8. RMSSD (Root Mean Square of Successive Differences)
        9. Low-frequency band power (0.5 - 2.0 Hz)
        10. High-frequency band power (2.0 - 5.0 Hz)
        """
        if bvp_window is None or len(bvp_window) == 0:
            return np.zeros(10, dtype=np.float32)

        mean_val = float(np.mean(bvp_window))
        std_val = float(np.std(bvp_window))
        range_val = float(np.ptp(bvp_window))
        energy = float(np.mean(bvp_window ** 2))

        # Systolic peak detection (minimum distance 0.4s -> 150 BPM max)
        min_dist = max(1, int(self.bvp_fs * 0.4))
        peaks, _ = signal.find_peaks(bvp_window, distance=min_dist, prominence=std_val * 0.5)

        if len(peaks) >= 2:
            ppi_samples = np.diff(peaks)
            ppi_seconds = ppi_samples / self.bvp_fs
            mean_ppi = float(np.mean(ppi_seconds))
            std_ppi = float(np.std(ppi_seconds))
            pulse_rate_bpm = 60.0 / mean_ppi if mean_ppi > 0 else 0.0

            # RMSSD
            diff_ppi = np.diff(ppi_seconds)
            if len(diff_ppi) > 0:
                rmssd = float(np.sqrt(np.mean(diff_ppi ** 2)))
            else:
                rmssd = 0.0
        else:
            mean_ppi = 0.0
            std_ppi = 0.0
            pulse_rate_bpm = 0.0
            rmssd = 0.0

        # Spectral power via Welch periodogram
        freqs, psd = signal.welch(bvp_window, fs=self.bvp_fs, nperseg=min(len(bvp_window), int(self.bvp_fs * 8)))
        lf_mask = (freqs >= 0.5) & (freqs <= 2.0)
        hf_mask = (freqs > 2.0) & (freqs <= 5.0)

        lf_power = float(np.sum(psd[lf_mask])) if np.any(lf_mask) else 0.0
        hf_power = float(np.sum(psd[hf_mask])) if np.any(hf_mask) else 0.0

        features = np.array([
            mean_val, std_val, range_val, energy, pulse_rate_bpm,
            mean_ppi, std_ppi, rmssd, lf_power, hf_power
        ], dtype=np.float32)

        return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_features_for_sensor_config(self, window_data: Dict[str, Any], sensor_config: str) -> np.ndarray:
        """
        Extracts feature vector for a specified sensor configuration:
            - 'EDA': 10 features
            - 'PPG': 10 features
            - 'EDA+PPG': 20 features
        """
        eda_feat = self.extract_eda_features(window_data["eda_window"])
        
        bvp_win = window_data.get("bvp_window")
        bvp_feat = self.extract_bvp_features(bvp_win) if (bvp_win is not None) else np.zeros(10, dtype=np.float32)

        if sensor_config == "EDA":
            return eda_feat
        elif sensor_config == "PPG":
            return bvp_feat
        elif sensor_config in ["EDA+PPG", "FUSION"]:
            return np.concatenate([eda_feat, bvp_feat])
        else:
            raise ValueError(f"Unknown sensor configuration: {sensor_config}")
