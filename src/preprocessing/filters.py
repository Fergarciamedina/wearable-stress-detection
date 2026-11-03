"""
Physiological Signal Filtering and Preprocessing
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Implements scientifically justified filtering:
- EDA: Butterworth low-pass (cutoff 1.0 Hz) and tonic/phasic separation.
- BVP/PPG: Butterworth band-pass (0.5 Hz - 8.0 Hz) to isolate cardiac pulses.
"""

from typing import Tuple
import numpy as np
from scipy import signal


def filter_eda(eda_signal: np.ndarray, fs: float = 4.0, cutoff: float = 1.0, order: int = 4) -> np.ndarray:
    """
    Applies zero-phase Butterworth low-pass filter to raw EDA.
    Physiological EDA frequencies are concentrated below 0.5 - 1.0 Hz.
    """
    if len(eda_signal) < (order * 3):
        return eda_signal
    
    nyquist = 0.5 * fs
    normal_cutoff = min(cutoff / nyquist, 0.99)
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered = signal.filtfilt(b, a, eda_signal)
    return np.clip(filtered, a_min=0.0, a_max=None)


def decompose_eda(eda_signal: np.ndarray, fs: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes filtered EDA into Tonic (SCL) and Phasic (SCR) components.
    Tonic represents slowly varying baseline conductance (cutoff 0.05 Hz).
    Phasic represents rapid skin conductance responses (SCR).
    """
    if len(eda_signal) < int(fs * 10):
        return eda_signal, np.zeros_like(eda_signal)

    nyquist = 0.5 * fs
    normal_cutoff = min(0.05 / nyquist, 0.99)
    b, a = signal.butter(2, normal_cutoff, btype='low', analog=False)
    tonic = signal.filtfilt(b, a, eda_signal)
    tonic = np.clip(tonic, a_min=0.0, a_max=None)
    phasic = np.clip(eda_signal - tonic, a_min=0.0, a_max=None)
    return tonic, phasic


def filter_bvp(bvp_signal: np.ndarray, fs: float = 64.0, lowcut: float = 0.5, highcut: float = 8.0, order: int = 4) -> np.ndarray:
    """
    Applies zero-phase Butterworth band-pass filter to raw BVP/PPG.
    Removes baseline wander/respiratory drift (< 0.5 Hz) and sensor noise (> 8.0 Hz).
    """
    if len(bvp_signal) < (order * 3):
        return bvp_signal

    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = min(highcut / nyquist, 0.99)
    b, a = signal.butter(order, [low, high], btype='bandpass', analog=False)
    filtered = signal.filtfilt(b, a, bvp_signal)
    return filtered
