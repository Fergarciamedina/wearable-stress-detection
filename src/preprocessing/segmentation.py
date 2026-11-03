"""
Window Segmentation Module (30-second temporal windows)
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Segments signals into boundary-safe 30-second windows.
Strictly ensures windows never cross subject, session, or protocol boundaries.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class WindowSegmenter:
    """Segments continuous signals into 30-second labeled windows."""
    
    def __init__(self, window_len_s: float = 30.0, stride_s: float = 15.0):
        self.window_len_s = window_len_s
        self.stride_s = stride_s

    def segment_stress_session(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Segments a subject's STRESS session into labeled 30-second windows.
        Windows are extracted strictly within documented phase boundaries.
        Returns list of window dicts:
            {
                'subject_id': str,
                'session_type': 'STRESS',
                'phase': str,
                'label': int (0 for REST, 1 for STRESS),
                'eda_window': np.ndarray,
                'bvp_window': np.ndarray (or None if invalid),
                'window_start_s': float,
                'window_end_s': float
            }
        """
        windows = []
        subject_id = session_data["subject_id"]
        eda_sig = session_data["eda_signal"]
        eda_fs = session_data["eda_fs"]
        bvp_sig = session_data["bvp_signal"]
        bvp_fs = session_data["bvp_fs"]
        bvp_valid = session_data["bvp_valid"]

        for interval in session_data["labeled_intervals"]:
            start_s = interval["start_s"]
            end_s = interval["end_s"]
            label = interval["label"]
            phase = interval["phase"]
            
            interval_duration = end_s - start_s
            if interval_duration < self.window_len_s:
                continue

            curr_start = start_s
            while (curr_start + self.window_len_s) <= end_s:
                curr_end = curr_start + self.window_len_s
                
                # EDA slice
                eda_idx_start = int(round(curr_start * eda_fs))
                eda_idx_end = int(round(curr_end * eda_fs))
                
                if eda_idx_end > len(eda_sig):
                    break
                    
                eda_win = eda_sig[eda_idx_start:eda_idx_end]
                
                # BVP slice if valid
                bvp_win = None
                if bvp_valid and bvp_fs > 0 and len(bvp_sig) > 0:
                    bvp_idx_start = int(round(curr_start * bvp_fs))
                    bvp_idx_end = int(round(curr_end * bvp_fs))
                    if bvp_idx_end <= len(bvp_sig):
                        bvp_win = bvp_sig[bvp_idx_start:bvp_idx_end]

                windows.append({
                    "subject_id": subject_id,
                    "session_type": "STRESS",
                    "phase": phase,
                    "label": label,
                    "eda_window": eda_win,
                    "bvp_window": bvp_win,
                    "bvp_valid": bvp_win is not None,
                    "window_start_s": curr_start,
                    "window_end_s": curr_end
                })

                curr_start += self.stride_s

        return windows

    def segment_exercise_session(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Segments exercise sessions into 30-second windows for the false-alarm study.
        No training labels assigned; evaluated as held-out challenge.
        """
        windows = []
        subject_id = session_data["subject_id"]
        session_type = session_data["session_type"]
        eda_sig = session_data["eda_signal"]
        eda_fs = session_data["eda_fs"]
        bvp_sig = session_data["bvp_signal"]
        bvp_fs = session_data["bvp_fs"]
        bvp_valid = session_data["bvp_valid"]

        total_duration_s = len(eda_sig) / eda_fs if eda_fs > 0 else 0.0
        curr_start = 0.0

        while (curr_start + self.window_len_s) <= total_duration_s:
            curr_end = curr_start + self.window_len_s
            
            eda_idx_start = int(round(curr_start * eda_fs))
            eda_idx_end = int(round(curr_end * eda_fs))
            if eda_idx_end > len(eda_sig):
                break
            eda_win = eda_sig[eda_idx_start:eda_idx_end]

            bvp_win = None
            if bvp_valid and bvp_fs > 0 and len(bvp_sig) > 0:
                bvp_idx_start = int(round(curr_start * bvp_fs))
                bvp_idx_end = int(round(curr_end * bvp_fs))
                if bvp_idx_end <= len(bvp_sig):
                    bvp_win = bvp_sig[bvp_idx_start:bvp_idx_end]

            windows.append({
                "subject_id": subject_id,
                "session_type": session_type,
                "phase": session_type,
                "label": -1,  # Challenge set, no training label
                "eda_window": eda_win,
                "bvp_window": bvp_win,
                "bvp_valid": bvp_win is not None,
                "window_start_s": curr_start,
                "window_end_s": curr_end
            })

            curr_start += self.stride_s

        return windows
