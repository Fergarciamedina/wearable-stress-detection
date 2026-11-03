"""
Signal Loader and Protocol Label Reconstructor
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Reconstructs exact REST vs. STRESS intervals from Empatica E4 tags and stage durations.
Extracts exercise sessions as out-of-distribution challenge sets.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class SignalLoader:
    """Loads raw Empatica E4 physiological signals and event tags."""
    
    def __init__(self, raw_dataset_dir: str):
        self.raw_dir = raw_dataset_dir
        self.wearable_dir = os.path.join(raw_dataset_dir, "Wearable_Dataset")

    @staticmethod
    def parse_empatica_csv(filepath: str) -> Tuple[Optional[datetime], float, np.ndarray]:
        """
        Parses an Empatica E4 CSV file.
        Returns:
            start_datetime: UTC datetime of the first sample
            sampling_rate: frequency in Hz
            signal: 1D numpy array of values (or 2D for ACC)
        """
        if not os.path.exists(filepath):
            return None, 0.0, np.array([], dtype=np.float32)

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = [line.strip() for line in f if line.strip()]

        if len(lines) < 2:
            return None, 0.0, np.array([], dtype=np.float32)

        start_str = lines[0]
        try:
            if "-" in start_str:
                start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
            else:
                start_dt = datetime.utcfromtimestamp(float(start_str))
        except Exception:
            start_dt = None

        try:
            fs = float(lines[1])
        except Exception:
            fs = 0.0

        raw_data = []
        for line in lines[2:]:
            parts = line.split(",")
            if len(parts) == 1:
                try:
                    raw_data.append(float(parts[0]))
                except ValueError:
                    continue
            else:
                try:
                    raw_data.append([float(p) for p in parts])
                except ValueError:
                    continue

        return start_dt, fs, np.array(raw_data, dtype=np.float32)

    @staticmethod
    def parse_tags_csv(filepath: str) -> List[datetime]:
        """Reads event tags into a list of UTC datetime objects."""
        if not os.path.exists(filepath):
            return []
        tags = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    if "-" in line:
                        dt = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
                    else:
                        dt = datetime.utcfromtimestamp(float(line))
                    tags.append(dt)
                except Exception:
                    continue
        return sorted(tags)

    def load_subject_stress_session(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """
        Loads the STRESS folder for a subject and reconstructs ground-truth labeled intervals.
        Enforces documented constraints:
            - S02 excluded due to duplicated raw rows.
            - f07 PPG/TEMP excluded due to dock cover; EDA valid.
            - f14 loaded from f14_b (main stress protocol).
        """
        if subject_id == "S02":
            return None  # Exclude corrupted subject S02

        folder_name = subject_id
        if subject_id == "f14":
            folder_name = "f14_b"

        folder_path = os.path.join(self.wearable_dir, "STRESS", folder_name)
        if not os.path.exists(folder_path):
            return None

        eda_dt, eda_fs, eda_sig = self.parse_empatica_csv(os.path.join(folder_path, "EDA.csv"))
        bvp_dt, bvp_fs, bvp_sig = self.parse_empatica_csv(os.path.join(folder_path, "BVP.csv"))
        tags = self.parse_tags_csv(os.path.join(folder_path, "tags.csv"))

        if eda_dt is None or eda_fs <= 0 or len(eda_sig) == 0:
            return None

        is_v1 = subject_id.startswith("S")
        labeled_intervals = self.reconstruct_stress_intervals(eda_dt, tags, is_v1)

        bvp_valid = (bvp_fs > 0 and len(bvp_sig) > 0 and subject_id != "f07")

        return {
            "subject_id": subject_id,
            "session_type": "STRESS",
            "protocol_version": "V1" if is_v1 else "V2",
            "start_time": eda_dt,
            "eda_fs": eda_fs,
            "eda_signal": eda_sig,
            "bvp_fs": bvp_fs,
            "bvp_signal": bvp_sig if bvp_valid else np.array([], dtype=np.float32),
            "bvp_valid": bvp_valid,
            "tags": tags,
            "labeled_intervals": labeled_intervals  # List of dicts: {'start_s', 'end_s', 'label': 0 or 1, 'phase'}
        }

    @staticmethod
    def reconstruct_stress_intervals(start_time: datetime, tags: List[datetime], is_v1: bool) -> List[Dict[str, Any]]:
        """
        Reconstructs REST (0) and STRESS (1) intervals according to Hongn et al. (2025).
        
        Protocol V1:
            - Baseline: 3 min (REST = 0)
            - Stroop test: 2 min (STRESS = 1)
            - First Rest: 5 min (REST = 0)
            - TMCT: 3 min (STRESS = 1)
            - Second Rest: 5 min (REST = 0)
            - Real Opinion: 30 s (STRESS = 1)
            - Opposite Opinion: 30 s (STRESS = 1)
            - Subtract test: 30 s (STRESS = 1)
            
        Protocol V2:
            - Baseline: 3 min (REST = 0)
            - First Rest: 5 min (REST = 0)
            - TMCT: 3 min (STRESS = 1)
            - Second Rest: 5 min (REST = 0)
            - Real Opinion: 30 s (STRESS = 1)
            - Opposite Opinion: 30 s (STRESS = 1)
            - Subtract test: 30 s (STRESS = 1)
        """
        intervals = []
        
        # If event tags match documented structure, use tag timestamps directly
        if is_v1 and len(tags) >= 12:
            # Tag offsets relative to start_time
            def get_offset(t_idx):
                return max(0.0, (tags[t_idx] - start_time).total_seconds())

            # Baseline Rest
            intervals.append({"start_s": get_offset(0), "end_s": get_offset(1), "label": 0, "phase": "Baseline"})
            # Stroop (Stress)
            intervals.append({"start_s": get_offset(2), "end_s": get_offset(3), "label": 1, "phase": "Stroop"})
            # First Rest
            intervals.append({"start_s": get_offset(3), "end_s": get_offset(4), "label": 0, "phase": "FirstRest"})
            # TMCT (Stress)
            intervals.append({"start_s": get_offset(4), "end_s": get_offset(5), "label": 1, "phase": "TMCT"})
            # Second Rest
            intervals.append({"start_s": get_offset(5), "end_s": get_offset(6), "label": 0, "phase": "SecondRest"})
            # Real Opinion (Stress)
            intervals.append({"start_s": get_offset(6), "end_s": get_offset(7), "label": 1, "phase": "RealOpinion"})
            # Opposite Opinion (Stress)
            intervals.append({"start_s": get_offset(8), "end_s": get_offset(9), "label": 1, "phase": "OppositeOpinion"})
            # Subtract (Stress)
            intervals.append({"start_s": get_offset(10), "end_s": get_offset(11), "label": 1, "phase": "Subtract"})

        elif (not is_v1) and len(tags) >= 8:
            def get_offset(t_idx):
                return max(0.0, (tags[t_idx] - start_time).total_seconds())

            # Baseline Rest
            intervals.append({"start_s": get_offset(0), "end_s": get_offset(1), "label": 0, "phase": "Baseline"})
            # TMCT (Stress)
            intervals.append({"start_s": get_offset(1), "end_s": get_offset(2), "label": 1, "phase": "TMCT"})
            # First Rest
            intervals.append({"start_s": get_offset(2), "end_s": get_offset(3), "label": 0, "phase": "FirstRest"})
            # Real Opinion (Stress)
            intervals.append({"start_s": get_offset(3), "end_s": get_offset(4), "label": 1, "phase": "RealOpinion"})
            # Opposite Opinion (Stress)
            intervals.append({"start_s": get_offset(5), "end_s": get_offset(6), "label": 1, "phase": "OppositeOpinion"})
            # Second Rest
            intervals.append({"start_s": get_offset(6), "end_s": get_offset(7), "label": 0, "phase": "SecondRest"})
            # Subtract (Stress) - if extra tag present
            if len(tags) >= 9:
                intervals.append({"start_s": get_offset(7), "end_s": get_offset(8), "label": 1, "phase": "Subtract"})
            else:
                intervals.append({"start_s": get_offset(7), "end_s": get_offset(7) + 30.0, "label": 1, "phase": "Subtract"})

        else:
            # Fallback to nominal cumulative protocol duration if tags are incomplete or misaligned
            t_curr = 0.0
            if is_v1:
                # Baseline 3m (180s)
                intervals.append({"start_s": t_curr, "end_s": t_curr + 180.0, "label": 0, "phase": "Baseline"})
                t_curr += 180.0
                # Stroop 2m (120s)
                intervals.append({"start_s": t_curr, "end_s": t_curr + 120.0, "label": 1, "phase": "Stroop"})
                t_curr += 120.0
                # First Rest 5m (300s)
                intervals.append({"start_s": t_curr, "end_s": t_curr + 300.0, "label": 0, "phase": "FirstRest"})
                t_curr += 300.0
                # TMCT 3m (180s)
                intervals.append({"start_s": t_curr, "end_s": t_curr + 180.0, "label": 1, "phase": "TMCT"})
                t_curr += 180.0
                # Second Rest 5m (300s)
                intervals.append({"start_s": t_curr, "end_s": t_curr + 300.0, "label": 0, "phase": "SecondRest"})
                t_curr += 300.0
                # Real Opinion 30s
                intervals.append({"start_s": t_curr, "end_s": t_curr + 30.0, "label": 1, "phase": "RealOpinion"})
                t_curr += 30.0
                # Opposite Opinion 30s
                intervals.append({"start_s": t_curr, "end_s": t_curr + 30.0, "label": 1, "phase": "OppositeOpinion"})
                t_curr += 30.0
                # Subtract 30s
                intervals.append({"start_s": t_curr, "end_s": t_curr + 30.0, "label": 1, "phase": "Subtract"})
            else:
                # Baseline 3m (180s)
                intervals.append({"start_s": t_curr, "end_s": t_curr + 180.0, "label": 0, "phase": "Baseline"})
                t_curr += 180.0
                # TMCT 3m (180s)
                intervals.append({"start_s": t_curr, "end_s": t_curr + 180.0, "label": 1, "phase": "TMCT"})
                t_curr += 180.0
                # First Rest 5m (300s)
                intervals.append({"start_s": t_curr, "end_s": t_curr + 300.0, "label": 0, "phase": "FirstRest"})
                t_curr += 300.0
                # Real Opinion 30s
                intervals.append({"start_s": t_curr, "end_s": t_curr + 30.0, "label": 1, "phase": "RealOpinion"})
                t_curr += 30.0
                # Opposite Opinion 30s
                intervals.append({"start_s": t_curr, "end_s": t_curr + 30.0, "label": 1, "phase": "OppositeOpinion"})
                t_curr += 30.0
                # Second Rest 5m (300s)
                intervals.append({"start_s": t_curr, "end_s": t_curr + 300.0, "label": 0, "phase": "SecondRest"})
                t_curr += 300.0
                # Subtract 30s
                intervals.append({"start_s": t_curr, "end_s": t_curr + 30.0, "label": 1, "phase": "Subtract"})

        return intervals

    def load_subject_exercise_session(self, subject_id: str, session_type: str = "AEROBIC") -> Optional[Dict[str, Any]]:
        """
        Loads exercise recordings (AEROBIC or ANAEROBIC).
        Used strictly as out-of-distribution evaluation challenge for the false-alarm study.
        """
        folder_name = subject_id
        if session_type == "AEROBIC" and subject_id == "S11":
            folder_name = "S11_a"
        elif session_type == "ANAEROBIC" and subject_id == "S16":
            folder_name = "S16_a"

        folder_path = os.path.join(self.wearable_dir, session_type, folder_name)
        if not os.path.exists(folder_path):
            return None

        eda_dt, eda_fs, eda_sig = self.parse_empatica_csv(os.path.join(folder_path, "EDA.csv"))
        bvp_dt, bvp_fs, bvp_sig = self.parse_empatica_csv(os.path.join(folder_path, "BVP.csv"))

        if eda_dt is None or eda_fs <= 0 or len(eda_sig) == 0:
            return None

        bvp_valid = (bvp_fs > 0 and len(bvp_sig) > 0)

        return {
            "subject_id": subject_id,
            "session_type": session_type,
            "start_time": eda_dt,
            "eda_fs": eda_fs,
            "eda_signal": eda_sig,
            "bvp_fs": bvp_fs,
            "bvp_signal": bvp_sig if bvp_valid else np.array([], dtype=np.float32),
            "bvp_valid": bvp_valid
        }
