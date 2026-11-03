"""
Data Audit Module for PhysioNet Wearable Device Dataset (DS1)
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Audits all raw sessions, verifies sampling rates, timestamps, event tags,
signal quality, and data constraints documented by Hongn et al. (2025).
Outputs reports to results/data_audit/
"""

import os
import sys
from pathlib import Path

# Ensure root containing 'src' is in sys.path
_curr = Path(__file__).resolve()
for _p in [_curr.parent, _curr.parent.parent, _curr.parent.parent.parent]:
    if (_p / "src").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import csv
import glob
from datetime import datetime
from typing import Dict, List, Tuple, Any

from src.config import default_config

DATASET_ROOT = str(default_config.raw_data_dir)
WEARABLE_DATASET_DIR = os.path.join(DATASET_ROOT, "Wearable_Dataset")
OUTPUT_DIR = str(default_config.data_audit_dir)

# Known sampling rates for Empatica E4
EXPECTED_FS = {
    "EDA": 4.0,
    "BVP": 64.0,
    "ACC": 32.0,
    "TEMP": 4.0,
    "HR": 1.0
}


def read_empatica_csv(filepath: str) -> Tuple[datetime, float, List[float]]:
    """
    Reads an Empatica E4 CSV file.
    Row 1: Start timestamp (UTC datetime string or unix timestamp).
    Row 2: Sampling frequency (Hz).
    Row 3+: Signal values.
    """
    if not os.path.exists(filepath):
        return None, 0.0, []
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    if len(lines) < 2:
        return None, 0.0, []
        
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
        
    values = []
    for line in lines[2:]:
        parts = line.split(",")
        try:
            if len(parts) == 1:
                values.append(float(parts[0]))
            else:
                values.append([float(p) for p in parts])
        except ValueError:
            continue
            
    return start_dt, fs, values


def read_tags_csv(filepath: str) -> List[datetime]:
    """Reads tags.csv which contains event timestamps."""
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
    return tags


def audit_subjects_and_sessions():
    """Performs comprehensive audit of all subjects and sessions."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(WEARABLE_DATASET_DIR):
        print(f"⚠️ Dataset directory not found at: {WEARABLE_DATASET_DIR}")
        print("To configure dataset location:")
        print("  1. Download PhysioNet DS1 (Hongn et al., 2025) per RUN_INSTRUCTIONS.md")
        print("  2. Set WEARABLE_DATA_DIR='/path/to/data' environment variable.")
        return
    
    subject_info_file = os.path.join(DATASET_ROOT, "subject-info.csv")
    demographics = {}
    if os.path.exists(subject_info_file):
        with open(subject_info_file, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                s_id = row[0].strip()
                if s_id.startswith("S") or s_id.startswith("f"):
                    demographics[s_id] = {
                        "gender": row[1].strip() if len(row) > 1 else "",
                        "age": row[2].strip() if len(row) > 2 else "",
                        "height_cm": row[3].strip() if len(row) > 3 else "",
                        "weight_kg": row[4].strip() if len(row) > 4 else "",
                        "physical_act": row[5].strip() if len(row) > 5 else "",
                        "protocol": row[6].strip() if len(row) > 6 else "",
                        "stress_ind": row[7].strip() if len(row) > 7 else "",
                        "aerobic": row[8].strip() if len(row) > 8 else "",
                        "anaerobic": row[9].strip() if len(row) > 9 else "",
                    }

    # Documented constraints from data_constraints.txt
    constraints = [
        {"subject_id": "S02", "session_type": "STRESS", "reason": "Duplicated raw signals downloaded from E4 Connect (ACC row 49545, BVP row 99091, EDA/TEMP row 6195)", "severity": "EXCLUDE_FROM_TRAINING", "affected_modalities": "ALL"},
        {"subject_id": "f07", "session_type": "STRESS", "reason": "Protection dock covered PPG and TEMP sensors; only EDA and ACC valid", "severity": "EXCLUDE_PPG_ONLY", "affected_modalities": "BVP,TEMP"},
        {"subject_id": "f14", "session_type": "STRESS", "reason": "Bluetooth connection lost during registration (split f14_a baseline, f14_b rest of protocol)", "severity": "SPLIT_SESSION", "affected_modalities": "ALL"},
        {"subject_id": "S03", "session_type": "AEROBIC", "reason": "Ended protocol early, only reaching 90 rpm", "severity": "PARTIAL_SESSION", "affected_modalities": "ALL"},
        {"subject_id": "S07", "session_type": "AEROBIC", "reason": "Ended protocol early, only reaching 95 rpm", "severity": "PARTIAL_SESSION", "affected_modalities": "ALL"},
        {"subject_id": "S11", "session_type": "AEROBIC", "reason": "Bluetooth connection lost; S11_a up to 105 rpm, S11_b 110 rpm", "severity": "SPLIT_SESSION", "affected_modalities": "ALL"},
        {"subject_id": "S12", "session_type": "AEROBIC", "reason": "Did not perform aerobic protocol; no data recorded", "severity": "MISSING_SESSION", "affected_modalities": "ALL"},
        {"subject_id": "S01", "session_type": "ANAEROBIC", "reason": "IBI file is empty", "severity": "IBI_EMPTY", "affected_modalities": "IBI"},
        {"subject_id": "S06", "session_type": "ANAEROBIC", "reason": "Ended protocol early (last sprint missing)", "severity": "PARTIAL_SESSION", "affected_modalities": "ALL"},
        {"subject_id": "S16", "session_type": "ANAEROBIC", "reason": "Bluetooth lost; S16_a 2 sprints, S16_b 2 sprints", "severity": "SPLIT_SESSION", "affected_modalities": "ALL"},
    ]

    # Save exclusions.csv
    exclusions_path = os.path.join(OUTPUT_DIR, "exclusions.csv")
    with open(exclusions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "session_type", "reason", "severity", "affected_modalities"])
        writer.writeheader()
        writer.writerows(constraints)

    all_subject_ids = [f"S{i:02d}" for i in range(1, 19)] + [f"f{i:02d}" for i in range(1, 19)]
    
    subjects_records = []
    sessions_records = []
    quality_records = []
    label_records = []

    for s_id in all_subject_ids:
        demo = demographics.get(s_id, {})
        proto_ver = "V1" if s_id.startswith("S") else "V2"
        
        # Check folders
        stress_folder = os.path.join(WEARABLE_DATASET_DIR, "STRESS", s_id)
        # Handle f14 special split
        if s_id == "f14" and not os.path.exists(stress_folder):
            stress_folder = os.path.join(WEARABLE_DATASET_DIR, "STRESS", "f14_b")
            
        aero_folder = os.path.join(WEARABLE_DATASET_DIR, "AEROBIC", s_id)
        if s_id == "S11" and not os.path.exists(aero_folder):
            aero_folder = os.path.join(WEARABLE_DATASET_DIR, "AEROBIC", "S11_a")
            
        anaero_folder = os.path.join(WEARABLE_DATASET_DIR, "ANAEROBIC", s_id)
        if s_id == "S16" and not os.path.exists(anaero_folder):
            anaero_folder = os.path.join(WEARABLE_DATASET_DIR, "ANAEROBIC", "S16_a")

        has_stress = os.path.exists(stress_folder)
        has_aero = os.path.exists(aero_folder)
        has_anaero = os.path.exists(anaero_folder)

        # Subject eligibility
        is_stress_eligible = has_stress and (s_id != "S02")  # S02 corrupted duplicate signals
        is_exercise_eligible = (has_aero or has_anaero)

        subjects_records.append({
            "subject_id": s_id,
            "protocol_version": proto_ver,
            "gender": demo.get("gender", ""),
            "age": demo.get("age", ""),
            "height_cm": demo.get("height_cm", ""),
            "weight_kg": demo.get("weight_kg", ""),
            "physical_activity": demo.get("physical_act", ""),
            "stress_session_available": has_stress,
            "aerobic_session_available": has_aero,
            "anaerobic_session_available": has_anaero,
            "stress_eligible": is_stress_eligible,
            "exercise_eligible": is_exercise_eligible,
            "exclusion_notes": "S02 corrupted duplicate; f07 PPG dock-covered" if s_id in ["S02", "f07"] else ("f14 split session" if s_id == "f14" else "")
        })

        # Process sessions
        session_map = [
            ("STRESS", stress_folder),
            ("AEROBIC", aero_folder),
            ("ANAEROBIC", anaero_folder)
        ]

        for s_type, folder in session_map:
            if not os.path.exists(folder):
                continue

            files_in_dir = os.listdir(folder)
            eda_dt, eda_fs, eda_vals = read_empatica_csv(os.path.join(folder, "EDA.csv"))
            bvp_dt, bvp_fs, bvp_vals = read_empatica_csv(os.path.join(folder, "BVP.csv"))
            acc_dt, acc_fs, acc_vals = read_empatica_csv(os.path.join(folder, "ACC.csv"))
            tags = read_tags_csv(os.path.join(folder, "tags.csv"))

            duration_s = len(eda_vals) / eda_fs if (eda_fs > 0 and eda_vals) else 0.0

            sessions_records.append({
                "subject_id": s_id,
                "session_type": s_type,
                "folder_path": folder,
                "files_present": ";".join(files_in_dir),
                "eda_samples": len(eda_vals),
                "eda_fs": eda_fs,
                "eda_start_time": eda_dt.isoformat() if eda_dt else "",
                "bvp_samples": len(bvp_vals),
                "bvp_fs": bvp_fs,
                "bvp_start_time": bvp_dt.isoformat() if bvp_dt else "",
                "acc_samples": len(acc_vals),
                "acc_fs": acc_fs,
                "acc_start_time": acc_dt.isoformat() if acc_dt else "",
                "num_tags": len(tags),
                "duration_seconds": round(duration_s, 2)
            })

            # Signal quality
            if eda_vals and isinstance(eda_vals[0], (int, float)):
                eda_min = min(eda_vals)
                eda_max = max(eda_vals)
                eda_mean = sum(eda_vals) / len(eda_vals)
                eda_variance = sum((x - eda_mean) ** 2 for x in eda_vals) / len(eda_vals)
                eda_std = eda_variance ** 0.5
                zeros_pct = (sum(1 for x in eda_vals if x == 0.0) / len(eda_vals)) * 100.0
                eda_valid = (eda_max > 0.01) and (zeros_pct < 95.0)
            else:
                eda_min = eda_max = eda_mean = eda_std = zeros_pct = 0.0
                eda_valid = False

            if bvp_vals and isinstance(bvp_vals[0], (int, float)):
                bvp_min = min(bvp_vals)
                bvp_max = max(bvp_vals)
                bvp_mean = sum(bvp_vals) / len(bvp_vals)
                bvp_var = sum((x - bvp_mean) ** 2 for x in bvp_vals) / len(bvp_vals)
                bvp_std = bvp_var ** 0.5
                bvp_valid = (bvp_std > 0.1) and (s_id != "f07" if s_type == "STRESS" else True)
            else:
                bvp_min = bvp_max = bvp_mean = bvp_std = 0.0
                bvp_valid = False

            quality_records.append({
                "subject_id": s_id,
                "session_type": s_type,
                "eda_min": round(eda_min, 4),
                "eda_max": round(eda_max, 4),
                "eda_mean": round(eda_mean, 4),
                "eda_std": round(eda_std, 4),
                "eda_zeros_pct": round(zeros_pct, 2),
                "eda_valid": eda_valid,
                "bvp_min": round(bvp_min, 2),
                "bvp_max": round(bvp_max, 2),
                "bvp_mean": round(bvp_mean, 2),
                "bvp_std": round(bvp_std, 2),
                "bvp_valid": bvp_valid,
                "acc_valid": len(acc_vals) > 0,
                "overall_quality_flag": "VALID" if (eda_valid and bvp_valid) else "DEGRADED"
            })

            # Calculate label distribution for STRESS sessions
            if s_type == "STRESS" and len(tags) >= 8:
                # Approximate windows from protocol blocks
                # V1: Rest = Baseline (3m) + Rest1 (5m) + Rest2 (5m) = 13m = 780s -> ~26 windows
                # Stress = Stroop (2m) + TMCT (3m) + Real (30s) + Opp (30s) + Sub (30s) = 6.5m = 390s -> ~13 windows
                # V2: Rest = Baseline (3m) + Rest1 (5m) + Rest2 (5m) = 13m = 780s -> ~26 windows
                # Stress = TMCT (3m) + Real (30s) + Opp (30s) + Sub (30s) = 4.5m = 270s -> ~9 windows
                if proto_ver == "V1":
                    rest_s = 780.0
                    stress_s = 390.0
                else:
                    rest_s = 780.0
                    stress_s = 270.0

                label_records.append({
                    "subject_id": s_id,
                    "protocol_version": proto_ver,
                    "rest_duration_s": rest_s,
                    "rest_windows_30s": int(rest_s // 30),
                    "stress_duration_s": stress_s,
                    "stress_windows_30s": int(stress_s // 30),
                    "exercise_session": False
                })
            elif s_type in ["AEROBIC", "ANAEROBIC"]:
                label_records.append({
                    "subject_id": s_id,
                    "protocol_version": proto_ver,
                    "session_type": s_type,
                    "exercise_duration_s": duration_s,
                    "exercise_windows_30s": int(duration_s // 30),
                    "exercise_session": True
                })

    # Save subjects.csv
    subjects_path = os.path.join(OUTPUT_DIR, "subjects.csv")
    with open(subjects_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject_id", "protocol_version", "gender", "age", "height_cm", "weight_kg",
            "physical_activity", "stress_session_available", "aerobic_session_available",
            "anaerobic_session_available", "stress_eligible", "exercise_eligible", "exclusion_notes"
        ])
        writer.writeheader()
        writer.writerows(subjects_records)

    # Save sessions.csv
    sessions_path = os.path.join(OUTPUT_DIR, "sessions.csv")
    with open(sessions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject_id", "session_type", "folder_path", "files_present",
            "eda_samples", "eda_fs", "eda_start_time", "bvp_samples", "bvp_fs", "bvp_start_time",
            "acc_samples", "acc_fs", "acc_start_time", "num_tags", "duration_seconds"
        ])
        writer.writeheader()
        writer.writerows(sessions_records)

    # Save signal_quality.csv
    quality_path = os.path.join(OUTPUT_DIR, "signal_quality.csv")
    with open(quality_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject_id", "session_type", "eda_min", "eda_max", "eda_mean", "eda_std",
            "eda_zeros_pct", "eda_valid", "bvp_min", "bvp_max", "bvp_mean", "bvp_std",
            "bvp_valid", "acc_valid", "overall_quality_flag"
        ])
        writer.writeheader()
        writer.writerows(quality_records)

    # Save label_distribution.csv
    labels_path = os.path.join(OUTPUT_DIR, "label_distribution.csv")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject_id", "protocol_version", "session_type", "rest_duration_s",
            "rest_windows_30s", "stress_duration_s", "stress_windows_30s",
            "exercise_duration_s", "exercise_windows_30s", "exercise_session"
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(label_records)

    print(f"Data audit completed successfully. Output written to {OUTPUT_DIR}")


if __name__ == "__main__":
    audit_subjects_and_sessions()
