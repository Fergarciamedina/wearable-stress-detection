"""
Runner for ACT Comparative Rubric Evaluation
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Executes the formal comparative evaluation between Deterministic Rules and LLM Recommender.
Generates results/metrics/act_rubric_summary.csv and llm_logs/act_evaluation_log.json.
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

import json
import pandas as pd

from src.act.act_evaluator import run_comparative_act_evaluation

from src.config import default_config
RESULTS_METRICS_DIR = str(default_config.metrics_dir)
os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)


def main():
    print("Running Comparative ACT Evaluation Suite...")
    res = run_comparative_act_evaluation()
    log_path = res["log_path"]
    
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    summary_rows = []
    for item in logs:
        sc_name = item["scenario"]
        # Rule scores
        r_rub = item["rule_rubric"]
        r_out = item["rule_output"]
        summary_rows.append({
            "scenario": sc_name,
            "approach": "Deterministic Rules",
            "latency_ms": r_out["latency_ms"],
            "tokens_used": r_out["tokens_used"],
            "cost_usd": r_out["cost_usd"],
            "data_coherence": r_rub["data_coherence"],
            "actionability": r_rub["actionability"],
            "safety": r_rub["safety"],
            "no_medical_claims": r_rub["no_medical_claims"],
            "uncertainty_handling": r_rub["uncertainty_handling"],
            "failure_handling": r_rub["failure_handling"]
        })
        # LLM scores
        l_rub = item["llm_rubric"]
        l_out = item["llm_output"]
        summary_rows.append({
            "scenario": sc_name,
            "approach": f"LLM ({l_out['model']})",
            "latency_ms": l_out["latency_ms"],
            "tokens_used": l_out["tokens_used"],
            "cost_usd": l_out["cost_usd"],
            "data_coherence": l_rub["data_coherence"],
            "actionability": l_rub["actionability"],
            "safety": l_rub["safety"],
            "no_medical_claims": l_rub["no_medical_claims"],
            "uncertainty_handling": l_rub["uncertainty_handling"],
            "failure_handling": l_rub["failure_handling"]
        })

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(RESULTS_METRICS_DIR, "act_rubric_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"ACT evaluation completed. Summary written to {csv_path}")


if __name__ == "__main__":
    main()
