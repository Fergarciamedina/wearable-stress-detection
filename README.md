# Wearable Stress Detection & Physiological ML Benchmark

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A standalone, reproducible research library evaluating subject-independent wearable cognitive stress detection using Electrodermal Activity (EDA) and Photoplethysmography (PPG/BVP).

This package benchmarks machine learning architectures (Random Forest vs. PyTorch 1D-CNN) across single-sensor and multimodal configurations under nested Leave-One-Subject-Out (LOSO) cross-validation on PhysioNet DS1 ($N=34$ paired cohort). It explicitly characterizes model vulnerability to **false stress alarms induced by physical exercise** and provides a closed-loop **Sense-Think-Act** prototype.

---

## 🔬 Empirical Findings & Benchmark Results

Under strict subject-independent LOSO cross-validation across 34 participants (PhysioNet DS1), tabular feature-engineered Random Forest consistently outperforms raw-tensor 1D-CNN:

| Model | Sensor Configuration | Macro-F1 (Mean ± 95% CI) | Balanced Accuracy | Exercise False Alarm Rate (FAR) |
|:---|:---|:---|:---|:---|
| **Majority Baseline** | Any | 0.4429 [0.437, 0.449] | 50.00% | 0.00% (No alarms) |
| **Random Forest** | EDA Only | 0.5971 [0.540, 0.654] | 64.50% | 58.22% |
| **Random Forest** | PPG Only | 0.5459 [0.490, 0.602] | 58.20% | 53.00% |
| **Random Forest** | **EDA + PPG (Fusion)** | **0.6528 [0.602, 0.702]** | **68.59%** | **59.62%** |
| **1D-CNN** | EDA Only | 0.4038 [0.361, 0.447] | 52.44% | 52.00% |
| **1D-CNN** | PPG Only | 0.5056 [0.428, 0.578] | 60.05% | 71.73% |
| **1D-CNN** | EDA + PPG (Fusion) | 0.4782 [0.428, 0.528] | 57.94% | 41.54% |

### Key Scientific Takeaways:
1. **The Representation Gap:** With small cohorts ($N=34$), domain-specific feature engineering (20 features: tonic/phasic EDA + cardiac variability) provides stronger inductive bias than end-to-end 1D convolutions on raw temporal tensors.
2. **The Exercise Confound:** Across 7,048 exercise windows, models trigger false stress alarms in **41% to 72%** of cases, showing that sympathetic arousal from physical movement mimics cognitive stress.
3. **Closed-Loop Action:** A deterministic rule engine provides sub-millisecond adaptive recommendations (e.g. seated breathing breaks for cognitive stress vs. hydration for exercise arousal).

---

## 📁 Repository Structure

```text
code/
├── README.md                      # This technical guide
├── requirements.txt               # Pinned dependencies
├── Makefile                       # Automation shortcuts (test, audit, figures, demo)
├── tests/                         # Automated unit & integration tests
│   ├── __init__.py
│   ├── test_pipeline.py           # 10 comprehensive pipeline tests
│   └── README.md
├── reference/                     # Reference notebooks & citations
│   ├── Wearable_Dataset.ipynb
│   └── CITATION.md
└── src/                           # Core library
    ├── __init__.py
    ├── config.py                  # Dynamic path resolution (ProjectConfig)
    ├── data_audit.py              # Cohort sanity & signal quality checks
    ├── preprocessing/             # Signal loading, Butterworth filtering, 30s windowing
    │   ├── filters.py
    │   ├── segmentation.py
    │   └── signal_loader.py
    ├── features/                  # 20 statistical & physiological features
    │   └── feature_extractor.py
    ├── models/                    # Model definitions & serialization
    │   ├── baseline.py            # Majority class baseline wrapper
    │   ├── random_forest.py       # StressRandomForest with StandardScaler
    │   ├── cnn1d.py               # PyTorch Small1DCNN & CNNTrainer
    │   └── tensor_builder.py      # Resampling to 16Hz & (C, 480) tensor builder
    ├── evaluation/                # Validation strategy & scoring
    │   ├── metrics.py             # Macro-F1, balanced acc, bootstrap CIs, exercise FAR
    │   └── splits.py              # LOSO partition manifest generation
    ├── act/                       # Closed-loop intervention
    │   ├── sensor_adapter.py      # Chronological streaming replay adapter
    │   ├── act_evaluator.py       # Deterministic rule engine & offline LLM adapter
    │   ├── dashboard.py           # Streamlit real-time demonstrator UI
    │   └── run_act_eval.py        # ACT rubric scoring runner
    ├── experiments/               # Factorial benchmark runner
    │   └── run_experiments.py     # 3 Models x 3 Sensors x 3 Seeds LOSO runner
    └── visualization/             # Publication figure generation
        └── generate_figures.py
```

---

## ⚡ Quickstart & Installation

For a comprehensive academic evaluation guide, see [`RUN_INSTRUCTIONS.md`](RUN_INSTRUCTIONS.md).

### Option A: Native Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Or install as an editable package:
# pip install -e .
```

### Option B: Docker Container
```bash
docker compose up --build
```
Open `http://localhost:8501` to access the Streamlit dashboard in an isolated, CPU-compatible container.

### Run Automated Verification Tests
```bash
python3 -m unittest discover tests
```
*Expected: 16 passing unit and regression tests in < 0.1s.*

### Launch the Sense-Think-Act Dashboard
```bash
streamlit run src/act/dashboard.py
```
*Note: If raw PhysioNet data is not downloaded, the dashboard automatically provides a zero-dependency **Synthetic Signal Demo** mode.*

Or execute an offline CLI smoke test from the terminal:
```bash
python3 src/act/dashboard.py --mock --interval 0.5
```

---

## 🚀 Execution Guide: Pipeline Stages

| Pipeline Stage | Native Command | Makefile Shortcut | Canonical Outputs Produced |
|:---|:---|:---|:---|
| **1. Unit Tests** | `python3 -m unittest discover tests` | `make test` | Console pass report (16 tests) |
| **2. Data Audit** | `python3 src/data_audit.py` | `make audit` | `data/results/data_audit/` (`subjects.csv`, `exclusions.csv`, `signal_quality.csv`) |
| **3. LOSO Splits** | `python3 src/evaluation/splits.py` | `make splits` | `data/public/splits/loso_splits.json` |
| **4. Factorial Experiments** | `python3 src/experiments/run_experiments.py` | `make train` | `data/results/metrics/` (`factorial_metrics.csv`), `data/results/checkpoints/` |
| **5. ACT Rubric Evaluation** | `python3 src/act/run_act_eval.py` | `make act-eval` | `data/results/metrics/act_rubric_summary.csv`, `data/llm_logs/act_evaluation_log.json` |
| **6. Generate Figures** | `python3 src/visualization/generate_figures.py` | `make figures` | `data/results/figures/` (heatmaps, boxplots, interaction curves) |
| **7. Streamlit Dashboard** | `streamlit run src/act/dashboard.py` | `make run-dashboard` | Browser UI at `http://localhost:8501` |

---

## 💾 Canonical Data & Model Artifact Locations

All experimental artifacts reside under `project/data/`:
- **Raw Data:** `data/public/raw/DS1_Hongn_Stress_Exercise/`
- **Validation Splits:** `data/public/splits/loso_splits.json`
- **Trained Checkpoints:**
  - `data/results/checkpoints/deploy_RF_EDA.pkl`
  - `data/results/checkpoints/deploy_RF_PPG.pkl`
  - `data/results/checkpoints/deploy_RF_EDA+PPG.pkl` (Best overall model, Macro-F1 0.6528)
  - `data/results/checkpoints/deploy_CNN_EDA.pt`
  - `data/results/checkpoints/deploy_CNN_PPG.pt`
  - `data/results/checkpoints/deploy_CNN_EDA+PPG.pt`
- **Benchmark CSVs:** `data/results/metrics/factorial_metrics.csv`, `exercise_far_metrics.csv`
- **Publication Figures:** `data/results/figures/` (`heatmap_macro_f1.png`, `heatmap_exercise_far.png`, etc.)

To run against a dataset located on an external drive or server, set the environment variable:
```bash
export WEARABLE_DATA_DIR="/path/to/my/data"
```

---

## 🔒 Scientific Scope, Validations & Disclosures

### 1. Implemented & Empirically Verified Functionality
- **Signal Preprocessing:** Butterworth lowpass (EDA 1.0 Hz) and bandpass (PPG 0.5–8.0 Hz) filtering, cvxEDA tonic/phasic decomposition, and 30-second windowing.
- **Statistical Feature Extraction:** 20 hand-crafted features (10 EDA: tonic mean/std/min/max/drift, phasic peak count/amp/sum; 10 PPG: mean/std, pulse rate, HRV SDNN/RMSSD/pNN50, spectral power).
- **Machine Learning Benchmark:** Majority Baseline, Random Forest, and PyTorch 1D-CNN across 3 sensor sets (EDA, PPG, Fusion) under nested Leave-One-Subject-Out (LOSO) cross-validation on PhysioNet DS1 ($N=34$ paired cohort).
- **Exercise Confounding Analysis:** Quantitative false-stress alarm rate evaluation across 7,048 exercise windows.
- **Sense–Think–Act Pipeline:** Deterministic sub-millisecond rule engine and offline LLM recommendation adapter (`offline-mock-v1`).
- **Interactive UI:** Chronological replay demonstrator in Streamlit.

### 2. Synthetic Demonstrations
- **Mode B (Synthetic Signal Demo):** Integrated into the Streamlit dashboard for zero-dependency evaluations. Demonstrates end-to-end signal buffering, feature scaling, model inference, and rule-based actuation without requiring the 2.5 GB PhysioNet raw dataset download.

### 3. Pending EmotiBit Hardware Validation
- **Hardware Integration:** The UDP broadcast receiver (port 31337) and local EmotiBit CSV ingestion classes are fully architected in code (`src/act/sensor_adapter.py`). Live on-body empirical recording and public-to-personal domain transfer evaluation remain scheduled for physical hardware commissioning.

### 4. Future Roadmap (Non-Implemented Extensions)
- **ACC-Aware Motion Cancellation:** Active accelerometer gating (3-axis ACC thresholding) to suppress false stress alarms during vigorous physical movement.
- **Vector-Similarity Cross-Subject Retrieval:** Embedding-based exemplar retrieval across participant baselines.

---

## 📄 License & Attribution

- **Codebase License:** MIT License.
- **PhysioNet DS1 Dataset:** Open Data Commons Attribution License (ODC-By v1.0).
- **Privacy Assurance:** No private student identities, raw personal recordings, or billable API keys are included in this repository.
