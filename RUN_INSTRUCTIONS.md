# Academic Delivery & Execution Guide
## COEN 6322: AI for Wearable Health Applications (Fall 2026)

**Research Project:** *Stress vs. Exercise in Wrist Wearables: Does the Model Ranking Depend on the Sensor Set?*  
**Artifact Package:** Standalone Reproducible Python Package (`project/code/`)

---

## 📋 Executive Overview

This package is a self-contained, reproducible research codebase evaluating machine learning models (Random Forest vs. PyTorch 1D-CNN) for wearable cognitive stress detection using Electrodermal Activity (EDA) and Photoplethysmography (PPG). It features:
- Rigorous Leave-One-Subject-Out (LOSO) cross-validation on PhysioNet DS1 ($N=34$ participants).
- Physical exercise false alarm evaluation across 7,048 temporal windows.
- Closed-loop Sense–Think–Act demonstration dashboard implemented in Streamlit.
- Dual execution support: **Native Python 3.11+ Virtual Environment** or **Docker Compose**.
- **Instant Synthetic Demo Mode**: The dashboard and unit tests can run immediately without requiring raw dataset downloads.

---

## 💻 Hardware & System Requirements

- **Operating System:** macOS (Apple Silicon or Intel), Linux (Ubuntu 20.04+), or Windows 10/11 (WSL2 recommended).
- **Python Version:** Python 3.11 or 3.12 (Python 3.10 is also compatible).
- **Hardware Resources:** Standard CPU (No GPU required; 1D-CNN is optimized for CPU inference; 4 GB RAM minimum).
- **Disk Space:**
  - Code & dependencies: ~250 MB.
  - Pre-trained checkpoints & benchmark results: ~15 MB.
  - Raw PhysioNet DS1 dataset (optional for full retraining): ~2.5 GB uncompressed.

---

## ⚡ Option 1: Native Python Execution (Recommended)

### Step 1: Create and Activate Virtual Environment
Open a terminal in the `code/` directory:
```bash
cd /path/to/project/code

# Create a clean virtual environment
python3 -m venv .venv

# Activate the virtual environment:
# On macOS / Linux:
source .venv/bin/activate

# On Windows (cmd.exe):
.venv\Scripts\activate.bat
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### Step 2: Install Package & Dependencies
Install dependencies using `pip` (or standard editable install via `pyproject.toml`):
```bash
# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Install pinned dependencies
pip install -r requirements.txt

# Alternatively, install as an editable package
pip install -e .
```

### Step 3: Run Automated Verification Tests
Execute the unit and regression test suite (26 tests verifying signal filtering, window segmentation, feature extraction, tensor shaping, model inference, metrics, and H1/H2 research contextualization):
```bash
python3 -m unittest discover tests
```
*Expected output: `Ran 26 tests in ~0.15s` with `OK`.*

### Step 4: Launch the Sense–Think–Act Dashboard
Launch the interactive Streamlit dashboard:
```bash
streamlit run src/act/dashboard.py
```
Open your browser at `http://localhost:8501`.
- **With raw dataset:** Replays authentic physiological sessions from PhysioNet DS1.
- **Without raw dataset:** The dashboard automatically provides **Mode B: Synthetic Signal Demo**, allowing instant interactive testing of SENSE, THINK (model inference), and ACT (rule vs. LLM recommendations).

---

## 🐳 Option 2: Docker Execution

If Docker and Docker Compose are installed on your machine, you can run the Streamlit dashboard in an isolated, CPU-compatible container without installing Python dependencies locally.

### Step 1: Build and Launch Container
From the `code/` directory:
```bash
docker compose up --build
```
This builds an OCI-compliant multi-arch image based on `python:3.11-slim`, installs all required dependencies, binds the dashboard to port `8501`, and configures read-only access to existing checkpoints and datasets.

### Step 2: Access the Dashboard
Navigate to `http://localhost:8501` in your web browser.

### Step 3: Stop Container
Press `Ctrl+C` in your terminal, or run:
```bash
docker compose down
```

---

## 📊 Dataset Configuration & Placement

The codebase is configured to look for the canonical PhysioNet DS1 dataset by default at:
`../data/public/raw/DS1_Hongn_Stress_Exercise/`

### If you want to run authentic retraining or raw data audit:
1. **Source:** PhysioNet *Wearable Device Dataset from Induced Stress and Structured Exercise Sessions* (Hongn et al., 2025).  
   URL: `https://physionet.org/content/wearable-device-dataset/1.0.0/`  
   License: Open Data Commons Attribution License (ODC-By v1.0).
2. **Directory Structure:**
   Ensure the directory contains the 34 subject folders (`S01/` to `S18/` and `f01/` to `f18/`).
3. **Custom Location via Environment Variable:**
   If you have downloaded the dataset to another directory, point `WEARABLE_DATA_DIR` to your data folder:
   ```bash
   export WEARABLE_DATA_DIR="/custom/path/to/project/data"
   ```
   The library will automatically resolve this path dynamically.

---

## 🔬 Complete Pipeline Verification Commands

All key research pipeline tasks can be executed directly using Python or via the provided `Makefile`:

| Task | Native Command | Makefile Shortcut | Expected Output |
|:---|:---|:---|:---|
| **Automated Tests** | `python3 -m unittest discover tests` | `make test` | 26 tests passed |
| **Data Audit** | `python3 src/data_audit.py` | `make audit` | Cohort summary & signal quality reports |
| **Generate LOSO Splits** | `python3 src/evaluation/splits.py` | `make splits` | Generates `loso_splits.json` |
| **Full LOSO Training** | `python3 src/experiments/run_experiments.py` | `make train` | Evaluates 3 models × 3 sensor sets across 34 subjects |
| **Research Experiments (H1/H2)** | `python3 -m src.research.run_research_experiments` | — | Evaluates E0–E3 contextualization under 34-fold LOSO |
| **Exercise FAR Analysis** | `python3 -m unittest tests.test_pipeline.TestWearablePipelineComprehensive.test_09_metrics_and_far_calculation` | `make eval` | Calculates exercise false alarm rate |
| **ACT Rubric Evaluation** | `python3 src/act/run_act_eval.py` | `make act-eval` | Benchmark comparison between Rules and LLM |
| **Figure Generation** | `python3 src/visualization/generate_figures.py` | `make figures` | Generates publication PNG figures |
| **CLI Streamlit Smoke Test**| `python3 src/act/dashboard.py --mock --interval 0.5` | `make demo` | Non-GUI streaming simulation check |

---

## 🔍 Pre-trained Model Checkpoints

Pre-computed checkpoints are included in `../data/results/checkpoints/` so you do not need to rerun hours of LOSO cross-validation:
- `deploy_RF_EDA.pkl` (Random Forest, EDA features)
- `deploy_RF_PPG.pkl` (Random Forest, PPG features)
- `deploy_RF_EDA+PPG.pkl` (Random Forest, Multimodal Fusion — Best Model)
- `deploy_CNN_EDA.pt` (PyTorch 1D-CNN, EDA tensor)
- `deploy_CNN_PPG.pt` (PyTorch 1D-CNN, PPG tensor)
- `deploy_CNN_EDA+PPG.pt` (PyTorch 1D-CNN, Multimodal tensor)

Research comparison benchmarks are pre-computed in `../data/results/research/comparison/`:
- `research_comparison.csv` (E0–E3 summary across 34 folds)
- `fold_predictions.csv` (Subject-level predictions across E0–E3)

---

## ⚠️ Research Scope, Validations & Disclosures

### 1. Implemented & Verified Functionality
- **Signal Preprocessing:** Butterworth lowpass/bandpass filtering, cvxEDA tonic/phasic decomposition, and 30-second windowing across EDA, PPG, and tri-axial ACC.
- **Statistical Features:** 20 statistical features extracted across EDA and PPG, plus 7 tri-axial ACC movement features.
- **LOSO Benchmark:** Evaluated on PhysioNet DS1 ($N=34$ participants) across Majority Baseline, Random Forest, and 1D-CNN.
- **Exercise FAR Evaluation:** Quantitative false stress rate computed on 7,048 exercise windows.
- **Context-Aware Research Extensions:** H1 ACC-aware movement gating and H2 physiological $k$-NN similarity retrieval evaluated under 34-fold LOSO.
- **Sense–Think–Act:** Deterministic rule engine and offline LLM adapter (`offline-mock-v1`).
- **Unit & Regression Tests:** 26 automated tests covering all components.

### 2. Synthetic Demonstrations
- **Mode B (Synthetic Demo):** Streamlit dashboard provides built-in synthetic signal generation for instant evaluation without downloading the 2.5 GB PhysioNet dataset.

### 3. Pending Hardware Validation
- **EmotiBit Integration:** UDP broadcast receiver (port 31337) and local CSV parsers are architected in code; on-body physical hardware validation and domain transfer remain scheduled for Phase 3.

### 4. Future Roadmap (Non-Implemented Extensions)
- **Personalized Online Adaptation:** Continual subject-specific fine-tuning on consumer-grade hardware.
- **WESAD Cross-Corpus Evaluation:** Benchmarking transfer learning across distinct wearable modalities.

---

## ❓ Frequently Asked Questions (FAQ)

**Q: Can I run the project without downloading the 2.5 GB PhysioNet dataset?**  
**A:** Yes! The automated unit tests (`make test`) and the Streamlit dashboard (`make run-dashboard`) run fully in self-contained mode. The dashboard includes **Mode B: Synthetic Signal Demo** which requires zero external files.

**Q: Why does Random Forest outperform 1D-CNN on this dataset?**  
**A:** As detailed in our paper, with $N=34$ participants and subject-independent splits, engineered physiological features (phasic EDA amplitude, pulse amplitude, heart rate variability) provide stronger inductive bias against overfitting than 1D convolutions trained on limited raw time series.

**Q: How do I cleanly uninstall?**  
**A:** Simply deactivate and remove the `.venv` directory:
```bash
deactivate
rm -rf .venv
```

