# Code and Architecture Guide

**Project:** Wearable Stress Detection & Physiological ML Benchmark  
**Scope:** Standalone Research Codebase (`code/`)

This document provides a technical walkthrough of every module and data pipeline in the codebase.

---

## Data Flow at a Glance

```
dataset/raw/DS1_Hongn_Stress_Exercise/
        │
        │  (1) AUDIT
        ▼
code/src/data_audit.py  ──────────────────►  results/data_audit/*.csv
        │
        │  (2) LOAD + FILTER + SEGMENT
        ▼
code/src/preprocessing/
  signal_loader.py  →  raw arrays in memory
  filters.py        →  clean EDA and BVP waveforms
  segmentation.py   →  30-second windows, REST/STRESS labels
        │
        ├──────────────────────────────────────────────────────────┐
        │  (3a) FEATURES for Random Forest                         │  (3b) TENSORS for CNN
        ▼                                                          ▼
code/src/features/                                     code/src/models/tensor_builder.py
  feature_extractor.py  →  20 hand-crafted numbers per window    →  (1 or 2, 480) numpy arrays
        │                                                          │
        └────────────────────────┬─────────────────────────────────┘
                                 │  (4) TRAIN + EVALUATE
                                 ▼
                  code/src/experiments/run_experiments.py
                    ├── code/src/models/baseline.py
                    ├── code/src/models/cnn1d.py
                    ├── code/src/evaluation/splits.py
                    └── code/src/evaluation/metrics.py
                                 │
                                 ▼
                  results/metrics/*.csv
                  results/checkpoints/*.pkl / *.pt
                                 │
        ┌────────────────────────┘
        │  (5) SENSE-THINK-ACT PROTOTYPE
        ▼
code/src/act/
  sensor_adapter.py  →  Sense layer (replay / EmotiBit / live UDP)
  dashboard.py       →  Think layer (load checkpoint, run inference)
  act_evaluator.py   →  Act layer (rules vs LLM recommendations)
  run_act_eval.py    →  CLI to run the evaluation and save scores
                                 │
                                 ▼
                  results/metrics/act_rubric_summary.csv
                  llm_logs/act_evaluation_log.json
```

---

## Top-Level Project Folders

| Folder | What it contains |
|---|---|
| `code/` | All Python source code |
| `dataset/` | Raw data (read-only) and split manifest |
| `results/` | All outputs: metrics, checkpoints, figures, audit CSVs |
| `llm_logs/` | Raw prompts, responses, and scores from the ACT evaluation |
| `personal_data/` | Placeholder for future EmotiBit personal recordings |
| `shared_latex/` | LaTeX template files shared by all reports |
| `00_proposal/` | Project proposal (PDF + LaTeX source) |
| `01_literature_review/` | Notes and matrix for the five papers reviewed |
| `02_interim_report/` | Interim report (PDF + LaTeX source) |
| `03_final_report/` | Final report (PDF + LaTeX source) |
| `04_presentation/` | Beamer slides, speaker notes, demo guide |
| `notes/` | Decision logs, implementation status, AI disclosure |
| `papers/` | PDF copies of the five reference papers |
| `tests/` | Unit tests for preprocessing logic |
| `archive/` | Old or superseded files kept for reference — do not use |

---

## `code/src/` — Python Source Files

### `code/src/data_audit.py`

**What it does:** Reads raw files from all 36 subjects and checks their quality.

Specifically it checks: file existence, sampling rates, signal lengths, duplicate
rows, and missing labels. It then writes five summary CSVs to `results/data_audit/`.

Run it once. You do not need to run it again unless the raw data changes.

---

### `code/src/preprocessing/` — Signal Loading and Cleaning

#### `signal_loader.py`

**What it does:** Opens the raw CSV files for one subject and one session
(`EDA.csv`, `BVP.csv`, `ACC.csv`, `tags.csv`) and returns a Python dict
with numpy arrays and metadata.

It handles the differences between the V1 protocol (males, S01–S18, includes
Stroop task) and the V2 protocol (females, f01–f18, Stroop excluded).

#### `filters.py`

**What it does:** Cleans the raw waveforms with digital filters.

- EDA (electrodermal activity): 4th-order Butterworth low-pass at 1.0 Hz.
  Also decomposes EDA into tonic (slow baseline) and phasic (fast response)
  components using exponential smoothing.
- BVP/PPG (blood volume pulse): 4th-order Butterworth bandpass, 0.5–8.0 Hz.

#### `segmentation.py`

**What it does:** Cuts the continuous filtered signal into windows.

- Window size: 30 seconds.
- Stride: 15 seconds (50% overlap).
- Labels each window as REST (0) or STRESS (1) using the timestamps in
  `tags.csv`.
- Also cuts exercise sessions into unlabelled windows (used only for the
  false-alarm-rate test, not for training).

---

### `code/src/features/` — Feature Extraction (for Random Forest)

#### `feature_extractor.py`

**What it does:** Turns one 30-second window into a row of numbers that the
Random Forest can use.

- **EDA features (10):** mean, std, min, max, linear slope, tonic mean,
  tonic std, SCR peak count, mean peak amplitude, sum of peak amplitudes.
- **PPG features (10):** mean, std, min, max, estimated heart rate, RMSSD
  (heart rate variability), and three spectral power bands.
- **Fusion (20):** EDA features + PPG features concatenated.

#### `extract_features.py`

**What it does:** Thin script wrapper. Calls `feature_extractor.py` and saves
the result to disk. You can run this standalone if you only want feature CSVs
without running the full experiment pipeline.

---

### `code/src/models/` — Model Definitions

#### `baseline.py`

**What it does:** A dummy classifier that always predicts the majority class
(REST). It is the performance floor. Every real model must beat it.

Uses `sklearn.dummy.DummyClassifier(strategy='most_frequent')`.

#### `tensor_builder.py`

**What it does:** Converts a 30-second filtered waveform into a PyTorch-ready
numpy tensor for the 1D-CNN.

- Resamples the signal to 16 Hz → 480 time steps.
- Single-sensor input shape: `(1, 480)`.
- EDA+PPG fusion input shape: `(2, 480)`.

#### `cnn1d.py`

**What it does:** Defines the 1D convolutional neural network architecture and
its training loop.

- **`Small1DCNN`:** Three convolutional layers, batch normalisation, max pooling,
  dropout, and a linear classification head. Input is a `(channels, 480)` tensor.
- **`CNNTrainer`:** Wraps the model with `fit()` and `predict()` methods.
  Uses early stopping on inner-validation loss.
- **`set_seed()`:** Sets the random seed for PyTorch, numpy, and Python so
  results are reproducible.

---

### `code/src/evaluation/` — Metrics and Data Splits

#### `splits.py`

**What it does:** Builds the 34-fold Leave-One-Subject-Out (LOSO) manifest.

For each fold, one subject is the test subject, and the remaining 33 subjects
are split into train (80%) and inner-validation (20%) by subject group. The
manifest is saved to `dataset/splits/loso_splits.json`.

Key guarantee: no subject ever appears in more than one partition of a fold.

#### `metrics.py`

**What it does:** Computes all the performance numbers used in the report.

- Macro-F1, balanced accuracy, per-class precision and recall.
- Confusion matrix (TP, TN, FP, FN).
- `compute_clustered_bootstrap_ci()`: 1,000 bootstrap resamples at the
  subject level to produce 95% confidence intervals.
- `compute_exercise_far()`: False-alarm rate on the exercise windows
  (windows that contain no stress but the model predicts stress).

---

### `code/src/experiments/run_experiments.py`

**What it does:** The main script that runs all 27 experiment configurations.

The factorial design is: 3 models (Baseline, RF, CNN) × 3 sensor sets
(EDA, PPG, EDA+PPG) × 3 random seeds.

For each configuration it:

1. Loads windows and extracts features or tensors.
2. Runs 34-fold LOSO using the manifest from `splits.py`.
3. In each fold: fits a `StandardScaler` **only on training subjects** (never
   on validation or test), trains the model, and evaluates on the test subject.
4. Collects predictions on exercise windows.
5. Aggregates results across seeds and computes bootstrap CIs.
6. Saves everything to `results/metrics/`.
7. Exports the best-seed model for each configuration to `results/checkpoints/`.

---

### `code/src/act/` — Sense-Think-Act Prototype

#### `sensor_adapter.py` — Sense Layer

**What it does:** Provides a stream of sensor readings to the rest of the ACT
pipeline. Three adapters, all inheriting from `BaseSensorAdapter`:

- **`ReplaySensorAdapter`:** Replays PhysioNet CSV files through a 30-second
  sliding buffer. This is what is used in the demo (offline, no hardware needed).
- **`EmotiBitFileAdapter`:** Parses CSV exports from a personal EmotiBit device.
  Used when personal data is available.
- **`LiveEmotiBitAdapter`:** Receives real-time data from EmotiBit hardware via
  UDP/OSC. Requires the physical device and an active Wi-Fi connection.

#### `act_evaluator.py` — Act Layer

**What it does:** Generates wellness recommendations and scores them.

- **`DeterministicRuleAct`:** A set of handwritten if-else rules that look at
  motion level, EDA, PPG, and sensor quality flags, then produce a
  recommendation (e.g., "take a break", "check sensor placement").
- **`LLMActAdapter`:** Sends the same sensor context to an LLM (OpenAI API or
  an offline mock). Falls back to the mock automatically when there is no
  network.
- **`RubricEvaluator`:** Scores each recommendation on six dimensions (1–5
  scale): specificity, actionability, data coherence, safety, timeliness, and
  appropriateness.

#### `run_act_eval.py` — CLI Runner

**What it does:** Command-line entry point. Calls `run_comparative_act_evaluation()`
with three test scenarios (normal stress, degraded sensor, exercise confounding),
then writes:

- `results/metrics/act_rubric_summary.csv` — rubric scores per scenario.
- `llm_logs/act_evaluation_log.json` — raw prompts, responses, and scores.

Run it with: `python3 code/src/act/run_act_eval.py`

#### `dashboard.py` — Streamlit Web App

**What it does:** Interactive visual dashboard for the full ACT pipeline.

Three-column layout:
- Column 1 (Sense): live or replayed EDA/BVP waveforms.
- Column 2 (Think): inference result (REST or STRESS) with confidence score,
  loaded from a saved checkpoint.
- Column 3 (Act): side-by-side rule recommendation vs. LLM recommendation.

Run it with: `streamlit run code/src/act/dashboard.py`

Requires the virtual environment (`project/.venv`) and network access for
the LLM column (offline mock is used automatically when offline).

---

### `code/src/visualization/generate_figures.py`

**What it does:** Reads the metric CSVs and generates six PNG figures saved
to `results/figures/`:

| File | What it shows |
|---|---|
| `heatmap_macro_f1.png` | Macro-F1 for all model/sensor combinations |
| `heatmap_balanced_acc.png` | Balanced accuracy for all combinations |
| `heatmap_exercise_far.png` | False-alarm rate on exercise windows |
| `confusion_matrices.png` | REST vs STRESS confusion matrices per model |
| `model_sensor_interaction.png` | Interaction effect plot (RF vs CNN × sensor) |
| `per_subject_distribution.png` | Window count distribution across subjects |

---

## `dataset/` — Data

| Path | What it contains |
|---|---|
| `dataset/raw/DS1_Hongn_Stress_Exercise/` | Original PhysioNet data. **Do not modify.** |
| `dataset/splits/loso_splits.json` | 34 LOSO folds. Each fold lists `test`, `val`, and `train` subject IDs. |
| `dataset/processed/` | Intermediate arrays saved to disk to avoid re-computing (if present). |

---

## `results/` — All Outputs

### `results/data_audit/`

Five CSVs produced by `data_audit.py`:

| File | What it contains |
|---|---|
| `subjects.csv` | One row per subject: ID, version, sex, age, exclusion flag. |
| `sessions.csv` | One row per session: subject, date, tasks recorded. |
| `signal_quality.csv` | Sampling rate and missing-sample fraction per signal per subject. |
| `label_distribution.csv` | REST and STRESS window counts per subject. |
| `exclusions.csv` | Which subjects were excluded and why (S02 = duplicate rows, f07 = dock cover on PPG). |

### `results/metrics/`

| File | What it contains |
|---|---|
| `factorial_metrics.csv` | Macro-F1, balanced accuracy, and 95% CI for all 27 configurations. |
| `subject_level_metrics.csv` | Per-subject macro-F1 for every fold. |
| `exercise_far_metrics.csv` | False-alarm rate on 7,048 exercise windows per model/sensor. |
| `act_rubric_summary.csv` | Rubric scores (1–5) for rules vs. LLM on three scenarios. |
| `interaction_analysis.json` | ANOVA interaction statistics (model × sensor). |

### `results/checkpoints/`

Six saved model files — one per model/sensor combination for the best seed:

| File | Model | Sensor |
|---|---|---|
| `deploy_RF_EDA.pkl` | Random Forest | EDA only |
| `deploy_RF_PPG.pkl` | Random Forest | PPG only |
| `deploy_RF_EDA+PPG.pkl` | Random Forest | EDA + PPG |
| `deploy_CNN_EDA.pt` | 1D-CNN | EDA only |
| `deploy_CNN_PPG.pt` | 1D-CNN | PPG only |
| `deploy_CNN_EDA+PPG.pt` | 1D-CNN | EDA + PPG |

### `results/figures/`

Six PNG files generated by `generate_figures.py`. Used in the report and slides.

---

## `llm_logs/`

| File | What it contains |
|---|---|
| `act_evaluation_log.json` | Full record of every LLM call: scenario name, prompt sent, response received, rubric scores, token count, latency, and cost. |

---

## `personal_data/`

Placeholder only. No personal EmotiBit recordings exist yet.

- `personal_data/raw/README.md` — explains the expected file structure.
- `personal_data/session_template.csv` — template to fill in when recording.

**Privacy rule:** Nothing in this folder may be sent to any external API or
committed to a public repository.

---

## `shared_latex/` — LaTeX Template Files

| File | What it contains |
|---|---|
| `usenix2019_v3.sty` | USENIX two-column 10pt style file. All reports use this. |
| `preamble_shared.tex` | Common package imports and macro definitions shared by all reports. |
| `bibliography/references.bib` | BibTeX bibliography with all six references used in the report. |

---

## `notes/` — Decisions and Status

| File/Folder | What it contains |
|---|---|
| `PROJECT_CONTEXT.md` | High-level description of the research question and design choices. |
| `implementation_status.md` | Current status of every implementation task (October 2026 update). |
| `REQUIREMENTS_CHECKLIST.md` | Course-graded requirements. Check boxes manually. |
| `SESSION_STATE.md` | What was done in the most recent work session. |
| `AI_DISCLOSURE.md` | Log of every AI-assisted action taken on this project. |
| `decisions/` | One file per major design decision (model choice, dataset choice, etc.). |
| `literature/` | Reading notes for each of the five reference papers. |

---

## `tests/`

| File | What it contains |
|---|---|
| `test_preprocessing.py` | Unit tests for `signal_loader.py`, `filters.py`, and `segmentation.py`. Checks shapes, label counts, and that the scaler is never fitted on test subjects. |

---

## `00_proposal/` through `04_presentation/` — Report Folders

| Folder | Contents |
|---|---|
| `00_proposal/` | Project proposal: `project_proposal.tex` + compiled PDF. |
| `01_literature_review/` | Literature review notes and matrix (Markdown only, no LaTeX). |
| `02_interim_report/` | Interim report: `interim_report.tex` + compiled `interim_report.pdf` (5 pages). |
| `03_final_report/` | Final report: `final_report.tex` + compiled PDF (not yet complete). |
| `04_presentation/` | Beamer slides (`presentation.tex` + PDF), `speaker_notes.md`, `demonstration_guide.md`. |

---

## `archive/`

Old or superseded files. Never use anything from here in the project.
Files are kept only to avoid losing work. Do not delete them.

---

## `papers/`

PDF copies of the seven reference papers. Read-only.

---

*This guide reflects the standalone research codebase structure.*
