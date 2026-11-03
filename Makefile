# ==============================================================================
# Wearable Physiological Stress Detection — Pipeline Execution Automation
# ==============================================================================

PYTHON ?= ../.venv/bin/python3
STREAMLIT ?= ../.venv/bin/streamlit

.PHONY: help install test audit eval train figures demo demo-cli clean

help:
	@echo "Available commands for wearable-stress research pipeline:"
	@echo "  make install    - Install required dependencies via pip"
	@echo "  make test       - Run full automated test suite (16 tests)"
	@echo "  make audit      - Audit raw PhysioNet DS1 cohort & generate CSV reports"
	@echo "  make eval       - Run ACT comparative evaluation against deterministic rules"
	@echo "  make figures    - Regenerate publication heatmaps, interaction plots, and boxplots"
	@echo "  make train      - Execute full 3x3 Factorial Matrix under LOSO (34 folds x 3 seeds)"
	@echo "  make demo       - Launch real-time Streamlit Sense-Think-Act dashboard"
	@echo "  make demo-cli   - Run headless smoke-test of the dashboard adapter"
	@echo "  make clean      - Remove bytecode, cache directories, and test artifacts"

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=. $(PYTHON) -m unittest discover tests

audit:
	PYTHONPATH=. $(PYTHON) src/data_audit.py

eval:
	PYTHONPATH=. $(PYTHON) src/act/run_act_eval.py

figures:
	PYTHONPATH=. $(PYTHON) src/visualization/generate_figures.py

train:
	PYTHONPATH=. $(PYTHON) src/experiments/run_experiments.py

demo:
	PYTHONPATH=. $(STREAMLIT) run src/act/dashboard.py

demo-cli:
	PYTHONPATH=. $(PYTHON) src/act/dashboard.py --mock --interval 0.5

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
