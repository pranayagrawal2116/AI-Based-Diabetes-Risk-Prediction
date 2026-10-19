#!/usr/bin/env bash
# ==============================================================================
# Master One-Command Runner for AI-Based Diabetes Risk Prediction Platform
# Usage:
#   ./run.sh                  (Runs audit, EDA, full model training, SHAP, reports, tests, and launches Streamlit)
#   ./run.sh --skip-train     (Skips retraining and immediately launches Streamlit)
#   ./run.sh --port 8502      (Runs on custom port)
# ==============================================================================

set -e

# Change to script directory
cd "$(dirname "$0")"

# Execute master python orchestrator
python3 run_all.py "$@"
