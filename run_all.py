"""
Master Execution Pipeline Script (run_all.py).
Executes the entire end-to-end lifecycle in a single terminal command:
1. Multi-Cohort Data Cryptographic & Integrity Audit
2. Exploratory Data Analysis & Statistical Hypothesis Testing
3. Model Training, Optimization, Probability Calibration & Persistence
4. Explainable AI (SHAP) Visualizations
5. Research Report Compilation (PDF, Markdown & CSV)
6. Automated Pytest Verification Suite
7. Live Streamlit Web Application Launch
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Terminal Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner(step_num: int, total_steps: int, title: str):
    print(f"\n{CYAN}{'='*80}{RESET}")
    print(f"{BOLD}{GREEN}[Step {step_num}/{total_steps}] {title}{RESET}")
    print(f"{CYAN}{'='*80}{RESET}\n")


def run_command(cmd: list, desc: str):
    t0 = time.time()
    print(f"{YELLOW}▶ Executing: {' '.join(cmd)}{RESET}")
    res = subprocess.run(cmd)
    elapsed = round(time.time() - t0, 2)
    if res.returncode != 0:
        print(f"\n{RED}✖ ERROR: {desc} failed with exit code {res.returncode}.{RESET}")
        sys.exit(res.returncode)
    print(f"{GREEN}✔ SUCCESS: {desc} completed in {elapsed}s.{RESET}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Master Orchestrator for AI-Based Diabetes Risk Prediction Platform"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the model training phase if models are already trained.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running pytest automated test suite.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port number for Streamlit application (default: 8501).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Launch Streamlit in headless mode without auto-opening browser.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    print(f"""
{CYAN}{BOLD}================================================================================
  AI-BASED DIABETES RISK PREDICTION - MASTER EXECUTION PIPELINE
================================================================================{RESET}
  Project Root: {project_root}
  Python Interpreter: {sys.executable}
  Streamlit Port: {args.port}
  Train / Retrain: {'SKIPPED' if args.skip_train else 'ACTIVE (Full Multi-Cohort Retraining)'}
{CYAN}================================================================================{RESET}
""")

    total_steps = 7 if not args.skip_train else 5
    current_step = 1

    # Step 1: Data Integrity & Cryptographic Audit
    print_banner(current_step, total_steps, "Data Integrity & Cryptographic Audit")
    run_command(
        [sys.executable, "-m", "src.data_audit"],
        "Data Audit & Cryptographic MD5 Verification",
    )
    current_step += 1

    # Step 2: Exploratory Data Analysis & Statistical Tests
    print_banner(current_step, total_steps, "Exploratory Data Analysis & Statistical Testing")
    run_command(
        [sys.executable, "-m", "src.eda"],
        "Exploratory Data Analysis & Hypothesis Tests",
    )
    current_step += 1

    # Step 3: Model Training & Retraining
    if not args.skip_train:
        print_banner(current_step, total_steps, "Model Training, Cross-Validation & Calibration")
        run_command(
            [sys.executable, "-m", "src.train_models"],
            "Model Training, Optimization, Probability Calibration & Persistence",
        )
        current_step += 1

        # Step 4: Explainable AI (SHAP)
        print_banner(current_step, total_steps, "Explainable AI (SHAP) Attribution Generation")
        run_command(
            [sys.executable, "-m", "src.explainability"],
            "Global SHAP Visualizations & Attribution",
        )
        current_step += 1

    # Step 5: Research Report Compilation
    print_banner(current_step, total_steps, "Compiling Research Reports (PDF & Markdown)")
    run_command(
        [sys.executable, "-m", "src.report_generator"],
        "Formal Research Report Compilation",
    )
    current_step += 1

    # Step 6: Automated Pytest Suite
    if not args.skip_tests:
        print_banner(current_step, total_steps, "Running Automated Pytest Test Suite")
        run_command(
            [sys.executable, "-m", "pytest", "-v", "tests/"],
            "Automated Test Suite Verification",
        )
        current_step += 1

    # Step 7: Launch Streamlit Web Application
    print_banner(current_step, total_steps, "Launching Streamlit Healthcare Web Application")
    print(f"{BOLD}{GREEN}All processing, retraining, and validation completed successfully!{RESET}")
    print(f"{CYAN}Starting web dashboard on http://localhost:{args.port} ...{RESET}\n")

    streamlit_cmd = [
        "streamlit", "run", "app.py",
        "--server.port", str(args.port),
    ]
    if args.no_browser:
        streamlit_cmd.extend(["--server.headless", "true"])

    try:
        subprocess.run(streamlit_cmd)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Streamlit server stopped by user.{RESET}")


if __name__ == "__main__":
    main()
