"""
Page 3: Model Performance.
Presents comprehensive empirical benchmark tables, 5-fold cross-validation metrics,
diagnostic ROC/PR/Calibration curves, confusion matrices, and scientific ablation studies.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import FIGURES_DIR, TABLES_DIR

st.set_page_config(page_title="Model Performance & Benchmarks", page_icon="📈", layout="wide")

st.title("📈 Model Performance & Comparative Benchmark Dashboard")
st.markdown(
    "Transparently evaluate all trained machine learning algorithms across each cohort. "
    "In health-risk screening, **Sensitivity (Recall)** and **Probability Calibration (Brier Score)** "
    "are critically prioritized alongside standard ROC-AUC."
)

cohort_choice = st.selectbox(
    "Select Clinical Cohort to Inspect:",
    [
        "Clinical & Laboratory Cohort (100,000 Records)",
        "CDC BRFSS Lifestyle Survey (Natural Imbalance 13.9%)",
        "CDC BRFSS 50-50 Balanced Subsample (Class Imbalance Study)",
        "Early-Stage Symptoms Cohort (Sylhet Hospital)",
    ],
    index=0,
)

cohort_file_map = {
    "Clinical & Laboratory Cohort (100,000 Records)": {
        "table": "clinical_model_comparison.csv",
        "diag": "clinical_best_diagnostics.png",
        "name": "Clinical & Laboratory Cohort",
    },
    "CDC BRFSS Lifestyle Survey (Natural Imbalance 13.9%)": {
        "table": "brfss_lifestyle_model_comparison.csv",
        "diag": "brfss_lifestyle_best_diagnostics.png",
        "name": "CDC BRFSS Lifestyle Cohort",
    },
    "CDC BRFSS 50-50 Balanced Subsample (Class Imbalance Study)": {
        "table": "brfss_5050_balanced_model_comparison.csv",
        "diag": "brfss_5050_balanced_best_diagnostics.png",
        "name": "BRFSS 50-50 Balanced Cohort",
    },
    "Early-Stage Symptoms Cohort (Sylhet Hospital)": {
        "table": "early_symptoms_model_comparison.csv",
        "diag": "early_symptoms_best_diagnostics.png",
        "name": "Early-Stage Symptoms Cohort",
    },
}

meta = cohort_file_map[cohort_choice]
t_path = TABLES_DIR / meta["table"]
d_path = FIGURES_DIR / meta["diag"]

if t_path.exists():
    df_perf = pd.read_csv(t_path)
    
    st.subheader(f"1. Algorithm Benchmark Comparison - {meta['name']}")
    
    # Highlight columns
    display_cols = [
        "model_name", "accuracy", "recall_sensitivity", "specificity",
        "precision", "f1_score", "roc_auc", "brier_score"
    ]
    if "cv_roc_auc_mean" in df_perf.columns:
        display_cols.append("cv_roc_auc_mean")
    if "fit_time_seconds" in df_perf.columns:
        display_cols.append("fit_time_seconds")

    clean_table = df_perf[[c for c in display_cols if c in df_perf.columns]].copy()
    clean_table = clean_table.rename(columns={
        "model_name": "Algorithm",
        "accuracy": "Accuracy",
        "recall_sensitivity": "Sensitivity (Recall)",
        "specificity": "Specificity",
        "precision": "Precision",
        "f1_score": "F1 Score",
        "roc_auc": "ROC-AUC",
        "brier_score": "Brier Score",
        "cv_roc_auc_mean": "CV Mean AUC",
        "fit_time_seconds": "Train Time (s)",
    })
    st.dataframe(clean_table.style.highlight_max(subset=["Sensitivity (Recall)", "ROC-AUC"], color="#D1FAE5"), use_container_width=True)

    # Diagnostic Curves
    st.markdown("---")
    st.subheader("2. Comprehensive Diagnostic Evaluation Panel")
    st.markdown(
        "4-panel diagnostic analysis on the **untouched test split**: "
        "Receiver Operating Characteristic (ROC), Precision-Recall Curve, Reliability Diagram (Probability Calibration), and Error Matrix."
    )
    if d_path.exists():
        st.image(str(d_path), use_container_width=True)
    else:
        st.info("Diagnostic chart not found. Run training script to generate.")

else:
    st.warning("Performance table not found. Please run `python -m src.train_models` first.")

st.markdown("---")

# Dedicated Research Experiments Section
st.subheader("3. Dedicated Research-Level Analyses")

exp_tabs = st.tabs([
    "🔬 Ablation Study: Clinical Biomarkers vs Lifestyle",
    "⚖️ Class Imbalance & Sensitivity Trade-Offs",
    "🧪 Prediabetes Multi-Class Detection Challenge",
])

# EXP 1: ABLATION STUDY
with exp_tabs[0]:
    st.markdown("#### Research Question: Do Laboratory Biomarkers Truly Improve Prediction Over Lifestyle Factors?")
    st.markdown(
        "To evaluate whether expensive or invasive laboratory blood tests provide meaningful clinical value over "
        "lifestyle and demographic questionnaires alone, we performed an ablation experiment holding the patient cohort and test partitions identical."
    )

    ablation_path = TABLES_DIR / "ablation_study_comparison.csv"
    if ablation_path.exists():
        df_abl = pd.read_csv(ablation_path)
        st.dataframe(
            df_abl[["model_name", "accuracy", "recall_sensitivity", "precision", "roc_auc", "Delta_Gain"]].rename(columns={
                "model_name": "Experiment Condition",
                "accuracy": "Accuracy",
                "recall_sensitivity": "Sensitivity (Recall)",
                "precision": "Precision",
                "roc_auc": "ROC-AUC",
                "Delta_Gain": "Diagnostic Value Gain",
            }),
            use_container_width=True,
            hide_index=True,
        )
    st.markdown("""
    **Core Findings:**
    - Non-invasive lifestyle features (Age, Gender, BMI, Smoking, Hypertension, Heart Disease) achieve a solid baseline **ROC-AUC of 0.8251** and **79.7% Sensitivity**.
    - Adding laboratory measurements (`HbA1c_level` and `blood_glucose_level`) yields a dramatic **+0.152 boost in ROC-AUC** (reaching **0.9770**) and increases Sensitivity to **92.1%**.
    - **Clinical Takeaway:** Lifestyle surveys are highly cost-effective for primary population screening, but clinical biomarkers are indispensable for definitive clinical triage.
    """)

# EXP 2: CLASS IMBALANCE
with exp_tabs[1]:
    st.markdown("#### Research Question: How Does Severe Class Imbalance Affect Clinical Sensitivity?")
    st.markdown("""
    In the full CDC BRFSS survey, only **13.9%** of respondents are diabetic.
    When training default models without class weighting (e.g. default XGBoost), the algorithm optimizes for majority-class accuracy (achieving 86% overall accuracy), but Sensitivity drops to a dismal **14.5%**, missing over **85% of diabetic individuals**!
    
    By employing **balanced class weights** and **calibrated decision thresholds**, our tuned LightGBM and HistGradientBoosting models recover Sensitivity to **74.5%**, properly protecting against dangerous false negatives.
    """)
    st.image(str(FIGURES_DIR / "brfss_cohort_distributions.png"), caption="Comparison of Natural Imbalance (13.9%) vs Balanced Subsample (50.0%)", use_container_width=True)

# EXP 3: PREDIABETES DETECTION
with exp_tabs[2]:
    st.markdown("#### Research Question: Can Lifestyle Surveys Distinguish Prediabetes from Healthy Individuals?")
    st.markdown("""
    In the CDC BRFSS 3-class dataset (`diabetes_012`), **Prediabetes accounts for only 1.83%** of respondents (4,631 cases out of 253,680).
    
    Multi-class modeling revealed that Prediabetes is virtually indistinguishable from healthy controls based on lifestyle questionnaires alone (Macro F1: ~0.42). 
    Because prediabetes is clinically asymptomatic and lacks overt manifestations like polyuria, it requires fasting plasma glucose or oral glucose tolerance testing for reliable detection.
    """)
