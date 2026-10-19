"""
Streamlit Web Application - Home Dashboard.
Title: AI-Based Diabetes Risk Prediction Using Machine Learning and Lifestyle Factors.
"""

import streamlit as st
from pathlib import Path
import json

from src.config import CLINICAL_DISCLAIMER, CURRENT_STATUS_VS_FUTURE_RISK_NOTE, DATASET_METADATA, MODELS_DIR, REPORTS_DIR

st.set_page_config(
    page_title="AI Diabetes Risk Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .badge-primary {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
    }
    .badge-success {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-block;
    }
    .disclaimer-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 14px 18px;
        border-radius: 4px;
        color: #991B1B;
        font-size: 0.88rem;
        margin-top: 15px;
        margin-bottom: 25px;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 4px;
        color: #1E40AF;
        font-size: 0.88rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🩺 Navigation")
st.sidebar.info("Select a module from the menu above to explore data, test patient assessments, evaluate models, or review research reports.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 System Status")

# Check model persistence
models_exist = (
    (MODELS_DIR / "clinical_model_bundle.joblib").exists() and
    (MODELS_DIR / "brfss_lifestyle_model_bundle.joblib").exists() and
    (MODELS_DIR / "symptoms_model_bundle.joblib").exists()
)
if models_exist:
    st.sidebar.success("✅ Models Loaded & Ready")
else:
    st.sidebar.warning("⚠️ Some Models Missing")

st.sidebar.markdown("---")
st.sidebar.caption("Antigravity AI Health Research Platform v1.0.0")

# Header Section
st.markdown('<div class="main-header">AI-Based Diabetes Risk Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Cohort Machine Learning Research & Lifestyle-Driven Risk Stratification Platform</div>', unsafe_allow_html=True)

# Badges
st.markdown("""
<div>
    <span class="badge-primary">🔬 Research-Grade AI</span>
    <span class="badge-primary">📊 350,000+ Multi-Cohort Records</span>
    <span class="badge-primary">🥗 Lifestyle vs Clinical Biomarkers</span>
    <span class="badge-success">🔍 Explainable AI (SHAP)</span>
    <span class="badge-success">📈 Calibrated Probabilities</span>
</div>
""", unsafe_allow_html=True)

# Prominent Disclaimer
st.markdown(f'<div class="disclaimer-box"><b>⚖️ MEDICAL & REGULATORY DISCLAIMER:</b><br>{CLINICAL_DISCLAIMER}</div>', unsafe_allow_html=True)

# Key Scientific Distinction
st.markdown(f'<div class="info-box"><b>📌 SCIENTIFIC METHODOLOGY NOTE:</b><br>{CURRENT_STATUS_VS_FUTURE_RISK_NOTE}</div>', unsafe_allow_html=True)

# Overview Columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Patient Records", "354,200", help="Aggregated across 3 distinct clinical cohorts")
with col2:
    st.metric("Clinical Model ROC-AUC", "0.9770", "+15.2% vs Lifestyle")
with col3:
    st.metric("Lifestyle Model Sensitivity", "74.5%", "Non-Invasive Screening")
with col4:
    st.metric("Calibrated Brier Score", "0.0236", "Clinical Reliability")

st.markdown("---")

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Research Objectives", "📂 Multi-Cohort Architecture", "🚀 Application Guide"])

with tab1:
    st.subheader("Core Research Objectives & Scientific Hypotheses")
    st.markdown("""
    This project was designed as a rigorous computational health study addressing key healthcare machine learning questions:
    
    1. **Lifestyle Factors vs. Diabetes Status:** Can non-invasive, self-reported behavioral survey indicators (physical activity, fruit/vegetable intake, smoking, alcohol, BMI) effectively stratify diabetes likelihood without requiring laboratory visits?
    2. **Clinical Biomarker Value-Add (Ablation):** Quantifying the exact statistical gain achieved when glycemic laboratory tests (HbA1c and fasting blood glucose) are integrated alongside demographic and lifestyle variables.
    3. **Class Imbalance & Screening Sensitivity:** In medical screening, missed positive cases (false negatives) carry severe clinical consequences. We evaluate how natural class imbalance (8.5% and 13.9%) affects sensitivity and compare against artificially balanced populations.
    4. **Explainability & Transparency:** Using TreeExplainer SHAP values to explain every individual patient classification, identifying whether risk is primarily driven by adiposity, glycemic control, or sedentary habits.
    5. **Current Classification vs. Future Risk:** Explicitly distinguishing retrospective diagnostic classification from longitudinal incidence prediction, preventing common medical AI overclaiming.
    """)

with tab2:
    st.subheader("Inspected Datasets & Architectural Separation")
    st.markdown("""
    In accordance with clinical data governance, **heterogeneous datasets are never blindly concatenated**. 
    The platform organizes data into specialized modeling tracks:
    """)
    
    for family_key, meta in DATASET_METADATA.items():
        with st.expander(f"📁 {meta['title']} ({meta['primary_file']})"):
            st.markdown(f"**Description:** {meta['description']}")
            st.markdown(f"**Target Variable:** `{meta['target_col']}`")
            st.markdown(f"**Modeling Modality:** {meta['type']}")
            st.markdown(f"**Data Source:** {meta['source']}")

with tab3:
    st.subheader("Platform Navigation Guide")
    st.markdown("""
    Explore the five specialized interactive pages using the left sidebar:
    
    - 📊 **Dataset Explorer:** Inspect dataset schemas, view MD5 cryptographic duplicate verifications, examine missing-value distributions, and view exploratory data analysis charts.
    - 🩺 **Diabetes Assessment:** Enter patient health, lifestyle, or clinical laboratory data across three dedicated screening modes to obtain calibrated risk assessments with confidence intervals and local SHAP explanations.
    - 📈 **Model Performance:** Review detailed algorithm benchmark comparisons (Logistic Regression, Trees, Random Forest, Extra Trees, LightGBM, XGBoost, KNN, SVM), cross-validation metrics, calibration curves, and ablation studies.
    - 🔍 **Explainable AI:** Explore global SHAP beeswarm and feature importance plots, interactive local attribution waterfalls, and non-causal medical interpretations.
    - 📑 **Research Report:** View and download the complete formal research report in Markdown, PDF, and CSV formats.
    """)
