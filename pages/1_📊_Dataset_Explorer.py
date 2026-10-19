"""
Page 1: Dataset Explorer.
Enables transparent inspection of datasets, schemas, cryptographic duplicate audits,
cross-cohort overlap findings, statistical tests, and publication-quality EDA figures.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import (
    DATASET_FILES,
    FIGURES_DIR,
    REPORTS_DIR,
    TABLES_DIR,
)
from src.data_audit import run_full_data_audit
from src.data_loader import load_dataset_by_filename

st.set_page_config(page_title="Dataset Explorer", page_icon="📊", layout="wide")

st.title("📊 Multi-Cohort Dataset Explorer & Data Integrity Audit")
st.markdown(
    "Explore the six benchmark datasets used in this research, inspect descriptive statistics, "
    "verify cryptographic file duplicate checks, and review non-parametric statistical hypothesis tests."
)

# Load audit data
audit_json_path = REPORTS_DIR / "data_audit_report.json"
if audit_json_path.exists():
    with open(audit_json_path, "r") as f:
        audit_data = json.load(f)
else:
    with st.spinner("Running initial data audit..."):
        audit_data = run_full_data_audit()

# Top Level Audit Summary Banner
st.subheader("1. Cryptographic & Multi-Cohort Data Integrity Verification")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("### 🔍 Duplicate File Check")
    clin_check = audit_data["cross_dataset_integrity"].get("clinical_duplicate_check", {})
    if clin_check.get("is_identical"):
        st.error(f"**Identical Duplicate Detected:** `{clin_check.get('file_2')}` is a byte-for-byte clone of `{clin_check.get('file_1')}`.")
        st.caption(f"MD5: `{clin_check.get('file_1_md5')}`")
        st.info("Decision: Excluded file 2 from modeling to prevent double-counting 100,000 records.")
with c2:
    st.markdown("### 🧬 Cohort Identity Check")
    brfss_id = audit_data["cross_dataset_integrity"].get("brfss_binary_vs_012_check", {})
    if brfss_id.get("features_identical"):
        st.success("**Population Match Confirmed:** BRFSS Binary & BRFSS 012 share the exact same 253,680 respondents.")
        st.caption("Binary version collapsed prediabetes (class 1) into non-diabetics (class 0).")
        st.info("Decision: Evaluated independently; never combined or used as external validation sets.")
with c3:
    st.markdown("### ⚖️ Subsample Overlap Check")
    bal_check = audit_data["cross_dataset_integrity"].get("brfss_binary_vs_5050_check", {})
    if bal_check.get("is_subsample"):
        st.warning("**Derived Subsample:** BRFSS 50-50 contains all 35,346 diabetes cases from BRFSS.")
        st.caption("35,346 positive cases perfectly overlap with the full 253k cohort.")
        st.info("Decision: Used strictly to study class imbalance sensitivity, not as an independent test set.")

st.markdown("---")

# Dataset Selection Dropdown
st.subheader("2. Dataset Schema & Descriptive Statistics")

file_options = list(DATASET_FILES.values())
selected_file = st.selectbox("Select Dataset to Inspect:", file_options, index=0)

audit_info = audit_data["dataset_audits"].get(selected_file, {})

if "error" in audit_info:
    st.error(f"Error loading {selected_file}: {audit_info['error']}")
else:
    # Summary Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Records", f"{audit_info['n_rows']:,}")
    with m2:
        st.metric("Total Features", audit_info["n_cols"])
    with m3:
        st.metric("Duplicate Rows", f"{audit_info['n_duplicates']:,} ({audit_info['dup_percentage']}%)")
    with m4:
        st.metric("Missing (Null) Values", audit_info["total_nulls"])
    with m5:
        st.metric("File Size (Disk)", f"{audit_info['file_size_mb']} MB")

    # Target Class Distribution
    if audit_info.get("targets"):
        st.markdown("#### 🎯 Target Variable Distribution")
        t_cols = st.columns(len(audit_info["targets"]))
        for idx, (t_name, t_data) in enumerate(audit_info["targets"].items()):
            with t_cols[idx]:
                st.write(f"**Target Column:** `{t_name}`")
                dist_df = pd.DataFrame({
                    "Class / Label": list(t_data["counts"].keys()),
                    "Count": list(t_data["counts"].values()),
                    "Percentage (%)": [f"{v}%" for v in t_data["percentages"].values()],
                })
                st.dataframe(dist_df, hide_index=True, use_container_width=True)

    # Column Details Table
    with st.expander("🔍 Detailed Column Profiles & Feature Types", expanded=False):
        col_list = []
        for cname, cprofile in audit_info.get("columns", {}).items():
            col_list.append({
                "Column": cname,
                "Data Type": cprofile["dtype"],
                "Unique Values": cprofile["unique_count"],
                "Null Count": cprofile["null_count"],
                "Sample Values": ", ".join(str(s) for s in cprofile.get("samples", [])[:4]),
                "Min": cprofile.get("min", "N/A"),
                "Max": cprofile.get("max", "N/A"),
                "Mean": cprofile.get("mean", "N/A"),
            })
        st.dataframe(pd.DataFrame(col_list), use_container_width=True, hide_index=True)

    # Raw Data Preview
    with st.expander("📄 Sample Data Records (First 15 Rows)", expanded=False):
        df_preview = load_dataset_by_filename(selected_file).head(15)
        st.dataframe(df_preview, use_container_width=True)

st.markdown("---")

# Statistical Analysis & Hypothesis Testing
st.subheader("3. Statistical Hypothesis Testing & Effect Sizes")
st.markdown(
    "To distinguish exploratory findings from statistical noise, we conducted **Mann-Whitney U tests** "
    "for continuous distributions and **Chi-Square Tests of Independence with Cramér's V** for categorical features."
)

stat_tabs = st.tabs(["Clinical Cohort Statistics", "BRFSS Lifestyle Statistics", "Early Symptoms Statistics"])

with stat_tabs[0]:
    t_path = TABLES_DIR / "clinical_statistical_summary.csv"
    if t_path.exists():
        st.dataframe(pd.read_csv(t_path), use_container_width=True)
    else:
        st.info("Run `python -m src.eda` to generate statistical tables.")

with stat_tabs[1]:
    t_path = TABLES_DIR / "brfss_statistical_summary.csv"
    if t_path.exists():
        st.dataframe(pd.read_csv(t_path), use_container_width=True)
    else:
        st.info("Run `python -m src.eda` to generate statistical tables.")

with stat_tabs[2]:
    t_path = TABLES_DIR / "symptoms_statistical_summary.csv"
    if t_path.exists():
        st.dataframe(pd.read_csv(t_path), use_container_width=True)
    else:
        st.info("Run `python -m src.eda` to generate statistical tables.")

st.markdown("---")

# EDA Visualizations Gallery
st.subheader("4. Exploratory Data Analysis Visualizations")

chart_category = st.radio(
    "Select Visual Category:",
    ["Clinical Biomarkers & Demographics", "BRFSS Lifestyle & Behavioral Habits", "Early-Stage Cardinal Symptoms"],
    horizontal=True,
)

if chart_category == "Clinical Biomarkers & Demographics":
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        st.image(str(FIGURES_DIR / "clinical_target_distribution.png"), caption="Clinical Target Class Distribution", use_container_width=True)
        st.image(str(FIGURES_DIR / "clinical_hba1c_glucose_scatter.png"), caption="Biomarker Bivariate Plane: HbA1c vs Glucose", use_container_width=True)
    with r1_col2:
        st.image(str(FIGURES_DIR / "clinical_biomarkers_boxplot.png"), caption="Biomarker Distributions (HbA1c & Glucose)", use_container_width=True)
        st.image(str(FIGURES_DIR / "clinical_correlation_matrix.png"), caption="Clinical Correlation Matrix", use_container_width=True)

elif chart_category == "BRFSS Lifestyle & Behavioral Habits":
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        st.image(str(FIGURES_DIR / "brfss_cohort_distributions.png"), caption="BRFSS Cohort Distribution Comparisons", use_container_width=True)
        st.image(str(FIGURES_DIR / "brfss_age_prevalence.png"), caption="Prevalence by CDC Age Categories", use_container_width=True)
    with r2_col2:
        st.image(str(FIGURES_DIR / "brfss_lifestyle_prevalence.png"), caption="Modifiable Lifestyle Behaviors vs Prevalence", use_container_width=True)
        st.image(str(FIGURES_DIR / "brfss_bmi_genhealth.png"), caption="BMI & Self-Reported General Health Impact", use_container_width=True)

elif chart_category == "Early-Stage Cardinal Symptoms":
    st.image(str(FIGURES_DIR / "symptoms_prevalence_comparison.png"), caption="Cardinal Symptoms Frequency in Diabetic Patients vs Controls", use_container_width=True)
