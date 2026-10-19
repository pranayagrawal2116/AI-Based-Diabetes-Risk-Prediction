"""
Page 4: Explainable AI (XAI).
Presents global SHAP summary beeswarms, mean absolute importance rankings,
an interactive local SHAP prediction simulator, and non-causal interpretability discussions.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import FIGURES_DIR
from src.prediction_service import PredictionService

st.set_page_config(page_title="Explainable AI & SHAP", page_icon="🔍", layout="wide")

st.title("🔍 Explainable AI (XAI) & Feature Attribution")
st.markdown(
    "Black-box predictions are inadequate in healthcare. Using **SHAP (SHapley Additive exPlanations)** "
    "grounded in cooperative game theory, this platform uncovers both the global drivers of risk across population cohorts "
    "and the exact patient-specific factors influencing any individual prediction."
)

st.info(
    "⚖️ **Critical Medical Distinction: Predictive Associations vs. Causal Mechanisms**<br>"
    "A high SHAP value indicates that a feature strongly correlates with diabetes classification in observational data. "
    "It does NOT establish that modifying that variable alone will directly cure or cause diabetes. "
    "Causal claims require randomized controlled trials (RCTs) or longitudinal target trial emulation.",
    icon="📌"
)

# Global vs Local Explanations Tabs
xai_tabs = st.tabs([
    "🌐 Global Cohort Drivers (SHAP Beeswarm & Importance)",
    "🔬 Interactive Patient-Level SHAP Simulator",
    "📚 Clinical Feature Interpretability Guide",
])

# ==========================================
# TAB 1: GLOBAL SHAP
# ==========================================
with xai_tabs[0]:
    st.subheader("Global Feature Attribution Across Cohorts")
    cohort_view = st.radio(
        "Select Clinical Cohort:",
        ["Clinical & Laboratory Cohort", "CDC BRFSS Lifestyle Survey", "Early-Stage Symptoms Cohort"],
        horizontal=True,
    )

    if cohort_view == "Clinical & Laboratory Cohort":
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.image(str(FIGURES_DIR / "clinical_shap_summary.png"), caption="SHAP Beeswarm Plot: Direction & Magnitude of Biomarkers", use_container_width=True)
        with c2:
            st.image(str(FIGURES_DIR / "clinical_shap_bar.png"), caption="Mean Absolute SHAP Value (Global Importance)", use_container_width=True)

        st.markdown("""
        **Clinical Interpretability Notes:**
        - `HbA1c_level` and `blood_glucose_level` vastly overshadow all other features in predictive power.
        - Red points (high HbA1c / glucose) heavily push log-odds to the right (higher diabetic likelihood).
        - `bmi` and `age` provide secondary stratification, with elevated BMI and age >45 elevating baseline risk.
        """)

    elif cohort_view == "CDC BRFSS Lifestyle Survey":
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.image(str(FIGURES_DIR / "brfss_lifestyle_shap_summary.png"), caption="SHAP Beeswarm: Non-Invasive Lifestyle & Health Drivers", use_container_width=True)
        with c2:
            st.image(str(FIGURES_DIR / "brfss_lifestyle_shap_bar.png"), caption="Mean Absolute SHAP Importance (BRFSS)", use_container_width=True)

        st.markdown("""
        **Lifestyle Interpretability Notes:**
        - `GenHlth` (self-rated general health on 1-5 scale) is the #1 non-invasive predictor. Individuals reporting 'Fair' or 'Poor' health have dramatically higher odds of diabetes.
        - `HighBP` (hypertension) and `BMI` are the top metabolic markers.
        - `PhysActivity`: Blue points (lack of physical activity, value=0) shift model predictions rightward towards higher risk.
        """)

    elif cohort_view == "Early-Stage Symptoms Cohort":
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.image(str(FIGURES_DIR / "early_symptoms_shap_summary.png"), caption="SHAP Beeswarm: Cardinal Diabetic Symptoms", use_container_width=True)
        with c2:
            st.image(str(FIGURES_DIR / "early_symptoms_shap_bar.png"), caption="Mean Absolute SHAP Importance (Sylhet Cohort)", use_container_width=True)

        st.markdown("""
        **Symptom Triage Notes:**
        - `Polyuria` (frequent urination) and `Polydipsia` (excessive thirst) constitute over 70% of model attribution.
        - Osmotic diuresis is the direct physiologic hallmark of persistent blood glucose exceeding renal reabsorption thresholds (~180 mg/dL).
        """)

# ==========================================
# TAB 2: INTERACTIVE LOCAL SHAP SIMULATOR
# ==========================================
with xai_tabs[1]:
    st.subheader("Interactive Single-Patient SHAP Attribution Simulator")
    st.markdown(
        "Adjust patient values below to observe in real time how local SHAP values dynamically shift "
        "and explain the model's decision."
    )

    @st.cache_resource
    def get_prediction_service():
        return PredictionService()

    svc = get_prediction_service()

    sim_col1, sim_col2 = st.columns([1, 1.2])

    with sim_col1:
        st.markdown("#### Patient Biomarker Inputs")
        s_hba1c = st.slider("HbA1c Level (%)", 4.0, 11.0, 6.2, 0.1, key="sim_hba1c")
        s_glucose = st.slider("Blood Glucose (mg/dL)", 70, 300, 150, 5, key="sim_glu")
        s_bmi = st.slider("BMI", 15.0, 55.0, 31.0, 0.5, key="sim_bmi")
        s_age = st.slider("Age (Years)", 18.0, 85.0, 54.0, 1.0, key="sim_age")
        s_htn = st.selectbox("Hypertension", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No", key="sim_htn")
        s_hd = st.selectbox("Heart Disease", [0, 1], index=0, format_func=lambda x: "Yes" if x == 1 else "No", key="sim_hd")
        s_smoke = st.selectbox("Smoking History", ["never", "former", "current", "No Info"], index=1, key="sim_smk")
        s_gender = st.selectbox("Gender", ["Female", "Male"], key="sim_gen")

    sim_input = {
        "gender": s_gender, "age": float(s_age), "hypertension": int(s_htn),
        "heart_disease": int(s_hd), "smoking_history": s_smoke, "bmi": float(s_bmi),
        "HbA1c_level": float(s_hba1c), "blood_glucose_level": int(s_glucose)
    }
    res_sim = svc.predict_clinical(sim_input)

    with sim_col2:
        st.markdown("#### Dynamic SHAP Feature Decomposition")
        st.metric(
            "Estimated Probability",
            f"{res_sim['calibrated_probability']:.1%}",
            delta=res_sim['risk_stratification']['badge'],
        )

        expl = res_sim.get("local_explanation", {})
        if expl and "contributions" in expl:
            top_k = expl["contributions"][:6]
            plot_df = pd.DataFrame({
                "Feature": [c["feature"] for c in top_k],
                "SHAP Impact": [c["shap_value"] for c in top_k],
            }).set_index("Feature")
            st.bar_chart(plot_df)

            st.write("##### Detailed Factor Narratives:")
            for c in top_k[:4]:
                st.write(f"- **{c['feature']}** ({c['direction']}): {c['clinical_narrative']}")

# ==========================================
# TAB 3: FEATURE GUIDE
# ==========================================
with xai_tabs[2]:
    st.subheader("📚 Clinical Reference Guide: Feature Definitions & Physiological Context")
    st.markdown("""
    | Feature | Modality | Normal Range | Clinical Association with Diabetes Mellitus |
    | :--- | :--- | :--- | :--- |
    | **HbA1c Level** | Laboratory Blood Test | < 5.7% | Reflects average glycated hemoglobin over 90-120 days. ADA gold standard diagnostic biomarker. |
    | **Blood Glucose** | Laboratory Plasma | 70 - 99 mg/dL (fasting) | Direct concentration of circulating glucose. Acute diagnostic indicator. |
    | **Body Mass Index (BMI)** | Anthropometric | 18.5 - 24.9 kg/m² | Correlates with visceral adiposity, free fatty acid release, and peripheral insulin resistance. |
    | **Physical Activity** | Modifiable Lifestyle | ≥ 150 min/week | Stimulates GLUT4 translocation in skeletal muscle, directly clearing glucose independently of insulin. |
    | **General Health (GenHlth)** | Self-Reported Survey | 1 (Excellent) to 5 (Poor) | Highly sensitive subjective composite marker of chronic cardiometabolic burden. |
    | **Polyuria & Polydipsia** | Cardinal Symptoms | Absent | Occurs when blood glucose exceeds renal threshold (~180 mg/dL), causing osmotic diuresis and dehydration. |
    """)
