"""
Page 2: Diabetes Assessment.
Interactive clinical assessment portal supporting distinct screening modes:
Lifestyle & General Health, Clinical Biomarkers, and Early-Stage Cardinal Symptoms.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.config import (
    BRFSS_AGE_MAP,
    BRFSS_EDUCATION_MAP,
    BRFSS_GENHLTH_MAP,
    BRFSS_INCOME_MAP,
    CLINICAL_DISCLAIMER,
    CURRENT_STATUS_VS_FUTURE_RISK_NOTE,
)
from src.prediction_service import PredictionService

st.set_page_config(page_title="Diabetes Risk Assessment", page_icon="🩺", layout="wide")

st.title("🩺 Interactive Diabetes Risk Assessment")
st.markdown(
    "Select an assessment mode tailored to the data available. The platform uses dedicated, "
    "leakage-free machine learning pipelines for non-invasive lifestyle screening, clinical laboratory measurements, "
    "and early-stage symptom triage."
)

st.warning(f"⚖️ **Notice:** {CLINICAL_DISCLAIMER}")

# Assessment Mode Tabs
mode_tabs = st.tabs([
    "🥗 Lifestyle & General Health Assessment",
    "🔬 Clinical & Laboratory Assessment",
    "⚠️ Early-Stage Symptom Screening",
    "🧪 3-Class Prediabetes vs Diabetes",
])

# Initialize Prediction Service
@st.cache_resource
def get_service():
    return PredictionService()

service = get_service()


def render_prediction_results(res: dict):
    """Renders standardized healthcare assessment results with uncertainty and SHAP."""
    st.markdown("---")
    st.subheader("📋 Assessment Results & Risk Stratification")

    risk_info = res.get("risk_stratification", {})
    prob = res.get("calibrated_probability", 0.0)
    ci = res.get("confidence_interval_95", [0.0, 1.0])
    color = risk_info.get("color", "#2563EB")

    # High level result cards
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        st.markdown(f"""
        <div style="background-color: {color}15; border-left: 5px solid {color}; padding: 15px; border-radius: 6px;">
            <div style="font-size: 0.85rem; color: #4B5563; font-weight: 600;">ESTIMATED RISK CATEGORY</div>
            <div style="font-size: 1.6rem; color: {color}; font-weight: 800; margin: 4px 0;">{risk_info.get('badge', 'RISK ASSESSMENT')}</div>
            <div style="font-size: 0.88rem; color: #374151;">{risk_info.get('description', '')}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.metric(
            label="Calibrated Probability",
            value=f"{prob:.1%}",
            delta=f"95% CI: [{ci[0]:.1%} - {ci[1]:.1%}]",
            delta_color="off",
            help="Probability calibrated using sigmoid/isotonic regression on out-of-fold validation data.",
        )

    with c3:
        st.metric(
            label="Model Classification",
            value=res.get("predicted_class", "Evaluated"),
            help="Threshold decision boundary at 50% probability.",
        )

    # Scientific Distinction Note
    st.info(f"📌 **Current Status vs. Future Risk:** {CURRENT_STATUS_VS_FUTURE_RISK_NOTE}")

    # Local Feature Attribution (SHAP)
    expl = res.get("local_explanation", {})
    if expl and "contributions" in expl:
        st.subheader("🔍 Contributing Factors (Explainable AI Attribution)")
        st.markdown(
            "Below is the breakdown of individual factors that influenced your estimated risk. "
            "Positive values push risk higher; negative values lower the estimated risk."
        )

        contrib_list = expl["contributions"][:6]
        df_contrib = pd.DataFrame(contrib_list)

        col_left, col_right = st.columns([1.3, 1])
        with col_left:
            st.dataframe(
                df_contrib[["feature", "direction", "relative_influence_pct", "clinical_narrative"]].rename(
                    columns={
                        "feature": "Feature",
                        "direction": "Direction of Influence",
                        "relative_influence_pct": "Relative Impact (%)",
                        "clinical_narrative": "Clinical Context",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        with col_right:
            # Bar chart of contributions
            chart_df = pd.DataFrame({
                "Feature": [c["feature"] for c in contrib_list],
                "SHAP Impact": [c["shap_value"] for c in contrib_list],
            }).set_index("Feature")
            st.bar_chart(chart_df)

    # Next Steps & Guidance
    st.subheader("💡 Recommended Next Steps")
    if prob >= 0.50:
        st.markdown("""
        - 🩺 **Schedule a Clinical Evaluation:** We recommend consulting a primary healthcare provider for confirmatory laboratory testing (fasting plasma glucose or HbA1c test).
        - 🥗 **Dietary Optimization:** Reduce intake of refined sugars and ultra-processed carbohydrates; emphasize high-fiber vegetables, whole grains, and lean proteins.
        - 🏃 **Physical Activity:** Aim for at least 150 minutes of moderate-intensity aerobic exercise per week as recommended by the American Diabetes Association (ADA).
        """)
    else:
        st.markdown("""
        - 🥦 **Maintain Healthy Lifestyle:** Continue balanced nutritional habits rich in dietary fiber, vegetables, and whole foods.
        - 🚶 **Sustain Daily Activity:** Maintain regular daily physical movement and an active routine.
        - 📅 **Routine Screening:** Re-evaluate periodic health markers annually with your primary care physician.
        """)


# ==========================================
# TAB 1: LIFESTYLE & GENERAL HEALTH (BRFSS)
# ==========================================
with mode_tabs[0]:
    st.markdown("### 🥗 Lifestyle and General Health Assessment (CDC BRFSS)")
    st.caption("Based on 253,680 CDC BRFSS survey records. Evaluates modifiable habits without requiring blood draws.")

    with st.form("form_lifestyle"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 👤 Demographics")
            life_sex = st.selectbox("Sex", options=[("Female", 0.0), ("Male", 1.0)], format_func=lambda x: x[0])[1]
            life_age_idx = st.selectbox("Age Bracket", options=list(BRFSS_AGE_MAP.keys()), format_func=lambda x: BRFSS_AGE_MAP[x])
            life_bmi = st.number_input("Body Mass Index (BMI)", min_value=12.0, max_value=75.0, value=26.5, step=0.1, help="Weight (kg) / Height (m)^2")
            life_edu = st.selectbox("Education Level", options=list(BRFSS_EDUCATION_MAP.keys()), format_func=lambda x: BRFSS_EDUCATION_MAP[x], index=3)
            life_inc = st.selectbox("Household Income", options=list(BRFSS_INCOME_MAP.keys()), format_func=lambda x: BRFSS_INCOME_MAP[x], index=5)

        with c2:
            st.markdown("#### 🏃 Lifestyle Habits")
            life_phys = st.radio("Routine Physical Activity (Past 30 Days)", options=[("Yes", 1.0), ("No", 0.0)], format_func=lambda x: x[0], horizontal=True)[1]
            life_fruits = st.radio("Consumes Fruit Daily (>=1 time/day)", options=[("Yes", 1.0), ("No", 0.0)], format_func=lambda x: x[0], horizontal=True)[1]
            life_veg = st.radio("Consumes Vegetables Daily (>=1 time/day)", options=[("Yes", 1.0), ("No", 0.0)], format_func=lambda x: x[0], horizontal=True)[1]
            life_smoker = st.radio("Ever Smoked >=100 Cigarettes in Lifetime", options=[("No", 0.0), ("Yes", 1.0)], format_func=lambda x: x[0], horizontal=True)[1]
            life_alcohol = st.radio("Heavy Alcohol Consumption (Men >=14, Women >=7 drinks/week)", options=[("No", 0.0), ("Yes", 1.0)], format_func=lambda x: x[0], horizontal=True)[1]

        with c3:
            st.markdown("#### 🩺 Health Conditions & Perception")
            life_bp = st.radio("High Blood Pressure History", options=[("No", 0.0), ("Yes", 1.0)], format_func=lambda x: x[0], horizontal=True)[1]
            life_chol = st.radio("High Cholesterol History", options=[("No", 0.0), ("Yes", 1.0)], format_func=lambda x: x[0], horizontal=True)[1]
            life_cholcheck = st.radio("Cholesterol Checked in Past 5 Years", options=[("Yes", 1.0), ("No", 0.0)], format_func=lambda x: x[0], horizontal=True)[1]
            life_stroke = st.radio("History of Stroke", options=[("No", 0.0), ("Yes", 1.0)], format_func=lambda x: x[0], horizontal=True)[1]
            life_heart = st.radio("Coronary Heart Disease / Myocardial Infarction", options=[("No", 0.0), ("Yes", 1.0)], format_func=lambda x: x[0], horizontal=True)[1]
            life_genhlth = st.selectbox("Self-Reported General Health", options=list(BRFSS_GENHLTH_MAP.keys()), format_func=lambda x: BRFSS_GENHLTH_MAP[x], index=1)
            life_physhlth = st.slider("Days of Physical Illness / Injury (Past 30 Days)", min_value=0, max_value=30, value=2)
            life_menthlth = st.slider("Days of Poor Mental Health (Past 30 Days)", min_value=0, max_value=30, value=2)
            life_diffwalk = st.radio("Serious Difficulty Walking / Stairs", options=[("No", 0.0), ("Yes", 1.0)], format_func=lambda x: x[0], horizontal=True)[1]
            life_healthcare = st.radio("Has Any Healthcare Coverage", options=[("Yes", 1.0), ("No", 0.0)], format_func=lambda x: x[0], horizontal=True)[1]
            life_nodoc = st.radio("Could Not Afford Doctor Visit in Past 12 Months", options=[("No", 0.0), ("Yes", 1.0)], format_func=lambda x: x[0], horizontal=True)[1]

        submit_lifestyle = st.form_submit_button("Submit Lifestyle Assessment", use_container_width=True)

    if submit_lifestyle:
        input_dict = {
            "HighBP": life_bp, "HighChol": life_chol, "CholCheck": life_cholcheck, "BMI": float(life_bmi),
            "Smoker": life_smoker, "Stroke": life_stroke, "HeartDiseaseorAttack": life_heart,
            "PhysActivity": life_phys, "Fruits": life_fruits, "Veggies": life_veg,
            "HvyAlcoholConsump": life_alcohol, "AnyHealthcare": life_healthcare,
            "NoDocbcCost": life_nodoc, "GenHlth": float(life_genhlth), "MentHlth": float(life_menthlth),
            "PhysHlth": float(life_physhlth), "DiffWalk": life_diffwalk, "Sex": life_sex,
            "Age": float(life_age_idx), "Education": float(life_edu), "Income": float(life_inc)
        }
        with st.spinner("Evaluating lifestyle risk with calibrated model..."):
            res = service.predict_lifestyle(input_dict)
            render_prediction_results(res)


# ==========================================
# TAB 2: CLINICAL MEASUREMENTS (EHR)
# ==========================================
with mode_tabs[1]:
    st.markdown("### 🔬 Clinical and Laboratory Assessment")
    st.caption("Incorporates gold-standard glycemic biomarkers (HbA1c and Blood Glucose) along with medical history.")

    with st.form("form_clinical"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 👤 Demographics & History")
            clin_gender = st.selectbox("Gender", ["Female", "Male"])
            clin_age = st.number_input("Age (Years)", min_value=1.0, max_value=100.0, value=50.0, step=1.0)
            clin_bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=28.0, step=0.1)
            clin_smoking = st.selectbox("Smoking History", ["never", "former", "current", "ever", "not current", "No Info"])
            clin_htn = st.radio("Diagnosed Hypertension", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], horizontal=True)[1]
            clin_hd = st.radio("Diagnosed Heart Disease", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], horizontal=True)[1]

        with c2:
            st.markdown("#### 🩸 Glycemic Laboratory Biomarkers")
            st.info("ℹ️ Standard Clinical Diagnostic Cutoffs:\n- **HbA1c:** Normal <5.7%, Prediabetes 5.7-6.4%, Diabetes ≥6.5%\n- **Fasting Glucose:** Normal <100 mg/dL, Prediabetes 100-125 mg/dL, Diabetes ≥126 mg/dL")
            clin_hba1c = st.slider("Glycated Hemoglobin (HbA1c %)", min_value=3.5, max_value=12.0, value=5.9, step=0.1)
            clin_glucose = st.slider("Blood Glucose Level (mg/dL)", min_value=70, max_value=350, value=130, step=1)

        submit_clinical = st.form_submit_button("Submit Clinical Assessment", use_container_width=True)

    if submit_clinical:
        input_dict = {
            "gender": clin_gender,
            "age": float(clin_age),
            "hypertension": int(clin_htn),
            "heart_disease": int(clin_hd),
            "smoking_history": clin_smoking,
            "bmi": float(clin_bmi),
            "HbA1c_level": float(clin_hba1c),
            "blood_glucose_level": int(clin_glucose),
        }
        with st.spinner("Evaluating clinical model with SHAP explanations..."):
            res = service.predict_clinical(input_dict)
            render_prediction_results(res)


# ==========================================
# TAB 3: EARLY-STAGE SYMPTOMS (SYLHET)
# ==========================================
with mode_tabs[2]:
    st.markdown("### ⚠️ Early-Stage Symptom Triage (Hospital Questionnaire)")
    st.caption("Based on 520 acute clinical presentations documenting cardinal symptoms like osmotic diuresis.")

    with st.form("form_symptoms"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### 👤 Patient Profile")
            sym_age = st.number_input("Patient Age", min_value=16, max_value=95, value=48)
            sym_gender = st.selectbox("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
            st.markdown("#### 💧 Hallmark Osmotic Signs")
            sym_polyuria = st.checkbox("Polyuria (Excessive / Frequent Urination)", value=False)
            sym_polydipsia = st.checkbox("Polydipsia (Excessive / Unquenchable Thirst)", value=False)
            sym_polyphagia = st.checkbox("Polyphagia (Excessive Hunger)", value=False)
            sym_weightloss = st.checkbox("Sudden Unexplained Weight Loss", value=False)

        with c2:
            st.markdown("#### ⚡ Systemic Symptoms")
            sym_weakness = st.checkbox("Generalized Weakness / Fatigue", value=False)
            sym_blur = st.checkbox("Visual Blurring", value=False)
            sym_irritability = st.checkbox("Irritability", value=False)
            sym_delayed_healing = st.checkbox("Delayed Wound Healing", value=False)
            sym_itching = st.checkbox("Persistent Skin Itching", value=False)

        with c3:
            st.markdown("#### 🩺 Physical Signs")
            sym_paresis = st.checkbox("Partial Paresis / Muscle Weakness", value=False)
            sym_stiffness = st.checkbox("Muscle Stiffness", value=False)
            sym_alopecia = st.checkbox("Alopecia (Hair Loss)", value=False)
            sym_obesity = st.checkbox("Obesity", value=False)
            sym_thrush = st.checkbox("Genital Thrush / Yeast Infection", value=False)

        submit_symptoms = st.form_submit_button("Submit Symptom Screening", use_container_width=True)

    if submit_symptoms:
        input_dict = {
            "Age": int(sym_age),
            "Gender": int(sym_gender),
            "Polyuria": int(sym_polyuria),
            "Polydipsia": int(sym_polydipsia),
            "sudden weight loss": int(sym_weightloss),
            "weakness": int(sym_weakness),
            "Polyphagia": int(sym_polyphagia),
            "Genital thrush": int(sym_thrush),
            "visual blurring": int(sym_blur),
            "Itching": int(sym_itching),
            "Irritability": int(sym_irritability),
            "delayed healing": int(sym_delayed_healing),
            "partial paresis": int(sym_paresis),
            "muscle stiffness": int(sym_stiffness),
            "Alopecia": int(sym_alopecia),
            "Obesity": int(sym_obesity),
        }
        with st.spinner("Evaluating symptom burden..."):
            res = service.predict_symptoms(input_dict)
            render_prediction_results(res)


# ==========================================
# TAB 4: MULTICLASS PREDIABETES VS DIABETES
# ==========================================
with mode_tabs[3]:
    st.markdown("### 🧪 BRFSS 3-Class Multiclass Assessment (Prediabetes vs Diabetes)")
    st.caption("Evaluates whether an individual is classified as Healthy, Prediabetic, or Diabetic based on BRFSS survey indicators.")

    with st.form("form_multiclass"):
        mc_c1, mc_c2 = st.columns(2)
        with mc_c1:
            mc_sex = st.selectbox("Sex (0=Female, 1=Male)", [0.0, 1.0])
            mc_age = st.selectbox("Age Bracket", options=list(BRFSS_AGE_MAP.keys()), format_func=lambda x: BRFSS_AGE_MAP[x], key="mc_age")
            mc_bmi = st.number_input("BMI", min_value=12.0, max_value=75.0, value=29.0, key="mc_bmi")
            mc_bp = st.selectbox("High BP (0=No, 1=Yes)", [0.0, 1.0], key="mc_bp")
            mc_chol = st.selectbox("High Chol (0=No, 1=Yes)", [0.0, 1.0], key="mc_chol")
        with mc_c2:
            mc_phys = st.selectbox("Physical Activity (0=No, 1=Yes)", [1.0, 0.0], key="mc_phys")
            mc_smoker = st.selectbox("Smoker (0=No, 1=Yes)", [0.0, 1.0], key="mc_smk")
            mc_genhlth = st.selectbox("General Health (1=Exc, 5=Poor)", [1.0, 2.0, 3.0, 4.0, 5.0], index=2, key="mc_gh")
            mc_income = st.selectbox("Income Bracket", list(BRFSS_INCOME_MAP.keys()), format_func=lambda x: BRFSS_INCOME_MAP[x], index=4, key="mc_inc")

        submit_mc = st.form_submit_button("Submit 3-Class Assessment", use_container_width=True)

    if submit_mc:
        input_dict = {
            "HighBP": mc_bp, "HighChol": mc_chol, "CholCheck": 1.0, "BMI": float(mc_bmi),
            "Smoker": mc_smoker, "Stroke": 0.0, "HeartDiseaseorAttack": 0.0,
            "PhysActivity": mc_phys, "Fruits": 1.0, "Veggies": 1.0, "HvyAlcoholConsump": 0.0,
            "AnyHealthcare": 1.0, "NoDocbcCost": 0.0, "GenHlth": float(mc_genhlth),
            "MentHlth": 2.0, "PhysHlth": 2.0, "DiffWalk": 0.0, "Sex": mc_sex,
            "Age": float(mc_age), "Education": 4.0, "Income": float(mc_income)
        }
        with st.spinner("Computing 3-class probabilities..."):
            res_mc = service.predict_multiclass_012(input_dict)
            st.success(f"**Predicted Status:** {res_mc['predicted_class']}")
            st.write("#### Class Probabilities Breakdown:")
            st.json(res_mc["probabilities"])
            st.bar_chart(pd.Series(res_mc["probabilities"]))
