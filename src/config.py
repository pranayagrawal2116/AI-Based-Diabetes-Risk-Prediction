"""
Configuration module for AI-Based Diabetes Risk Prediction.
Defines paths, dataset metadata, feature sets, risk stratifications, and clinical disclaimers.
"""

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Ensure directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Dataset Definitions
DATASET_FILES = {
    "diabetes_prediction_primary": "diabetes_prediction_dataset.csv",
    "diabetes_prediction_copy": "diabetes_prediction_dataset(1).csv",
    "diabetes_symptoms": "diabetes_data_upload.csv",
    "brfss_binary": "diabetes_binary_health_indicators_BRFSS2015.csv",
    "brfss_5050": "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
    "brfss_012": "diabetes_012_health_indicators_BRFSS2015.csv",
}

# Dataset Family Identifiers
FAMILY_CLINICAL = "clinical_measurements"
FAMILY_LIFESTYLE = "brfss_lifestyle"
FAMILY_LIFESTYLE_BALANCED = "brfss_5050"
FAMILY_SYMPTOMS = "early_symptoms"
FAMILY_MULTICLASS = "brfss_multiclass"

# Human-Readable Metadata
DATASET_METADATA = {
    FAMILY_CLINICAL: {
        "title": "Clinical & Laboratory Measurement Cohort",
        "primary_file": "diabetes_prediction_dataset.csv",
        "duplicate_file": "diabetes_prediction_dataset(1).csv",
        "description": "Patient records containing clinical biomarkers (HbA1c, blood glucose), medical history (hypertension, heart disease), demographics (age, gender), and lifestyle markers (BMI, smoking history).",
        "target_col": "diabetes",
        "target_labels": {0: "Non-Diabetic", 1: "Diabetic"},
        "type": "Binary Classification (Laboratory & Clinical)",
        "source": "Kaggle Healthcare Cohort / Electronic Health Records",
    },
    FAMILY_LIFESTYLE: {
        "title": "CDC BRFSS Survey (Binary Health & Lifestyle Indicators)",
        "primary_file": "diabetes_binary_health_indicators_BRFSS2015.csv",
        "description": "253,680 behavioral survey responses from CDC's Behavioral Risk Factor Surveillance System (2015). Focuses on lifestyle habits (physical activity, fruit/vegetable intake, alcohol, smoking) and self-reported health.",
        "target_col": "Diabetes_binary",
        "target_labels": {0.0: "Non-Diabetic / Prediabetic", 1.0: "Diabetic"},
        "type": "Binary Classification (Survey & Lifestyle)",
        "source": "CDC Behavioral Risk Factor Surveillance System (BRFSS 2015)",
    },
    FAMILY_LIFESTYLE_BALANCED: {
        "title": "CDC BRFSS 50-50 Balanced Cohort",
        "primary_file": "diabetes_binary_5050split_health_indicators_BRFSS2015.csv",
        "description": "Artificially balanced 70,692-row subset of BRFSS 2015 with equal positive and negative cases (35,346 each), used for comparative class-imbalance research.",
        "target_col": "Diabetes_binary",
        "target_labels": {0.0: "Non-Diabetic", 1.0: "Diabetic"},
        "type": "Balanced Binary Classification",
        "source": "CDC BRFSS 2015 Undersampled",
    },
    FAMILY_SYMPTOMS: {
        "title": "Early-Stage Symptom Screening Cohort",
        "primary_file": "diabetes_data_upload.csv",
        "description": "520 clinical interview records of newly diagnosed or suspected diabetic patients from Sylhet Diabetes Hospital, documenting early cardinal symptoms.",
        "target_col": "class",
        "target_labels": {"Negative": "Low Risk / Negative", "Positive": "Diabetic / High Risk"},
        "type": "Symptom-Based Binary Classification",
        "source": "Sylhet Diabetes Hospital (UCI Machine Learning Repository)",
    },
    FAMILY_MULTICLASS: {
        "title": "CDC BRFSS 3-Class Cohort (Prediabetes vs Diabetes)",
        "primary_file": "diabetes_012_health_indicators_BRFSS2015.csv",
        "description": "Full 253,680 BRFSS cohort preserving the distinction between healthy individuals, prediabetic status, and diagnosed diabetes.",
        "target_col": "Diabetes_012",
        "target_labels": {0.0: "No Diabetes", 1.0: "Prediabetes", 2.0: "Diabetes"},
        "type": "Multiclass Classification (3 Classes)",
        "source": "CDC BRFSS 2015",
    },
}

# CDC BRFSS Codebook Mappings
BRFSS_AGE_MAP = {
    1: "Age 18 to 24",
    2: "Age 25 to 29",
    3: "Age 30 to 34",
    4: "Age 35 to 39",
    5: "Age 40 to 44",
    6: "Age 45 to 49",
    7: "Age 50 to 54",
    8: "Age 55 to 59",
    9: "Age 60 to 64",
    10: "Age 65 to 69",
    11: "Age 70 to 74",
    12: "Age 75 to 79",
    13: "Age 80 or older",
}

BRFSS_GENHLTH_MAP = {
    1: "Excellent",
    2: "Very Good",
    3: "Good",
    4: "Fair",
    5: "Poor",
}

BRFSS_EDUCATION_MAP = {
    1: "Never attended school or kindergarten only",
    2: "Elementary (Grades 1-8)",
    3: "Some high school (Grades 9-11)",
    4: "High school graduate / GED",
    5: "Some college or technical school (1-3 yrs)",
    6: "College graduate (4+ years)",
}

BRFSS_INCOME_MAP = {
    1: "Less than $10,000",
    2: "$10,000 to <$15,000",
    3: "$15,000 to <$20,000",
    4: "$20,000 to <$25,000",
    5: "$25,000 to <$35,000",
    6: "$35,000 to <$50,000",
    7: "$50,000 to <$75,000",
    8: "$75,000 or more",
}

# Risk Band Definitions (Model Probability Stratification)
# Note: These are model probability intervals designed for triage and research ranking,
# not clinically validated absolute 10-year incidence probabilities.
RISK_CATEGORIES = {
    "Low": {
        "range": (0.0, 0.25),
        "color": "#10B981",  # Green
        "badge": "LOW ESTIMATED RISK",
        "description": "Model-estimated probability is below 25%. Maintain routine healthy lifestyle and regular check-ups.",
    },
    "Moderate": {
        "range": (0.25, 0.50),
        "color": "#F59E0B",  # Amber
        "badge": "MODERATE ESTIMATED RISK",
        "description": "Model-estimated probability is between 25% and 50%. Suggests lifestyle optimization and periodic screening.",
    },
    "High": {
        "range": (0.50, 0.75),
        "color": "#EF4444",  # Red
        "badge": "HIGH ESTIMATED RISK",
        "description": "Model-estimated probability is between 50% and 75%. Medical consultation and confirmatory testing recommended.",
    },
    "Very High": {
        "range": (0.75, 1.00),
        "color": "#7F1D1D",  # Dark Red
        "badge": "VERY HIGH ESTIMATED RISK",
        "description": "Model-estimated probability exceeds 75%. Strongly recommend prompt clinical evaluation with HbA1c or fasting glucose.",
    },
}

# Clinical and Ethical Disclaimers
CLINICAL_DISCLAIMER = (
    "RESEARCH & EDUCATIONAL USE ONLY. This system is an artificial intelligence research prototype "
    "designed to explore machine learning associations with diabetes status. It is NOT a certified medical diagnostic "
    "device and does NOT provide medical advice, diagnosis, treatment, or prevention of any medical condition. "
    "The predicted probabilities reflect classification patterns observed in retrospective public datasets, "
    "not clinically calibrated absolute lifetime risks. Always consult a qualified physician or healthcare provider "
    "for any medical questions, clinical evaluation, or laboratory testing (e.g., HbA1c, oral glucose tolerance test)."
)

CURRENT_STATUS_VS_FUTURE_RISK_NOTE = (
    "IMPORTANT DISTINCTION: CURRENT STATUS VS. FUTURE INCIDENCE. The supplied datasets contain retrospective "
    "cross-sectional health surveys and clinical records labeled with existing diabetes status. They do NOT contain "
    "prospective longitudinal tracking. Consequently, these models predict current classification likelihood "
    "consistent with survey or clinical diagnostic criteria, rather than guaranteeing future disease onset."
)
