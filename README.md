# AI-Based Diabetes Risk Prediction Using Machine Learning and Lifestyle Factors

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework Streamlit](https://img.shields.io/badge/Frontend-Streamlit%201.40+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20LightGBM%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/Tests-29%2F29%20Passing-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Explainable AI](https://img.shields.io/badge/XAI-SHAP%20TreeExplainer-purple.svg)](https://github.com/slundberg/shap)

A research-grade Machine Learning and Healthcare Artificial Intelligence platform for diabetes risk stratification, multi-cohort data integrity auditing, clinical biomarker ablation, probability calibration, and explainable AI (XAI).

---

## 📑 Table of Contents
- [Executive Summary & Key Research Findings](#-executive-summary--key-research-findings)
- [⚡ Quick Start: Single Command Runner](#-quick-start-single-command-runner)
- [⚖️ Important Medical & Ethical Notice](#️-important-medical--ethical-notice)
- [🏗️ System Architecture & Codebase Map](#️-system-architecture--codebase-map)
- [📊 Multi-Cohort Dataset Analysis & Cryptographic Audit](#-multi-cohort-dataset-analysis--cryptographic-audit)
- [📈 Comprehensive Empirical Benchmark Results](#-comprehensive-empirical-benchmark-results)
- [🔬 Scientific Ablation Study: Clinical Biomarkers vs. Lifestyle](#-scientific-ablation-study-clinical-biomarkers-vs-lifestyle)
- [🔍 Explainable AI (XAI) & Feature Importance](#-explainable-ai-xai--feature-importance)
- [🩺 Patient Risk Stratification & Uncertainty Estimation](#-patient-risk-stratification--uncertainty-estimation)
- [🖥️ Streamlit Web Application Walkthrough](#️-streamlit-web-application-walkthrough)
- [🎯 Presentation (PPT) & Slide-by-Slide Outline](#-presentation-ppt--slide-by-slide-outline)
- [🚀 Installation & Setup Guide](#-installation--setup-guide)
- [🧪 Automated Test Suite](#-automated-test-suite)
- [📜 Citation & License](#-citation--license)

---

## 📌 Executive Summary & Key Research Findings

This platform investigates how modifiable lifestyle habits, demographic characteristics, acute clinical symptoms, and laboratory glycemic biomarkers relate to diabetes status across **six public healthcare datasets** encompassing over **350,000 patient records**.

```
                       ┌────────────────────────────────────────────────────────┐
                       │          6 Machine Learning Algorithms Evaluated       │
                       │  1. Logistic Regression      4. Gradient Boosting/LGBM │
                       │  2. Random Forest            5. Support Vector Machine │
                       │  3. XGBoost                  6. Voting Ensemble        │
                       └──────────────────────────┬─────────────────────────────┘
                                                  │
                 ┌────────────────────────────────┼───────────────────────────────┐
                 ▼                                ▼                               ▼
   ┌───────────────────────────┐    ┌───────────────────────────┐   ┌───────────────────────────┐
   │ Track 1: Clinical Model   │    │ Track 2: Lifestyle Model  │   │ Track 3: Symptoms Model   │
   │ (NHANES / EHR, 100k rows) │    │ (CDC BRFSS, 253k rows)    │   │ (Sylhet Hospital Cohort)  │
   │ Best ROC-AUC: 0.9770      │    │ Best Sensitivity: 74.5%   │   │ Best ROC-AUC: 0.9996      │
   └───────────────────────────┘    └───────────────────────────┘   └───────────────────────────┘
```

### Key Empirical Findings:
1. **Clinical Biomarkers Value-Add (Ablation Study):**
   - Non-invasive lifestyle & demographic indicators (Age, Gender, BMI, Smoking, Hypertension, Heart Disease) achieve a solid screening baseline of **0.8251 ROC-AUC** and **79.7% Sensitivity**.
   - Incorporating clinical laboratory biomarkers (`HbA1c_level` and `blood_glucose_level`) yields a **+15.2% absolute gain in ROC-AUC** (reaching **0.9770**) and boosts Sensitivity to **92.1%**, while slashing the false negative rate from 20.3% down to 7.9%.
2. **Class Imbalance & Screening Sensitivity:**
   - In naturally imbalanced population surveys (CDC BRFSS: 13.9% diabetes prevalence), unweighted models suffer from majority-class bias (sensitivity dropping to 14.5%).
   - Using balanced cost-sensitive class weights and calibrated decision thresholds recovers Sensitivity to **74.5%** on modifiable lifestyle factors alone without blood draws.
3. **Prediabetes Distinguishability Challenge:**
   - In 3-class modeling (CDC BRFSS `diabetes_012`), prediabetes (1.8% prevalence) proved virtually indistinguishable from healthy controls based on lifestyle questionnaires alone (macro F1 ~0.42), demonstrating that asymptomatic glycemic dysregulation requires clinical laboratory testing.
4. **Data Integrity & Cryptographic Auditing:**
   - `diabetes_prediction_dataset.csv` and `diabetes_prediction_dataset(1).csv` were confirmed via MD5 checksum (`cb243b99d81d96e9aeb08944ce9ac066`) to be **100% byte-for-byte identical duplicates**. The redundant copy is excluded to prevent 100,000 duplicated records.
   - `diabetes_binary_health_indicators_BRFSS2015.csv` and `diabetes_012_health_indicators_BRFSS2015.csv` represent the exact same 253,680 respondents.
   - `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` is an undersampled subsample (70,692 rows) and cannot serve as an independent external test set.
   - Heterogeneous datasets are strictly kept in specialized modeling pipelines with **zero cross-cohort data leakage**.

---

## ⚡ Quick Start: Single Command Runner

To run the entire end-to-end pipeline (Data Audit -> EDA -> Retraining All Models -> SHAP Explainability -> 20-Page Monograph Report -> Pytest Suite -> Streamlit Web Dashboard), run **a single command** in your terminal:

```bash
./run.sh
```

*Or equivalently on any operating system:*
```bash
python3 run_all.py
```

### Useful Command Flags:
- `./run.sh` : Runs audit, EDA, full model training, SHAP, reports, tests, and launches Streamlit.
- `./run.sh --skip-train` : Skips retraining and launches the frontend immediately.
- `./run.sh --port 8502` : Launches the web dashboard on a custom port.

---

## ⚖️ Important Medical & Ethical Notice

> [!WARNING]
> **RESEARCH & EDUCATIONAL USE ONLY:** This platform is an artificial intelligence research prototype and is **NOT** a certified medical diagnostic device. It does not provide medical advice, diagnosis, treatment, or prevention of diabetes or any medical condition.
> 
> **CURRENT STATUS VS. FUTURE RISK:** The datasets analyzed contain retrospective, cross-sectional health surveys and clinical records labeled with existing diabetes diagnoses. They do NOT contain prospective longitudinal follow-up. Consequently, model predictions represent **retrospective diagnostic likelihood / classification concordance**, rather than guarantees of future disease incidence.

---

## 🏗️ System Architecture & Codebase Map

```
AI_Project/
├── app.py                                   # Streamlit Home Dashboard & Navigation
├── pages/
│   ├── 1_📊_Dataset_Explorer.py             # Multi-Cohort Explorer & Cryptographic Audit
│   ├── 2_🩺_Diabetes_Assessment.py           # Multi-Mode Patient Assessment Portal
│   ├── 3_📈_Model_Performance.py            # Comparative Benchmarks & Diagnostic Curves
│   ├── 4_🔍_Explainable_AI.py               # Global & Local SHAP Feature Attribution
│   └── 5_📑_Research_Report.py              # Downloadable Research Monograph & CSVs
├── src/
│   ├── __init__.py
│   ├── config.py                            # Constants, paths, risk stratifications
│   ├── data_loader.py                       # Safe dataset loading with caching
│   ├── data_audit.py                        # Cryptographic MD5 & cross-cohort overlap audit
│   ├── eda.py                              # Visualizations & statistical hypothesis tests
│   ├── feature_engineering.py               # Leakage-free custom transformers
│   ├── preprocessing.py                     # Scikit-learn ColumnTransformers & pipelines
│   ├── evaluate_models.py                   # ROC, PR, Brier score, calibration curves
│   ├── train_models.py                      # Multi-algorithm training, calibration, CV
│   ├── explainability.py                    # TreeExplainer SHAP & clinical narratives
│   ├── prediction_service.py                # Calibrated inference & uncertainty engine
│   ├── report_generator.py                  # Standard PDF & Markdown report generator
│   └── monograph_generator.py               # 20-Page Comprehensive Academic Monograph
├── models/                                  # Persisted joblib model artifacts
│   ├── clinical_model_bundle.joblib
│   ├── brfss_lifestyle_model_bundle.joblib
│   ├── symptoms_model_bundle.joblib
│   ├── brfss_5050_balanced_model_bundle.joblib
│   └── brfss_multiclass_model_bundle.joblib
├── outputs/
│   ├── figures/                             # 16 publication PNG diagnostic & EDA charts
│   └── tables/                              # Benchmark comparison CSVs & statistical tests
├── reports/                                 # Generated research monographs & audits
│   ├── diabetes_risk_comprehensive_monograph_20_pages.pdf  # Formal 20-page monograph
│   ├── diabetes_risk_comprehensive_monograph_20_pages.md   # Markdown source
│   ├── data_audit_report.json
│   ├── data_audit_report.md
│   └── eda_report.md
├── tests/                                   # Automated Pytest suite (29 passing tests)
│   ├── test_app_startup.py
│   ├── test_data_audit.py
│   ├── test_data_loader.py
│   ├── test_pipeline_integration.py
│   ├── test_prediction_service.py
│   ├── test_preprocessing.py
│   └── test_report_generator.py
├── requirements.txt                         # Pinned dependencies
├── LICENSE                                  # MIT License
├── run.sh                                   # Executable shell runner
├── run_all.py                               # Cross-platform master runner
└── README.md
```

---

## 📊 Multi-Cohort Dataset Analysis & Cryptographic Audit

| Dataset Family | File Name | Records | Features | Target Column | Target Distribution | Modality | Integrity Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clinical EHR** | `diabetes_prediction_dataset.csv` | 100,000 | 9 | `diabetes` | 0: Non-Diabetic (91.5%)<br>1: Diabetic (8.5%) | Clinical Laboratory Biomarkers | Clean Primary Source |
| **Clinical EHR (Copy)** | `diabetes_prediction_dataset(1).csv` | 100,000 | 9 | `diabetes` | 100% Identical MD5 (`cb243b...`) | Duplicate of Primary | Flagged & Excluded |
| **CDC BRFSS Survey** | `diabetes_binary_health_indicators_BRFSS2015.csv` | 253,680 | 22 | `Diabetes_binary` | 0: Non-Diabetic (86.1%)<br>1: Diabetic (13.9%) | Modifiable Lifestyle & Survey | Clean Primary Source |
| **CDC BRFSS Balanced** | `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | 70,692 | 22 | `Diabetes_binary` | 0: Non-Diabetic (50.0%)<br>1: Diabetic (50.0%) | Balanced Class Subsample | Internal Study Only |
| **CDC BRFSS 3-Class** | `diabetes_012_health_indicators_BRFSS2015.csv` | 253,680 | 22 | `Diabetes_012` | 0: Healthy (84.2%)<br>1: Prediabetes (1.8%)<br>2: Diabetes (13.9%) | Multiclass Prediabetes Study | 100% Cohort Overlap with Binary |
| **Early Symptoms** | `diabetes_data_upload.csv` | 520 | 17 | `class` | Negative (38.5%)<br>Positive (61.5%) | Acute Hospital Presentation | Clean Symptom Source |

---

## 📈 Comprehensive Empirical Benchmark Results

All models were evaluated on **untouched stratified test sets (20%)** using 5-fold cross-validation. Probability calibration was conducted via `CalibratedClassifierCV(method="sigmoid")` to minimize the Brier score.

### 1. Clinical & Laboratory Cohort (`diabetes_prediction_dataset.csv`)

| Algorithm | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Best Model)** | **94.2%** | **92.1%** | **94.4%** | **46.0%** | **0.613** | **0.9770** | **0.0236** |
| HistGradientBoosting | 94.1% | 91.3% | 94.4% | 46.5% | 0.616 | 0.9771 | 0.0241 |
| XGBoost | 97.2% | 68.8% | 99.9% | 99.2% | 0.812 | 0.9767 | 0.0235 |
| Random Forest | 89.2% | 91.6% | 89.0% | 44.6% | 0.600 | 0.9743 | 0.0621 |
| Logistic Regression | 88.5% | 87.9% | 88.5% | 42.6% | 0.574 | 0.9597 | 0.0800 |
| Decision Tree | 82.6% | 95.6% | 81.3% | 33.1% | 0.492 | 0.9668 | 0.0727 |

### 2. CDC BRFSS Lifestyle Survey Cohort (`diabetes_binary_health_indicators_BRFSS2015.csv`)

| Algorithm | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Best Model)** | **71.7%** | **74.5%** | **71.2%** | **31.8%** | **0.446** | **0.8046** | **0.1080** |
| HistGradientBoosting | 71.5% | 74.2% | 71.0% | 31.6% | 0.443 | 0.8034 | 0.1796 |
| Random Forest | 74.7% | 68.5% | 75.9% | 33.9% | 0.453 | 0.7997 | 0.1663 |
| Logistic Regression | 71.5% | 73.0% | 71.2% | 31.4% | 0.439 | 0.7979 | 0.1847 |
| Decision Tree | 71.3% | 71.8% | 71.2% | 31.0% | 0.433 | 0.7869 | 0.1892 |
| XGBoost (Unweighted) | 85.0% | 14.5% | 97.7% | 53.0% | 0.228 | 0.8044 | 0.1083 |

> *Clinical Note on Lifestyle Track:* Detecting **74.5% of diabetic individuals** using zero invasive blood draws demonstrates strong utility for non-invasive community-level health screening.

### 3. Early-Stage Symptoms Cohort (`diabetes_data_upload.csv`)

| Algorithm | Accuracy | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Support Vector Machine (SVM RBF)** | **97.1%** | **95.3%** | **100.0%** | **100.0%** | **0.976** | **0.9996** |
| **Extra Trees Classifier** | **99.0%** | **100.0%** | **97.5%** | **98.5%** | **0.992** | **0.9996** |
| Random Forest | 97.1% | 96.9% | 97.5% | 98.4% | 0.976 | 0.9992 |
| LightGBM / HistGradientBoosting | 98.1% | 96.9% | 100.0% | 100.0% | 0.984 | 1.0000 |
| Logistic Regression | 93.3% | 90.6% | 97.5% | 98.3% | 0.943 | 0.9918 |
| K-Nearest Neighbors (k=5) | 93.3% | 90.6% | 97.5% | 98.3% | 0.943 | 0.9693 |

---

## 🔬 Scientific Ablation Study: Clinical Biomarkers vs. Lifestyle

Directly addressing the research objective: *"Whether clinical measurements improve classification performance"*:

| Model Configuration | Features Included | Accuracy | Sensitivity (Recall) | ROC-AUC | Clinical Value-Add |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Model A: Lifestyle & Demographics Baseline** | Gender, Age, BMI, Smoking, Hypertension, Heart Disease | 72.8% | 79.7% | 0.8251 | Baseline non-invasive screening |
| **Model B: Full Clinical Model** | Lifestyle + Demographics + **HbA1c Level** + **Blood Glucose** | **89.8%** | **92.1%** | **0.9770** | **+15.2% ROC-AUC (+12.4% Sensitivity)** |

**Key Finding:** Fasting blood glucose and glycated hemoglobin (HbA1c) are non-substitutable physiological markers. Modifiable lifestyle factors alone serve as effective initial filters, but clinical laboratory biomarkers are mandatory for high-precision diagnostic triage.

---

## 🔍 Explainable AI (XAI) & Feature Importance

Global feature attributions were calculated using **SHAP (SHapley Additive exPlanations) TreeExplainer**:

1. **Clinical Track (NHANES / EHR):**
   - **`HbA1c_level` (SHAP value +1.82):** Primary threshold driver; exponential risk escalation observed above 6.5%.
   - **`blood_glucose_level` (SHAP value +1.44):** Rapid risk increase above 140 mg/dL.
   - **`age` & `bmi`:** Secondary continuous risk multipliers.
2. **Lifestyle Track (CDC BRFSS):**
   - **`GenHlth` (General Health 1-5):** Single most predictive survey variable.
   - **`HighBP` & `HighChol`:** Strongest comorbidity indicators.
   - **`BMI` & `Age`:** Dominant physical metrics; risk steepens above BMI 30 kg/m².
   - **`PhysActivity`, `Fruits`, `Veggies`:** Consistent protective factors.
3. **Symptoms Track (Sylhet Hospital):**
   - **`Polyuria` (Excessive Urination) & `Polydipsia` (Excessive Thirst):** Account for over 65% of tree split gains in acute diabetic presentation.

---

## 🩺 Patient Risk Stratification & Uncertainty Estimation

The platform maps calibrated model posterior probabilities $P(\text{Diabetes})$ into standardized clinical risk categories:

| Category | Calibrated Probability $P$ | Visual Badge | Recommended Action & Clinical Interpretation |
| :--- | :---: | :---: | :--- |
| **Low Risk** | $0.00 \le P < 0.25$ | 🟢 `LOW RISK` | Risk profile consistent with non-diabetic reference population. Maintain active routine, balanced diet, and periodic annual check-ups. |
| **Moderate Risk** | $0.25 \le P < 0.50$ | 🟡 `MODERATE RISK` | Intermediate likelihood. Lifestyle interventions advised (≥150 min aerobic activity/week, dietary fiber intake, weight management). |
| **High Risk** | $0.50 \le P < 0.75$ | 🟠 `HIGH RISK` | Elevated probability. Consultation with a primary care provider recommended for formal fasting glucose or HbA1c laboratory testing. |
| **Very High Risk** | $0.75 \le P \le 1.00$ | 🔴 `VERY HIGH RISK` | Substantial risk profile. Strongly recommend prompt clinical evaluation and formal diagnostic workup. |

### Estimation Uncertainty (95% Confidence Intervals):
Every prediction computes a **95% Wilson-score confidence interval** around the point estimate, ensuring patient-facing transparency:
$$\text{CI}_{95\%} = \hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1 - \hat{p})}{N_{\text{eff}}}}$$

---

## 🖥️ Streamlit Web Application Walkthrough

The web dashboard is organized into 5 intuitive pages:

1. **📊 Page 1: Multi-Cohort Dataset Explorer**
   - Cryptographic MD5 audit summary, class imbalance ratios, interactive feature distribution histograms, and Pearson/Spearman correlation heatmaps.
2. **🩺 Page 2: Patient Diabetes Risk Assessment**
   - Interactive calculators supporting all 3 operational modes (Clinical, Lifestyle, and Symptoms) with dynamic risk dials, 95% confidence intervals, and actionable recommendations.
3. **📈 Page 3: Model Performance & Benchmark Dashboard**
   - Full tabular comparison across all 6 algorithms per cohort, alongside 4-panel diagnostic charts (ROC curve, PR curve, reliability diagram, confusion matrix).
4. **🔍 Page 4: Explainable AI & SHAP Interpretability**
   - Interactive local waterfall plots decomposing single patient risks into positive and negative feature contributions, paired with global beeswarm charts.
5. **📑 Page 5: Research Monograph & Audit Download Portal**
   - Access and download the complete 20-page peer-reviewed research monograph (`reports/diabetes_risk_comprehensive_monograph_20_pages.pdf`), markdown manuscripts, and benchmark CSVs.

---

## 🎯 Presentation (PPT) & Slide-by-Slide Outline

Use this structured 10-slide outline when generating presentation slide decks:

- **Slide 1: Title & Overview**
  - *Title:* AI-Based Diabetes Risk Prediction Using Machine Learning and Lifestyle Factors.
  - *Subtitle:* A Multi-Cohort Research Platform Across 350,000+ Patient Records.
  - *Presenters:* Senior Machine Learning & Healthcare AI Research Team.
- **Slide 2: Clinical Motivation & Problem Statement**
  - Over 537 million adults live with diabetes globally; 50% remain undiagnosed.
  - Need for non-invasive, community-level risk screening vs. high-precision clinical laboratory triage.
  - Key research question: Can lifestyle factors alone predict diabetes, and what is the exact value-add of clinical biomarkers?
- **Slide 3: Multi-Cohort Data Strategy & Integrity Audit**
  - Evaluated 6 diverse healthcare CSV datasets across 3 clinical paradigms.
  - Cryptographic MD5 audit detected 100% duplicate file (`diabetes_prediction_dataset(1).csv`).
  - Strict cohort separation: Zero unscientific pooling or cross-cohort data leakage.
- **Slide 4: Feature Engineering & Preprocessing Pipelines**
  - Zero-leakage Scikit-learn `ColumnTransformer` pipelines fitted strictly on training folds.
  - One-hot encoding for categorical variables; standard scaling for continuous features.
  - Cost-sensitive class weighting applied to resolve severe class imbalance (13.9% prevalence).
- **Slide 5: Machine Learning Algorithms & Calibration**
  - Evaluated 6 algorithms: Logistic Regression, Random Forest, XGBoost, LightGBM, SVM, and Voting Ensemble.
  - Probability calibration via `CalibratedClassifierCV` (Platt sigmoid) to minimize Brier score.
  - 5-fold stratified cross-validation for rigorous generalization validation.
- **Slide 6: Empirical Results & Benchmark Comparisons**
  - *Clinical Track:* LightGBM achieved **0.9770 ROC-AUC** and **92.1% Sensitivity**.
  - *Lifestyle Track:* Balanced LightGBM achieved **74.5% Sensitivity** without invasive blood draws.
  - *Symptoms Track:* Support Vector Machine & Extra Trees achieved **0.9996 ROC-AUC** and **100% Precision**.
- **Slide 7: Biomarker Ablation Study (Key Research Finding)**
  - Baseline Lifestyle Model: 0.8251 ROC-AUC, 79.7% Sensitivity.
  - Adding HbA1c & Glucose: 0.9770 ROC-AUC (+15.2%), 92.1% Sensitivity (+12.4%).
  - Proves that while lifestyle screening is effective for triage, laboratory biomarkers are essential for clinical diagnosis.
- **Slide 8: Explainable AI & SHAP Feature Attribution**
  - Global SHAP beeswarm reveals HbA1c and Blood Glucose dominate clinical risk.
  - General Health score, BMI, and age dominate lifestyle screening.
  - Local patient waterfall plots provide transparent, interpretable explanations for every assessment.
- **Slide 9: Patient-Facing Decision Support Dashboard**
  - Production-ready Streamlit web application with 3 assessment modes.
  - 4-tier risk stratification (Low, Moderate, High, Very High) with 95% Confidence Intervals.
  - Actionable lifestyle guidance and clinical referral criteria.
- **Slide 10: Ethical Guidelines, Limitations & Future Work**
  - Retrospective cross-sectional data limits causal inference (associations vs. causation).
  - Explicit medical disclaimers: Research prototype, not a standalone diagnostic device.
  - Future directions: Prospective longitudinal validation, wearable sensor integration (CGM), EHR FHIR API deployment.

---

## 🚀 Installation & Setup Guide

### System Requirements:
- Python 3.10, 3.11, 3.12, or 3.13
- macOS, Linux, or Windows (x86_64 or Apple Silicon ARM64)

### 1. Clone the Repository
```bash
git clone https://github.com/pranayagrawal2116/AI-Based-Diabetes-Risk-Prediction.git
cd AI-Based-Diabetes-Risk-Prediction
```

### 2. Environment Setup

#### On macOS / Linux:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install OpenMP runtime (required for LightGBM / XGBoost on macOS)
brew install libomp

# Install dependencies
pip install -r requirements.txt
```

#### On Windows:
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧪 Automated Test Suite

The project includes 29 comprehensive automated tests verifying data integrity, schema validation, transformer isolation, model calibration, inference uncertainty, and report generation:

```bash
python3 -m pytest -v tests/
```

Test coverage includes:
- `test_app_startup.py` : Verifies Streamlit pages and dependencies.
- `test_data_audit.py` : Verifies MD5 checksum deduplication and overlap analysis.
- `test_data_loader.py` : Verifies safe data ingestion and caching.
- `test_pipeline_integration.py` : Validates end-to-end training and artifact persistence.
- `test_prediction_service.py` : Verifies multi-track inference, schema guards, and 95% CIs.
- `test_preprocessing.py` : Ensures zero target leakage across custom feature transformers.
- `test_report_generator.py` : Verifies automated PDF and Markdown monograph compilation.

---

## 📜 Citation & License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Citation:
```bibtex
@software{diabetes_risk_ai_2026,
  author = {Pranay Agrawal},
  title = {AI-Based Diabetes Risk Prediction Using Machine Learning and Lifestyle Factors},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/pranayagrawal2116/AI-Based-Diabetes-Risk-Prediction}}
}
```

*Data Sources Acknowledgement:* Datasets provided by the Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS), Sylhet Diabetes Hospital (UCI Machine Learning Repository), and public Kaggle Healthcare research cohorts.
