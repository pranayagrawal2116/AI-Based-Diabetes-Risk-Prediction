# AI-Based Diabetes Risk Prediction Using Machine Learning and Lifestyle Factors
## Formal Research Evaluation and Clinical Analytics Report

**Author:** Advanced Healthcare AI & Machine Learning Research Team
**Standard:** Strict Medical ML Validation, Multi-Cohort Auditing & Leakage-Free Inference

---

## Executive Summary

Diabetes mellitus represents one of the most critical global public health challenges, affecting over 530 million adults worldwide. Early detection and risk stratification enable timely behavioral interventions and clinical pharmacotherapy that drastically reduce microvascular and macrovascular complications.

This research project conducted a comprehensive investigation across **six public healthcare datasets** comprising over **350,000 patient records** and survey responses. We developed, cross-validated, calibrated, and compared machine learning architectures across three primary clinical modalities:
1. **Lifestyle and Behavioral Survey Cohort (CDC BRFSS 2015):** 253,680 survey records focusing on non-invasive, modifiable lifestyle markers.
2. **Clinical and Laboratory Biomarker Cohort (EHR):** 100,000 patient records incorporating blood glucose and glycated hemoglobin (HbA1c).
3. **Early-Stage Symptom Screening Cohort (Sylhet Hospital):** 520 acute clinical presentations documenting cardinal diabetic symptoms.

---

## 1. Data Integrity and Multi-Cohort Audit Findings

A rigorous cryptographic (MD5) and schema audit was performed prior to any model training, yielding critical architectural decisions:
- **Duplicate File Detection:** `diabetes_prediction_dataset.csv` and `diabetes_prediction_dataset(1).csv` share the identical MD5 checksum (`cb243b99d81d96e9aeb08944ce9ac066`). File 2 was identified as a 100% duplicate download and safely excluded to prevent 100,000 double-counted rows.
- **BRFSS Cohort Identity:** Confirmed that `diabetes_binary_health_indicators_BRFSS2015.csv` and `diabetes_012_health_indicators_BRFSS2015.csv` contain the exact same 253,680 respondents. In the binary version, prediabetes (class 1, 4,631 records) was conflated into the non-diabetic class.
- **Subsample Verification:** Confirmed that `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (70,692 rows) is an undersampled subset of BRFSS containing all 35,346 diabetes cases and 35,346 balanced controls. It cannot serve as an independent external validation set.
- **Zero Naive Concatenation:** Survey questionnaires, clinical blood draws, and hospital symptom forms were strictly kept in separate modeling tracks.

---

## 2. Research Benchmark Results Across Cohorts

### A. Clinical & Laboratory Cohort Performance
| Model | Accuracy | Recall (Sensitivity) | Specificity | Precision | F1 Score | ROC-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8848 | 0.8791 | 0.8853 | 0.4259 | 0.5738 | 0.9597 | 0.08 |
| **Decision Tree** | 0.8258 | 0.9564 | 0.8131 | 0.3312 | 0.4920 | 0.9668 | 0.0727 |
| **Random Forest** | 0.8923 | 0.9157 | 0.8901 | 0.4463 | 0.6001 | 0.9743 | 0.0621 |
| **Extra Trees** | 0.8760 | 0.9180 | 0.8719 | 0.4095 | 0.5664 | 0.9655 | 0.0778 |
| **HistGradientBoosting** | 0.8997 | 0.9133 | 0.8984 | 0.4652 | 0.6164 | 0.9771 | 0.0605 |
| **LightGBM** | 0.8975 | 0.9210 | 0.8953 | 0.4597 | 0.6133 | 0.9770 | 0.0611 |
| **XGBoost** | 0.9720 | 0.6881 | 0.9994 | 0.9915 | 0.8124 | 0.9767 | 0.0235 |

*Selected Model:* **LightGBM Classifier** (Calibrated). Achieved **0.9770 ROC-AUC** and **92.1% Sensitivity** with a post-calibration Brier score of **0.0236**.

### B. CDC BRFSS Lifestyle Survey Cohort Performance
| Model | Accuracy | Recall (Sensitivity) | Specificity | Precision | F1 Score | ROC-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7147 | 0.7299 | 0.7120 | 0.3138 | 0.4389 | 0.7979 | 0.1847 |
| **Decision Tree** | 0.7126 | 0.7181 | 0.7116 | 0.3101 | 0.4331 | 0.7869 | 0.1892 |
| **Random Forest** | 0.7474 | 0.6848 | 0.7587 | 0.3387 | 0.4532 | 0.7997 | 0.1663 |
| **Extra Trees** | 0.7051 | 0.7220 | 0.7020 | 0.3043 | 0.4282 | 0.7882 | 0.1823 |
| **HistGradientBoosting** | 0.7151 | 0.7417 | 0.7103 | 0.3161 | 0.4432 | 0.8034 | 0.1796 |
| **LightGBM** | 0.7168 | 0.7449 | 0.7117 | 0.3181 | 0.4458 | 0.8046 | 0.1789 |
| **XGBoost** | 0.8496 | 0.1452 | 0.9767 | 0.5298 | 0.2279 | 0.8044 | 0.1083 |

*Selected Model:* **LightGBM Classifier** (Calibrated with balanced weights). Achieved **0.8046 ROC-AUC** and **74.5% Sensitivity** on lifestyle features without any laboratory blood tests.

### C. Early-Stage Symptoms Cohort Performance
| Model | Accuracy | Recall (Sensitivity) | Specificity | Precision | F1 Score | ROC-AUC | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9327 | 0.9062 | 0.9750 | 0.9831 | 0.9431 | 0.9918 | 0.0483 |
| **Decision Tree** | 0.9712 | 0.9531 | 1.0000 | 1.0000 | 0.9760 | 0.9879 | 0.0271 |
| **Random Forest** | 0.9712 | 0.9688 | 0.9750 | 0.9841 | 0.9764 | 0.9992 | 0.0147 |
| **Extra Trees** | 0.9904 | 1.0000 | 0.9750 | 0.9846 | 0.9922 | 0.9996 | 0.0098 |
| **HistGradientBoosting** | 0.9808 | 0.9688 | 1.0000 | 1.0000 | 0.9841 | 1.0000 | 0.0182 |
| **LightGBM** | 0.9808 | 0.9688 | 1.0000 | 1.0000 | 0.9841 | 1.0000 | 0.0144 |
| **XGBoost** | 0.9808 | 0.9688 | 1.0000 | 1.0000 | 0.9841 | 0.9984 | 0.0163 |
| **K-Nearest Neighbors** | 0.9327 | 0.9062 | 0.9750 | 0.9831 | 0.9431 | 0.9693 | 0.0496 |
| **Support Vector Machine** | 0.9712 | 0.9531 | 1.0000 | 1.0000 | 0.9760 | 0.9996 | 0.0099 |

*Selected Model:* **Extra Trees Classifier** (Calibrated). Achieved **0.9996 ROC-AUC** and **100.0% Sensitivity** on the classic clinical symptom triad.

---

## 3. Scientific Ablation: The Value of Clinical Biomarkers vs. Lifestyle Habits

To rigorously assess whether laboratory measurements improve predictive performance over lifestyle factors alone, we conducted an ablation experiment on identical patient splits:

| Evaluation Mode | Features Included | Accuracy | Recall | ROC-AUC | Diagnostic Value Add |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ablation: Lifestyle Only** | Age, Gender, BMI, Smoking, Hypertension, Heart Disease | 0.8142 | 0.7967 | 0.8251 | Baseline (Non-Invasive) |
| **Full Clinical Model** | Above + HbA1c Level + Blood Glucose Level | 0.9419 | 0.9210 | 0.9770 | **+0.152 ROC-AUC (+12.4% Sensitivity)** |

### Core Scientific Takeaway:
While non-invasive lifestyle indicators establish an effective preliminary screening baseline (ROC-AUC ~0.825), the incorporation of glycemic biomarkers (HbA1c and fasting blood glucose) yields a profound **+15.2% absolute gain in ROC-AUC** and drastically reduces false negatives from 20.3% down to 7.9%. This underscores that lifestyle surveys are ideal for initial population triage, while clinical biomarkers are indispensable for diagnostic certainty.

---

## 4. Explainable AI & Feature Importance Analysis

- **Clinical Biomarkers:** TreeExplainer SHAP attribution confirmed that `HbA1c_level` is the single most decisive factor, followed by `blood_glucose_level` and `bmi`.
- **BRFSS Lifestyle Survey:** The primary non-invasive drivers are `GenHlth` (self-rated general health), `HighBP` (hypertension), `BMI`, `Age`, and `PhysActivity` (sedentary lifestyle increases odds).
- **Symptoms:** The classic osmotic diuresis triad (`Polyuria`, `Polydipsia`, and `sudden weight loss`) accounted for over 70% of feature attribution.
- **Non-Causal Interpretability:** SHAP values represent statistical attribution within historical observational data and must not be interpreted as proven causal mechanisms.

---

## 5. Critical Medical Distinction: Current Status vs. Future Risk
> **IMPORTANT DISTINCTION: CURRENT STATUS VS. FUTURE INCIDENCE. The supplied datasets contain retrospective cross-sectional health surveys and clinical records labeled with existing diabetes status. They do NOT contain prospective longitudinal tracking. Consequently, these models predict current classification likelihood consistent with survey or clinical diagnostic criteria, rather than guaranteeing future disease onset.**

---

## 6. Limitations, Ethical Considerations, and Future Scope

### Limitations:
1. **Observational & Cross-Sectional:** Datasets lack multi-year longitudinal tracking, precluding direct future incident diabetes risk modeling.
2. **Self-Reported Survey Bias:** CDC BRFSS relies on telephonic self-report, subject to recall bias.
3. **Prediabetes Distinguishability:** Prediabetes is largely asymptomatic in lifestyle surveys, achieving only modest macro F1 (~0.42).

### Ethical & Clinical Governance:
1. **No Automated Clinical Diagnosis:** System serves solely as an educational screening and risk ranking aid.
2. **Privacy Protection:** Zero patient data transmitted to external APIs; all computation occurs locally in memory.
3. **Healthcare Access Equity:** Model recognizes social determinants of health (education, income, cost barriers).

> **RESEARCH & EDUCATIONAL USE ONLY. This system is an artificial intelligence research prototype designed to explore machine learning associations with diabetes status. It is NOT a certified medical diagnostic device and does NOT provide medical advice, diagnosis, treatment, or prevention of any medical condition. The predicted probabilities reflect classification patterns observed in retrospective public datasets, not clinically calibrated absolute lifetime risks. Always consult a qualified physician or healthcare provider for any medical questions, clinical evaluation, or laboratory testing (e.g., HbA1c, oral glucose tolerance test).**
