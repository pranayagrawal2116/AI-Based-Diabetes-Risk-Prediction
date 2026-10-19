# AI-Based Diabetes Risk Prediction Using Machine Learning and Lifestyle Factors
## Comprehensive 20-Page Academic Research Monograph & Technical Evaluation

**Author:** Advanced Healthcare AI & Machine Learning Research Team  
**Standard:** Strict Medical ML Validation, Multi-Cohort Auditing & Leakage-Free Inference  
**Document Format:** 20 Dedicated Academic Chapters (Matching PDF Monograph)  

---

## Frontispiece: Regulatory and Ethical Notice
> **IMPORTANT NOTICE:** This platform is an artificial intelligence research monograph and educational screening prototype. It is NOT a certified medical diagnostic device under FDA/CE guidelines. The models classify retrospective risk consistent with historical observational cohorts and do NOT establish prospective future incidence. Clinical consultation, laboratory blood draws, and professional diagnosis remain mandatory.

---

## Chapter 1: Executive Summary & Scientific Abstract
**Abstract:** Diabetes screening faces an intractable trade-off between invasive diagnostic precision and community accessibility. In this study, we engineered, cross-validated, calibrated, and interpreted machine learning architectures across three disparate healthcare data modalities: (1) 100,000 clinical electronic health records with glycemic biomarkers, (2) 253,680 CDC Behavioral Risk Factor Surveillance System (BRFSS) survey responses with modifiable lifestyle indicators, and (3) 520 acute clinical hospital presentations documenting hallmark osmotic symptoms. Prior to model induction, a rigorous cryptographic MD5 audit identified a 100% duplicate file download, preventing 100,000 double-counted records. Models were strictly evaluated on untouched holdout partitions using clinical screening metrics (Sensitivity, Specificity, ROC-AUC, Brier score).

### Summary of Primary Research Discoveries:
1. **The Biomarker Premium:** Laboratory biomarkers (HbA1c & fasting glucose) provide a decisive +0.152 boost in ROC-AUC (reaching 0.9770) over lifestyle factors alone, slashing false negatives by 61%.
2. **Lifestyle Feasibility:** Modifiable survey factors (physical activity, fruit/veg intake, smoking, general health perception) achieve 0.8046 ROC-AUC and 74.5% Sensitivity under balanced cost-sensitive learning.
3. **Symptom Screening Precision:** Classic osmotic diuresis triad (polyuria, polydipsia, weight loss) separates diabetic states with 0.9996 ROC-AUC and 100% Sensitivity in acute triage.
4. **Prediabetes Detection Bottleneck:** Prediabetic metabolic dysregulation is virtually indistinguishable from healthy controls using lifestyle surveys alone (macro F1: 0.42), proving laboratory testing is clinically mandatory.

| Clinical Cohort Track | Sample (N) | Features | Optimal Classifier | Test AUC | Sensitivity | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clinical Laboratory EHR** | 100,000 | 9 | LightGBM (Calibrated) | 0.9770 | 92.1% | 0.0236 |
| **CDC BRFSS Lifestyle Survey** | 253,680 | 21 | LightGBM (Balanced) | 0.8046 | 74.5% | 0.1080 |
| **Early Symptoms (Sylhet)** | 520 | 16 | Extra Trees (Calibrated) | 0.9996 | 100.0% | 0.0074 |
| **BRFSS Balanced (50-50)** | 70,692 | 21 | XGBoost (Calibrated) | 0.8192 | 80.2% | 0.1726 |

---

## Chapter 2: Clinical Background & The Global Diabetes Crisis
Diabetes mellitus has emerged as one of the defining global health emergencies of modern civilization. According to the International Diabetes Federation (IDF), more than 537 million individuals aged 20-79 were living with diabetes in 2021, a figure projected to surge to 783 million by 2045. Approximately 90-95% of all cases represent Type 2 Diabetes Mellitus (T2DM), characterized by chronic systemic insulin resistance coupled with progressive pancreatic beta-cell secretory exhaustion.

The development of T2DM spans a multi-year asymptomatic continuum. Excess visceral adiposity stimulates the chronic release of free fatty acids and inflammatory cytokines, precipitating peripheral insulin resistance in skeletal muscle and hepatic tissue. Over years, glucotoxicity and beta-cell decline escalate fasting and postprandial hyperglycemia. Current clinical guidelines mandate invasive laboratory testing for definitive diagnosis (HbA1c ≥ 6.5%, FPG ≥ 126 mg/dL). While highly specific, these modalities require overnight fasting, cold-chain phlebotomy logistics, and substantial laboratory expenditures. Consequently, machine learning models leveraging non-invasive lifestyle surveys and electronic medical histories offer an unprecedented opportunity for low-cost, scalable triage.

---

## Chapter 3: Multi-Cohort Landscape & Cryptographic Integrity Audit
Prior to feature engineering, a cryptographic audit using MD5 hashing and relational cross-referencing was executed across all six supplied CSV files:
- `diabetes_prediction_dataset.csv` (100,000 rows, MD5 `cb243b99d81d96e9aeb08944ce9ac066`): Retained as primary clinical cohort.
- `diabetes_prediction_dataset(1).csv` (100,000 rows, MD5 `cb243b99d81d96e9aeb08944ce9ac066`): Proven 100% byte duplicate; safely excluded.
- `diabetes_binary_health_indicators_BRFSS2015.csv` (253,680 rows, MD5 `8f90e44047182c5f32d10f0c4dd4890f`): Retained as primary lifestyle cohort (13.9% prevalence).
- `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` (70,692 rows): Derived balanced subset containing all 35,346 diabetes cases; used for imbalance analysis.
- `diabetes_012_health_indicators_BRFSS2015.csv` (253,680 rows): Cohort identity match with BRFSS binary; preserves prediabetes (class 1).
- `diabetes_data_upload.csv` (520 rows): Acute hospital symptoms cohort from Sylhet Diabetes Hospital.

**The Strict Non-Concatenation Doctrine:** Never concatenate heterogeneous healthcare cohorts into a single table. Survey responses, clinical blood draws, and acute hospital symptoms reflect fundamentally different sampling methodologies and clinical modalities.

---

## Chapter 4: Exploratory Analysis: Clinical & Laboratory Cohort
In the 100,000-record clinical dataset, diabetic patients exhibit marked shifts across key biomarkers:
- **HbA1c Level:** Diabetic median 6.6% (IQR: 6.5 - 7.5%) vs. Non-Diabetic median 5.8% (IQR: 5.0 - 6.2%), Mann-Whitney U = 9.87e7, $p < 1e-300$.
- **Blood Glucose:** Diabetic median 160.0 mg/dL vs. Non-Diabetic median 130.0 mg/dL, Mann-Whitney U = 1.05e8, $p < 1e-300$.
- **Body Mass Index:** Diabetic median 31.9 kg/m² vs. Non-Diabetic median 27.1 kg/m².
- **Age:** Diabetic median 61.0 years vs. Non-Diabetic median 40.0 years.

---

## Chapter 5: Exploratory Analysis: CDC BRFSS Lifestyle Survey
Across 253,680 respondents in the CDC BRFSS lifestyle cohort:
- **Physical Activity:** Routine physical activity lowers diabetes prevalence from 23.3% down to 11.0% (RR = 0.47, $\chi^2 = 5840$).
- **General Health Perception (`GenHlth`):** Produced the single strongest correlation in survey data ($V = 0.301$). Prevalence in excellent health is 2.1% vs. 38.4% in poor health.
- **Dietary Indicators:** Daily vegetable consumption reduces prevalence from 17.8% to 13.0%; daily fruit consumption reduces prevalence from 16.2% to 12.6%.
- **Smoking:** Ever-smokers exhibit 17.1% prevalence vs 11.4% in non-smokers.

---

## Chapter 6: Exploratory Analysis: Early-Stage Acute Symptoms
In the Sylhet Hospital cohort (520 presentations):
- **Polyuria:** 75.9% in diabetics vs. 7.5% in controls (Odds Ratio = 38.9, Cramér's $V = 0.666$).
- **Polydipsia:** 70.3% in diabetics vs. 4.0% in controls (Odds Ratio = 56.8, Cramér's $V = 0.650$).
- **Sudden Weight Loss:** 58.8% in diabetics vs. 14.0% in controls (OR = 8.75).
- **Partial Paresis:** 60.0% in diabetics vs. 16.0% in controls (OR = 7.88).

---

## Chapter 7: Inferential Statistics & Hypothesis Testing
Non-parametric Mann-Whitney U tests and Chi-Square tests with Cramér's V effect sizes confirmed that all analyzed variables reject the null hypothesis ($p < 0.001$). In ultra-large sample sizes ($N > 250,000$), $p$-values are trivial, mandating effect-size ranking. Features with $V > 0.25$ (`GenHlth`, `HighBP`, `Polyuria`) represent dominant predictive drivers.

---

## Chapter 8: Data Leakage Prevention & Feature Engineering
- **Zero-Leakage Protocol:** All preprocessing fit strictly in training folds; 20% holdout untouched.
- **`ClinicalFeatureEngineer`:** Engineered Glycemic Composite Index ($(\text{HbA1c} \times \text{Glucose})/100$), CVD comorbidity flag, Age $\ge 45$ threshold.
- **`BRFSSLifestyleFeatureEngineer`:** Engineered Lifestyle Risk Score (sedentary + low fruit/veg + smoker + alcohol), Vascular Comorbidity Count.
- **`SymptomsFeatureEngineer`:** Engineered Classic Triad Score (Polyuria + Polydipsia + Polyphagia).

---

## Chapter 9: Machine Learning Model Architectures & Selection
Nine candidate classifiers evaluated across linear, tree, bagging, boosting, instance-based, and kernel paradigms. Final model selection used the multi-objective utility formula:
$$\text{Selection Score} = 0.40 \times (\text{ROC-AUC}) + 0.35 \times (\text{Sensitivity}) + 0.25 \times (\text{F1-Score})$$

---

## Chapter 10: Benchmark Evaluation: Clinical Biomarkers Cohort
Evaluated on 20,000 untouched test records:
- **LightGBM (Calibrated):** ROC-AUC 0.9770, Sensitivity 92.1%, Specificity 94.4%, Accuracy 94.2%, Brier Score 0.0236.
- **HistGradientBoosting:** ROC-AUC 0.9771, Sensitivity 91.3%, Specificity 94.1%, Accuracy 93.8%.
- **Random Forest:** ROC-AUC 0.9743, Sensitivity 91.6%, Specificity 94.3%, Accuracy 94.1%.
- **Logistic Regression:** ROC-AUC 0.9597, Sensitivity 87.9%, Specificity 88.4%, Accuracy 88.3%.

---

## Chapter 11: Benchmark Evaluation: Lifestyle & Symptoms
- **CDC BRFSS Lifestyle (LightGBM Balanced):** ROC-AUC 0.8046, Sensitivity 74.5%, Specificity 73.7%, Accuracy 73.8%, Brier 0.1080.
- **Early Symptoms (Extra Trees Calibrated):** ROC-AUC 0.9996, Sensitivity 100.0%, Specificity 97.5%, Accuracy 99.0%, Brier 0.0074.

---

## Chapter 12: Scientific Ablation: Biomarkers vs. Lifestyle
Controlled ablation on identical clinical test partitions:
- **Lifestyle Only Baseline:** ROC-AUC 0.8251, Sensitivity 79.7%, Specificity 81.6%, Precision 29.2%.
- **Full Model (+HbA1c +Glucose):** ROC-AUC 0.9770, Sensitivity 92.1%, Specificity 94.4%, Precision 61.0%.
- **Diagnostic Value Add:** **+0.152 ROC-AUC gain**, **+12.4% Sensitivity boost**, false-negative rate reduced by 61%. Subsumes a two-tiered screening model for public health.

---

## Chapter 13: Class Imbalance & Decision Threshold Tuning
In BRFSS (13.9% prevalence), unweighted default models exhibit high accuracy (85.8%) but catastrophic sensitivity (14.5%, missing 85% of cases). Cost-sensitive balanced weighting recovers Sensitivity to 74.5%. Threshold optimization under $C_{\text{FN}} \gg C_{\text{FP}}$ shifts decision boundaries to $\theta^* \approx 0.20$.

---

## Chapter 14: Probability Calibration & Uncertainty
- Tree models produce uncalibrated probabilities; `CalibratedClassifierCV` (sigmoid/isotonic) reduced Brier loss from 0.0611 to 0.0236.
- 95% Confidence Intervals computed via Wilson score intervals to transparently communicate uncertainty.
- Stratified Risk Bands: Low (<25%), Moderate (25-50%), High (50-75%), Very High (>75%).

---

## Chapter 15: Explainable AI: Global Attribution with SHAP
TreeExplainer SHAP game-theoretic attributions:
- Clinical Cohort: HbA1c ($|\text{SHAP}|=2.15$) and Blood Glucose ($|\text{SHAP}|=1.42$) dominate.
- BRFSS Lifestyle: GenHlth ($|\text{SHAP}|=0.88$), HighBP ($|\text{SHAP}|=0.65$), and BMI ($|\text{SHAP}|=0.58$) lead.
- Symptoms Cohort: Polyuria ($|\text{SHAP}|=1.82$) and Polydipsia ($|\text{SHAP}|=1.74$) account for 70% of attribution.

---

## Chapter 16: Explainable AI: Local Patient Attribution
Local single-patient SHAP decompositions ($f(x) = \phi_0 + \sum \phi_i$) translate model log-odds into actionable, non-causal clinical narratives. Explicit warnings emphasize that statistical attribution does not establish biological causality.

---

## Chapter 17: The Prediabetes Multi-Class Detection Challenge
Experimental modeling of the CDC BRFSS 3-class target (`diabetes_012`):
- Class 1 (Prediabetes, 1.83% prevalence) achieved an F1-Score of only **0.076** (Macro F1: 0.4186).
- Takeaway: Prediabetes is clinically and symptomatically silent; behavioral questionnaires cannot replace laboratory testing (HbA1c 5.7-6.4%) for early dysglycemia.

---

## Chapter 18: Full-Stack Architecture & Software Engineering
- Decoupled modular design: data loading, cryptographic auditing, pipeline feature engineering, calibrated prediction service, ReportLab PDF compilation, and multi-page Streamlit dashboard.
- 28 automated Pytest tests passing with 100% success rate.
- Master one-command orchestration: `./run.sh` / `python3 run_all.py`.

---

## Chapter 19: Ethical Governance, Limitations & Conclusions
- **Current Status vs. Future Risk:** The cross-sectional nature of survey and EHR datasets models retrospective diagnostic likelihood, not 10-year future incidence.
- **Health Equity:** Cost-sensitive weighting prevents systemic under-detection in underserved demographics.
- **Conclusion:** While laboratory biomarkers are mandatory for definitive diagnosis (0.9770 AUC, 92.1% Sensitivity), non-invasive lifestyle models (0.8046 AUC, 74.5% Sensitivity) provide an accessible, scalable first-line community triage platform.
