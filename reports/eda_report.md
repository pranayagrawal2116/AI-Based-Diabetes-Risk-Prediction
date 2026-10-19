# Exploratory Data Analysis & Statistical Findings Report

**Project:** AI-Based Diabetes Risk Prediction Using Machine Learning and Lifestyle Factors

---

## 1. Clinical Cohort Statistical Profile (100,000 Records)
- **Strongest Statistical Predictors:**
  - `age` (Numerical): Test=Mann-Whitney U, Effect=Median Diff (22.0), p=0.00e+00
  - `blood_glucose_level` (Numerical): Test=Mann-Whitney U, Effect=Median Diff (20.0), p=0.00e+00
  - `bmi` (Numerical): Test=Mann-Whitney U, Effect=Median Diff (2.65), p=0.00e+00
  - `HbA1c_level` (Numerical): Test=Mann-Whitney U, Effect=Median Diff (0.8), p=0.00e+00
  - `age_group` (Categorical): Test=Chi-Square, Effect=Cramer's V (0.267), p=0.00e+00

### Key Clinical Observations:
- **Biomarker Thresholds:** Diabetic patients exhibit a median HbA1c of 6.6% compared to 5.8% in non-diabetics. Blood glucose median is 160.0 mg/dL vs. 130.0 mg/dL.
- **Age & Smoking:** Diabetes prevalence increases sharply from 1.3% in <20 years to 19.8% in the 65-80 bracket. Former smokers display the highest prevalence (17.0%), largely correlated with older age.

---

## 2. CDC BRFSS Lifestyle Survey Profile (253,680 Records)
- **Strongest Survey Predictors:**
  - `BMI` (Numerical): Test=Mann-Whitney U, Effect=Median Diff (4.0), p=0.00e+00
  - `Age` (Numerical): Test=Mann-Whitney U, Effect=Median Diff (2.0), p=0.00e+00
  - `PhysHlth` (Numerical): Test=Mann-Whitney U, Effect=Median Diff (1.0), p=0.00e+00
  - `GenHlth` (Categorical): Test=Chi-Square, Effect=Cramer's V (0.299), p=0.00e+00
  - `HighBP` (Categorical): Test=Chi-Square, Effect=Cramer's V (0.263), p=0.00e+00

### Key Lifestyle & Behavioral Observations:
- **General Health Perception:** Self-rated general health (`GenHlth`) has the strongest association (Cramér's V = 0.301). In poor general health, diabetes prevalence reaches 38.4% vs 2.1% in excellent health.
- **Modifiable Lifestyle Factors:** Lack of physical activity increases prevalence from 11.0% to 23.3%. Individuals with zero fruit consumption have 16.2% prevalence vs 12.6% in daily consumers.
- **BMI Impact:** Median BMI for diabetic respondents is 31.0 (obese class I) compared to 27.0 (overweight) in non-diabetic controls.

---

## 3. Early Stage Symptoms Profile (Sylhet Cohort, 520 Records)
- **Strongest Symptom Predictors:**
  - `Age` (Numerical): Test=Mann-Whitney U, Effect=Median Diff (3.0), p=0.0124
  - `Polyuria` (Categorical): Test=Chi-Square, Effect=Cramer's V (0.661), p=1.74e-51
  - `Polydipsia` (Categorical): Test=Chi-Square, Effect=Cramer's V (0.644), p=6.19e-49
  - `Gender` (Categorical): Test=Chi-Square, Effect=Cramer's V (0.443), p=3.29e-24
  - `sudden weight loss` (Categorical): Test=Chi-Square, Effect=Cramer's V (0.431), p=5.97e-23

### Key Symptom Observations:
- **Cardinal Diabetic Symptoms:** `Polyuria` (excessive urination, Cramér's V = 0.666) and `Polydipsia` (excessive thirst, Cramér's V = 0.650) are overwhelmingly prevalent in diabetic individuals (75.9% and 70.3% vs 7.5% and 4.0% in controls).
- **Secondary Signs:** Sudden weight loss (58.8% vs 14.0%) and partial paresis (60.0% vs 16.0%) also demonstrated robust discriminatory power.

---

## 4. Visualizations Generated
- `outputs/figures/clinical_target_distribution.png`
- `outputs/figures/clinical_biomarkers_boxplot.png`
- `outputs/figures/clinical_hba1c_glucose_scatter.png`
- `outputs/figures/clinical_correlation_matrix.png`
- `outputs/figures/clinical_smoking_age_prevalence.png`
- `outputs/figures/brfss_cohort_distributions.png`
- `outputs/figures/brfss_lifestyle_prevalence.png`
- `outputs/figures/brfss_age_prevalence.png`
- `outputs/figures/brfss_bmi_genhealth.png`
- `outputs/figures/symptoms_prevalence_comparison.png`
