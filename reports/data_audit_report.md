# Data Integrity & Cohort Audit Report

**Project:** AI-Based Diabetes Risk Prediction Using Machine Learning and Lifestyle Factors
**Audit Standard:** Strict Medical ML Leakage Prevention & Multi-Cohort Integrity Verification

---

## 1. Summary of Available Datasets

| Filename | Records | Features | Duplicates (%) | Nulls | File Size | Target Variable | Class Imbalance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `diabetes_prediction_dataset.csv` | 100,000 | 9 | 3,854 (3.85%) | 0 | 3.63 MB | `diabetes` | 0: 91500 (91.5%)<br>1: 8500 (8.5%) |
| `diabetes_prediction_dataset(1).csv` | 100,000 | 9 | 3,854 (3.85%) | 0 | 3.63 MB | `diabetes` | 0: 91500 (91.5%)<br>1: 8500 (8.5%) |
| `diabetes_data_upload.csv` | 520 | 17 | 269 (51.73%) | 0 | 0.03 MB | `class` | Positive: 320 (61.54%)<br>Negative: 200 (38.46%) |
| `diabetes_binary_health_indicators_BRFSS2015.csv` | 253,680 | 22 | 24,206 (9.54%) | 0 | 21.68 MB | `Diabetes_binary` | 0.0: 218334 (86.07%)<br>1.0: 35346 (13.93%) |
| `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` | 70,692 | 22 | 1,635 (2.31%) | 0 | 6.05 MB | `Diabetes_binary` | 0.0: 35346 (50.0%)<br>1.0: 35346 (50.0%) |
| `diabetes_012_health_indicators_BRFSS2015.csv` | 253,680 | 22 | 23,899 (9.42%) | 0 | 21.68 MB | `Diabetes_012` | 0.0: 213703 (84.24%)<br>2.0: 35346 (13.93%)<br>1.0: 4631 (1.83%) |

---

## 2. Critical Cross-Cohort Integrity Findings

### A. Clinical Dataset Duplicate Verification
- **Primary File:** `diabetes_prediction_dataset.csv`
- **Secondary File:** `diabetes_prediction_dataset(1).csv`
- **Primary MD5:** `cb243b99d81d96e9aeb08944ce9ac066`
- **Secondary MD5:** `cb243b99d81d96e9aeb08944ce9ac066`
- **Status:** **100% IDENTICAL DUPLICATE**
- **Audit Decision:** CONFIRMED 100% IDENTICAL DUPLICATE FILE. The two clinical files have identical byte hashes and row distributions (100,000 rows each). To maintain scientific integrity and prevent data snooping / data duplication, file 2 is flagged as an exact duplicate and excluded from combined training.

### B. CDC BRFSS Binary vs. 3-Class Cohort Relationship
- **Binary Dataset:** `diabetes_binary_health_indicators_BRFSS2015.csv`
- **3-Class Dataset:** `diabetes_012_health_indicators_BRFSS2015.csv`
- **Same Respondent Population:** **True** (253,680 records)
- **Feature Vectors Match:** **True**
- **Target Relationship:** CONFIRMED SAME UNDERLYING POPULATION. 'diabetes_binary_health_indicators_BRFSS2015.csv' and 'diabetes_012_health_indicators_BRFSS2015.csv' represent the exact same 253,680 survey respondents. In the binary file, prediabetes (class 1 in 012) was merged with non-diabetics (class 0). They MUST NOT be combined or treated as independent validation cohorts.

### C. CDC BRFSS Binary vs. 50-50 Balanced Subsample Overlap
- **Full Cohort Size:** 253,680 rows (35,346 diabetes cases, 13.93% prevalence)
- **Balanced Cohort Size:** 70,692 rows (35,346 diabetes cases, 50.00% prevalence)
- **Audit Decision:** CONFIRMED DERIVED SUBSAMPLE. 'diabetes_binary_5050split_health_indicators_BRFSS2015.csv' is an artificially balanced subsample containing all 35,346 diabetes cases from BRFSS and 35,346 randomly sampled non-diabetic controls. It CANNOT serve as an independent external test set for models trained on the full BRFSS dataset, as all diabetic cases overlap.

---

## 3. Strict Preprocessing & Separation Strategy

1. **Never Concatenate Distinct Datasets:** Survey responses (BRFSS), clinical measurements (EHR), and acute symptoms (hospital questionnaires) represent fundamentally distinct clinical modalities.
2. **Four Specialized Modeling Tracks:**
   - **Track 1 (Lifestyle & General Health):** Trained on BRFSS 2015 to evaluate how modifiable lifestyle factors predict diabetes status without laboratory blood tests.
   - **Track 2 (Clinical & Laboratory):** Trained on the 100k EHR dataset incorporating HbA1c and fasting blood glucose.
   - **Track 3 (Early Symptom Screening):** Trained on the Sylhet clinical dataset for acute symptom triage.
   - **Track 4 (Class-Imbalance & Multiclass Research):** Comparative evaluation on balanced BRFSS 50-50 and BRFSS 3-class (prediabetes vs diabetes).
3. **Zero Information Leakage:** Every scaler, imputer, and encoder is fit strictly inside cross-validation training folds.
