"""
Custom Feature Engineering Transformers.
Implements domain-informed clinical and lifestyle feature representations
as scikit-learn compatible transformers with zero data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering for the Clinical & Laboratory Measurement Cohort:
    1. Interaction between HbA1c and Blood Glucose (Glycemic Composite Index).
    2. Comorbidity flag (Hypertension + Heart Disease).
    3. Age >= 45 screening flag (ADA standard threshold).
    4. Categorical BMI indicators (Underweight, Normal, Overweight, Obese).
    """

    def __init__(self, add_interactions: bool = True):
        self.add_interactions = add_interactions
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        # 1. Glycemic composite index
        if "HbA1c_level" in X_df.columns and "blood_glucose_level" in X_df.columns:
            X_df["glycemic_composite"] = (X_df["HbA1c_level"] * X_df["blood_glucose_level"]) / 100.0

        # 2. Cardiovascular comorbidity
        if "hypertension" in X_df.columns and "heart_disease" in X_df.columns:
            X_df["cvd_comorbidity"] = ((X_df["hypertension"] == 1) & (X_df["heart_disease"] == 1)).astype(int)

        # 3. High risk age (ADA guideline threshold)
        if "age" in X_df.columns:
            X_df["age_ge_45"] = (X_df["age"] >= 45.0).astype(int)

        # 4. BMI categories
        if "bmi" in X_df.columns:
            X_df["bmi_obese"] = (X_df["bmi"] >= 30.0).astype(int)
            X_df["bmi_overweight"] = ((X_df["bmi"] >= 25.0) & (X_df["bmi"] < 30.0)).astype(int)

        self.feature_names_out_ = list(X_df.columns)
        return X_df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class BRFSSLifestyleFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering for CDC BRFSS Lifestyle & Health Indicators:
    1. Lifestyle Risk Index (Unhealthy triad: Sedentary + Low Fruit/Veg + Smoker).
    2. Vascular & Metabolic Comorbidity Count (HighBP + HighChol + Stroke + HeartDisease).
    3. Total Disability/Distress Days (PhysHlth + MentHlth).
    4. Healthcare Barrier Flag (Cost barrier or no healthcare).
    """

    def __init__(self):
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # 1. Lifestyle Risk Index
        lifestyle_sum = 0
        if "PhysActivity" in X_df.columns:
            lifestyle_sum += (X_df["PhysActivity"] == 0).astype(int)
        if "Fruits" in X_df.columns:
            lifestyle_sum += (X_df["Fruits"] == 0).astype(int)
        if "Veggies" in X_df.columns:
            lifestyle_sum += (X_df["Veggies"] == 0).astype(int)
        if "Smoker" in X_df.columns:
            lifestyle_sum += (X_df["Smoker"] == 1).astype(int)
        if "HvyAlcoholConsump" in X_df.columns:
            lifestyle_sum += (X_df["HvyAlcoholConsump"] == 1).astype(int)
        X_df["lifestyle_risk_score"] = lifestyle_sum

        # 2. Vascular comorbidity count
        comorb = 0
        for col in ["HighBP", "HighChol", "Stroke", "HeartDiseaseorAttack"]:
            if col in X_df.columns:
                comorb += (X_df[col] == 1.0).astype(int)
        X_df["vascular_comorbidity_count"] = comorb

        # 3. Combined impairment days
        if "PhysHlth" in X_df.columns and "MentHlth" in X_df.columns:
            X_df["total_distress_days"] = np.clip(X_df["PhysHlth"] + X_df["MentHlth"], 0, 60)

        # 4. Healthcare access constraint
        if "NoDocbcCost" in X_df.columns and "AnyHealthcare" in X_df.columns:
            X_df["healthcare_access_barrier"] = (
                (X_df["NoDocbcCost"] == 1.0) | (X_df["AnyHealthcare"] == 0.0)
            ).astype(int)

        self.feature_names_out_ = list(X_df.columns)
        return X_df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)


class SymptomsFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Feature engineering for Early-Stage Symptoms (Sylhet Cohort):
    1. Cardinal Triad (Polyuria + Polydipsia + Polyphagia).
    2. Total Affirmative Symptom Burden (Count of positive symptoms out of 14).
    """

    def __init__(self):
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Classic Triad of Diabetes (Polyuria, Polydipsia, Polyphagia)
        triad_cols = ["Polyuria", "Polydipsia", "Polyphagia"]
        triad_count = 0
        for c in triad_cols:
            if c in X_df.columns:
                triad_count += (X_df[c] == 1).astype(int)
        X_df["classic_triad_count"] = triad_count

        # Total affirmative symptom burden
        symptom_cols = [
            c for c in X_df.columns
            if c not in ["Age", "Gender", "classic_triad_count", "class"]
        ]
        X_df["symptom_burden_score"] = X_df[symptom_cols].sum(axis=1)

        self.feature_names_out_ = list(X_df.columns)
        return X_df

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
