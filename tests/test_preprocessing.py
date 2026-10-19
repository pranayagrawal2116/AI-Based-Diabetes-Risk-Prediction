"""
Tests for Feature Engineering and Preprocessing Pipelines.
"""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    ClinicalFeatureEngineer,
    BRFSSLifestyleFeatureEngineer,
    SymptomsFeatureEngineer,
)
from src.preprocessing import (
    get_clinical_preprocessor,
    get_brfss_preprocessor,
    get_symptoms_preprocessor,
    prepare_clinical_data,
    prepare_symptoms_data,
)


def test_clinical_feature_engineer():
    df = pd.DataFrame({
        "age": [50.0, 30.0],
        "bmi": [32.0, 24.0],
        "HbA1c_level": [7.0, 5.2],
        "blood_glucose_level": [180, 95],
        "hypertension": [1, 0],
        "heart_disease": [1, 0],
    })
    fe = ClinicalFeatureEngineer()
    out = fe.transform(df)
    assert "glycemic_composite" in out.columns
    assert "cvd_comorbidity" in out.columns
    assert "age_ge_45" in out.columns
    assert "bmi_obese" in out.columns
    assert out.loc[0, "cvd_comorbidity"] == 1
    assert out.loc[1, "cvd_comorbidity"] == 0


def test_brfss_lifestyle_feature_engineer():
    df = pd.DataFrame({
        "PhysActivity": [0.0, 1.0],
        "Fruits": [0.0, 1.0],
        "Veggies": [0.0, 1.0],
        "Smoker": [1.0, 0.0],
        "HvyAlcoholConsump": [1.0, 0.0],
        "HighBP": [1.0, 0.0],
        "HighChol": [1.0, 0.0],
        "Stroke": [0.0, 0.0],
        "HeartDiseaseorAttack": [1.0, 0.0],
        "PhysHlth": [10.0, 0.0],
        "MentHlth": [5.0, 0.0],
        "NoDocbcCost": [1.0, 0.0],
        "AnyHealthcare": [0.0, 1.0],
    })
    fe = BRFSSLifestyleFeatureEngineer()
    out = fe.transform(df)
    assert "lifestyle_risk_score" in out.columns
    assert "vascular_comorbidity_count" in out.columns
    assert "total_distress_days" in out.columns
    assert "healthcare_access_barrier" in out.columns
    assert out.loc[0, "lifestyle_risk_score"] == 5
    assert out.loc[1, "lifestyle_risk_score"] == 0


def test_symptoms_feature_engineer():
    df = pd.DataFrame({
        "Age": [50, 40],
        "Gender": [1, 0],
        "Polyuria": [1, 0],
        "Polydipsia": [1, 0],
        "Polyphagia": [1, 0],
        "sudden weight loss": [1, 0],
        "weakness": [1, 0],
    })
    fe = SymptomsFeatureEngineer()
    out = fe.transform(df)
    assert "classic_triad_count" in out.columns
    assert "symptom_burden_score" in out.columns
    assert out.loc[0, "classic_triad_count"] == 3
    assert out.loc[1, "classic_triad_count"] == 0


def test_clinical_preprocessor_pipeline():
    pipe = get_clinical_preprocessor()
    df = pd.DataFrame({
        "gender": ["Male", "Female"],
        "age": [55.0, 35.0],
        "hypertension": [1, 0],
        "heart_disease": [0, 0],
        "smoking_history": ["former", "never"],
        "bmi": [29.0, 22.0],
        "HbA1c_level": [6.5, 5.4],
        "blood_glucose_level": [160, 90],
    })
    pipe.fit(df)
    transformed = pipe.transform(df)
    assert transformed.shape[0] == 2
    assert not np.isnan(transformed).any()


def test_data_leakage_and_stratification():
    X_tr, X_te, y_tr, y_te, _ = prepare_clinical_data(test_size=0.2)
    # Check that indices are strictly disjoint
    assert len(set(X_tr.index).intersection(set(X_te.index))) == 0
    # Check that stratification preserves class ratio within 1%
    tr_ratio = y_tr.mean()
    te_ratio = y_te.mean()
    assert abs(tr_ratio - te_ratio) < 0.01
