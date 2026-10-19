"""
Tests for Prediction and Inference Service.
"""

from pathlib import Path
import pytest
from src.prediction_service import PredictionService


@pytest.fixture
def service():
    return PredictionService()


def test_predict_clinical_valid(service):
    inp = {
        "gender": "Male",
        "age": 60.0,
        "hypertension": 1,
        "heart_disease": 1,
        "smoking_history": "former",
        "bmi": 32.5,
        "HbA1c_level": 7.5,
        "blood_glucose_level": 210,
    }
    res = service.predict_clinical(inp)
    assert res["predicted_class"] == "Diabetic"
    assert 0.0 <= res["calibrated_probability"] <= 1.0
    assert len(res["confidence_interval_95"]) == 2
    assert res["confidence_interval_95"][0] <= res["confidence_interval_95"][1]
    assert res["risk_stratification"]["category"] in ["High", "Very High"]
    assert "local_explanation" in res
    assert "disclaimer" in res


def test_predict_clinical_healthy(service):
    inp = {
        "gender": "Female",
        "age": 25.0,
        "hypertension": 0,
        "heart_disease": 0,
        "smoking_history": "never",
        "bmi": 21.0,
        "HbA1c_level": 4.8,
        "blood_glucose_level": 85,
    }
    res = service.predict_clinical(inp)
    assert res["predicted_class"] == "Non-Diabetic"
    assert res["calibrated_probability"] < 0.35
    assert res["risk_stratification"]["category"] in ["Low", "Moderate"]


def test_predict_clinical_missing_input(service):
    inp = {
        "gender": "Male",
        "age": 50.0,
        # Missing HbA1c and glucose
    }
    with pytest.raises(ValueError):
        service.predict_clinical(inp)


def test_predict_lifestyle_valid(service):
    inp = {
        "HighBP": 0.0, "HighChol": 0.0, "CholCheck": 1.0, "BMI": 22.0, "Smoker": 0.0,
        "Stroke": 0.0, "HeartDiseaseorAttack": 0.0, "PhysActivity": 1.0, "Fruits": 1.0,
        "Veggies": 1.0, "HvyAlcoholConsump": 0.0, "AnyHealthcare": 1.0, "NoDocbcCost": 0.0,
        "GenHlth": 1.0, "MentHlth": 0.0, "PhysHlth": 0.0, "DiffWalk": 0.0, "Sex": 0.0,
        "Age": 3.0, "Education": 6.0, "Income": 8.0
    }
    res = service.predict_lifestyle(inp)
    assert 0.0 <= res["calibrated_probability"] <= 1.0
    assert res["risk_stratification"]["category"] in ["Low", "Moderate"]


def test_predict_symptoms_valid(service):
    inp = {
        "Age": 55, "Gender": 1, "Polyuria": 1, "Polydipsia": 1, "sudden weight loss": 1,
        "weakness": 1, "Polyphagia": 1, "Genital thrush": 0, "visual blurring": 1,
        "Itching": 1, "Irritability": 1, "delayed healing": 1, "partial paresis": 1,
        "muscle stiffness": 1, "Alopecia": 0, "Obesity": 1
    }
    res = service.predict_symptoms(inp)
    assert "Positive" in res["predicted_class"]
    assert res["calibrated_probability"] > 0.80


def test_predict_multiclass_012(service):
    inp = {
        "HighBP": 1.0, "HighChol": 1.0, "CholCheck": 1.0, "BMI": 30.0,
        "Smoker": 0.0, "Stroke": 0.0, "HeartDiseaseorAttack": 0.0,
        "PhysActivity": 1.0, "Fruits": 1.0, "Veggies": 1.0, "HvyAlcoholConsump": 0.0,
        "AnyHealthcare": 1.0, "NoDocbcCost": 0.0, "GenHlth": 3.0,
        "MentHlth": 0.0, "PhysHlth": 0.0, "DiffWalk": 0.0, "Sex": 1.0,
        "Age": 8.0, "Education": 5.0, "Income": 6.0
    }
    res = service.predict_multiclass_012(inp)
    assert "predicted_class" in res
    assert "probabilities" in res
    assert len(res["probabilities"]) == 3
    assert abs(sum(res["probabilities"].values()) - 1.0) < 0.01
