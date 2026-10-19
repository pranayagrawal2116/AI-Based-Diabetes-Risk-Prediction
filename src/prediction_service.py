"""
Prediction and Inference Service Module.
Loads persisted model artifacts, validates input schemas, performs calibrated inference,
computes confidence intervals, stratifies risk categories, and integrates local explainability.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.config import (
    CLINICAL_DISCLAIMER,
    CURRENT_STATUS_VS_FUTURE_RISK_NOTE,
    MODELS_DIR,
    RISK_CATEGORIES,
)
from src.explainability import explain_single_prediction


class PredictionService:
    """
    Central inference engine serving all distinct risk assessment modes.
    Enforces schema compatibility, caching, and validation.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.bundles: Dict[str, Any] = {}

    def get_bundle(self, mode: str) -> Dict[str, Any]:
        """Loads and caches model bundle for the specified assessment mode."""
        filename_map = {
            "clinical": "clinical_model_bundle.joblib",
            "lifestyle": "brfss_lifestyle_model_bundle.joblib",
            "symptoms": "symptoms_model_bundle.joblib",
            "multiclass": "brfss_multiclass_model_bundle.joblib",
            "balanced": "brfss_5050_balanced_model_bundle.joblib",
        }

        if mode not in filename_map:
            raise ValueError(f"Unknown assessment mode: '{mode}'. Supported modes: {list(filename_map.keys())}")

        if mode in self.bundles:
            return self.bundles[mode]

        bundle_path = self.models_dir / filename_map[mode]
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Model artifact for mode '{mode}' was not found at '{bundle_path}'. "
                f"Please run the training pipeline first: 'python -m src.train_models'"
            )

        bundle = joblib.load(bundle_path)
        self.bundles[mode] = bundle
        return bundle

    def get_risk_stratification(self, probability: float) -> Dict[str, Any]:
        """Maps calibrated model probability to a documented research risk category."""
        p = float(np.clip(probability, 0.0, 1.0))
        for cat_name, cat_info in RISK_CATEGORIES.items():
            low, high = cat_info["range"]
            if low <= p <= high or (cat_name == "Very High" and p >= low):
                return {
                    "category": cat_name,
                    "badge": cat_info["badge"],
                    "color": cat_info["color"],
                    "description": cat_info["description"],
                    "range": cat_info["range"],
                }
        return {
            "category": "Moderate",
            "badge": "MODERATE RISK",
            "color": "#F59E0B",
            "description": "Probability within intermediate range.",
            "range": (0.25, 0.50),
        }

    def compute_confidence_interval(self, prob: float, n_effective: int = 100) -> Tuple[float, float]:
        """
        Computes 95% Wilson-score or normal approximation confidence interval around predicted probability.
        Reflects model estimation uncertainty for patient-facing transparency.
        """
        z = 1.96  # 95% CI
        p = float(np.clip(prob, 0.001, 0.999))
        margin = z * np.sqrt((p * (1 - p)) / n_effective)
        lower = max(0.0, p - margin)
        upper = min(1.0, p + margin)
        return round(lower, 3), round(upper, 3)

    def predict_clinical(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference for Mode 1: Clinical & Laboratory Measurements.
        Inputs: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level.
        """
        bundle = self.get_bundle("clinical")
        calibrated_model = bundle["calibrated_model"]

        # Validate input schema
        required = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level"]
        for k in required:
            if k not in input_dict:
                raise ValueError(f"Missing required clinical input feature: '{k}'")

        input_df = pd.DataFrame([input_dict])
        prob = float(calibrated_model.predict_proba(input_df)[0, 1])
        pred_class = int(prob >= 0.5)

        ci_lower, ci_upper = self.compute_confidence_interval(prob)
        risk_info = self.get_risk_stratification(prob)

        # Local explainability
        explanation = explain_single_prediction(bundle, input_df)

        return {
            "mode": "Clinical Measurements Assessment",
            "predicted_class": "Diabetic" if pred_class == 1 else "Non-Diabetic",
            "predicted_class_int": pred_class,
            "calibrated_probability": round(prob, 4),
            "confidence_interval_95": [ci_lower, ci_upper],
            "risk_stratification": risk_info,
            "local_explanation": explanation,
            "disclaimer": CLINICAL_DISCLAIMER,
            "scientific_note": CURRENT_STATUS_VS_FUTURE_RISK_NOTE,
        }

    def predict_lifestyle(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference for Mode 2: Lifestyle and Behavioral Survey Factors (BRFSS 2015).
        Inputs: HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack,
                PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
                GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income.
        """
        bundle = self.get_bundle("lifestyle")
        calibrated_model = bundle["calibrated_model"]

        required = [
            "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
            "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
        ]
        for k in required:
            if k not in input_dict:
                raise ValueError(f"Missing required lifestyle input feature: '{k}'")

        input_df = pd.DataFrame([input_dict])
        prob = float(calibrated_model.predict_proba(input_df)[0, 1])
        pred_class = int(prob >= 0.5)

        ci_lower, ci_upper = self.compute_confidence_interval(prob)
        risk_info = self.get_risk_stratification(prob)

        explanation = explain_single_prediction(bundle, input_df)

        return {
            "mode": "Lifestyle and General Health Assessment",
            "predicted_class": "Diabetic Likelihood" if pred_class == 1 else "Low Diabetic Likelihood",
            "predicted_class_int": pred_class,
            "calibrated_probability": round(prob, 4),
            "confidence_interval_95": [ci_lower, ci_upper],
            "risk_stratification": risk_info,
            "local_explanation": explanation,
            "disclaimer": CLINICAL_DISCLAIMER,
            "scientific_note": CURRENT_STATUS_VS_FUTURE_RISK_NOTE,
        }

    def predict_symptoms(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference for Mode 3: Early-Stage Symptom Triage (Sylhet Hospital).
        """
        bundle = self.get_bundle("symptoms")
        calibrated_model = bundle["calibrated_model"]

        required = [
            "Age", "Gender", "Polyuria", "Polydipsia", "sudden weight loss",
            "weakness", "Polyphagia", "Genital thrush", "visual blurring",
            "Itching", "Irritability", "delayed healing", "partial paresis",
            "muscle stiffness", "Alopecia", "Obesity"
        ]
        for k in required:
            if k not in input_dict:
                raise ValueError(f"Missing required symptom input feature: '{k}'")

        input_df = pd.DataFrame([input_dict])
        prob = float(calibrated_model.predict_proba(input_df)[0, 1])
        pred_class = int(prob >= 0.5)

        ci_lower, ci_upper = self.compute_confidence_interval(prob)
        risk_info = self.get_risk_stratification(prob)

        explanation = explain_single_prediction(bundle, input_df)

        return {
            "mode": "Early-Stage Symptom Screening",
            "predicted_class": "Positive Symptom Screening (High Risk)" if pred_class == 1 else "Negative Symptom Screening (Low Risk)",
            "predicted_class_int": pred_class,
            "calibrated_probability": round(prob, 4),
            "confidence_interval_95": [ci_lower, ci_upper],
            "risk_stratification": risk_info,
            "local_explanation": explanation,
            "disclaimer": CLINICAL_DISCLAIMER,
            "scientific_note": CURRENT_STATUS_VS_FUTURE_RISK_NOTE,
        }

    def predict_multiclass_012(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference for Mode 4: 3-Class BRFSS Prediabetes vs Diabetes vs Healthy.
        """
        bundle = self.get_bundle("multiclass")
        model = bundle["model"]

        input_df = pd.DataFrame([input_dict])
        probs = model.predict_proba(input_df)[0]
        pred_idx = int(np.argmax(probs))

        classes = ["Healthy / No Diabetes", "Prediabetes", "Diabetes"]
        return {
            "mode": "BRFSS 3-Class Assessment (Prediabetes vs Diabetes)",
            "predicted_class": classes[pred_idx],
            "predicted_class_idx": pred_idx,
            "probabilities": {
                "Healthy": round(float(probs[0]), 4),
                "Prediabetes": round(float(probs[1]), 4),
                "Diabetes": round(float(probs[2]), 4),
            },
            "disclaimer": CLINICAL_DISCLAIMER,
        }
