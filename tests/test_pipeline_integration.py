"""
Integration Test for the complete Training-to-Inference Lifecycle.
Uses a small fixture to verify end-to-end compatibility without data leakage or schema mismatch.
"""

import tempfile
from pathlib import Path
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src.evaluate_models import evaluate_binary_model
from src.preprocessing import get_clinical_preprocessor
from src.prediction_service import PredictionService


def test_end_to_end_training_to_inference():
    # 1. Create small test fixture (n=60)
    data = {
        "gender": ["Male", "Female"] * 30,
        "age": [45.0, 62.0, 31.0, 70.0, 24.0, 58.0] * 10,
        "hypertension": [0, 1, 0, 1, 0, 1] * 10,
        "heart_disease": [0, 0, 0, 1, 0, 0] * 10,
        "smoking_history": ["never", "former", "current", "never", "not current", "No Info"] * 10,
        "bmi": [24.0, 31.5, 22.0, 36.0, 21.0, 29.0] * 10,
        "HbA1c_level": [5.2, 7.1, 4.9, 8.2, 5.0, 6.8] * 10,
        "blood_glucose_level": [90, 190, 85, 240, 95, 170] * 10,
        "diabetes": [0, 1, 0, 1, 0, 1] * 10,
    }
    df = pd.DataFrame(data)

    X = df.drop(columns=["diabetes"])
    y = df["diabetes"]

    # 2. Build pipeline
    preprocessor = get_clinical_preprocessor()
    clf = HistGradientBoostingClassifier(max_iter=15, random_state=42)
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    pipe.fit(X, y)
    calib = CalibratedClassifierCV(pipe, cv=2)
    calib.fit(X, y)

    # 3. Save to temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        bundle = {
            "cohort_name": "clinical_fixture",
            "best_model_name": "HistGradientBoosting",
            "raw_pipeline": pipe,
            "calibrated_model": calib,
            "test_metrics": {"accuracy": 1.0},
            "calibrated_metrics": {"brier_score": 0.05},
            "feature_names_in": list(X.columns),
            "random_seed": 42,
        }
        bundle_file = tmp_path / "clinical_model_bundle.joblib"
        joblib.dump(bundle, bundle_file)

        # 4. Initialize PredictionService with custom directory
        service = PredictionService(models_dir=tmp_path)
        test_patient = {
            "gender": "Female",
            "age": 65.0,
            "hypertension": 1,
            "heart_disease": 0,
            "smoking_history": "former",
            "bmi": 33.0,
            "HbA1c_level": 7.8,
            "blood_glucose_level": 210,
        }
        res = service.predict_clinical(test_patient)

        assert 0.0 <= res["calibrated_probability"] <= 1.0
        assert res["risk_stratification"]["category"] in ["Moderate", "High", "Very High"]
        assert len(res["local_explanation"]["contributions"]) > 0
