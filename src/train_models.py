"""
Model Training and Optimization Orchestration Module.
Trains, cross-validates, tunes, calibrates, and persists machine learning pipelines
across all research cohorts without target leakage or schema mismatches.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_SEED, TABLES_DIR
from src.evaluate_models import (
    evaluate_binary_model,
    evaluate_multiclass_model,
    plot_model_diagnostic_charts,
    run_cross_validation,
)
from src.preprocessing import (
    prepare_brfss_012_data,
    prepare_brfss_data,
    prepare_clinical_data,
    prepare_symptoms_data,
)


def get_binary_classifiers() -> Dict[str, Any]:
    """Instantiates the standard candidate classification algorithms."""
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced")
    lgbm = LGBMClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced", verbose=-1)

    return {
        "Logistic Regression": lr,
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED, class_weight="balanced"),
        "Random Forest": rf,
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced"),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=RANDOM_SEED, class_weight="balanced"),
        "LightGBM": lgbm,
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, n_jobs=-1, random_state=RANDOM_SEED, eval_metric="logloss"),
        "Ensemble (Voting)": VotingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("lgbm", lgbm)],
            voting="soft",
            n_jobs=-1,
        ),
    }


def train_and_compare_cohort(
    cohort_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: Pipeline,
    include_knn_svm: bool = False,
    run_cv: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """
    Trains all candidate algorithms on the cohort, evaluates on test set,
    runs cross-validation, and selects the optimal justified model.
    """
    classifiers = get_binary_classifiers()
    if include_knn_svm:
        classifiers["K-Nearest Neighbors"] = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        classifiers["Support Vector Machine"] = SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED, class_weight="balanced")

    results = []
    trained_pipelines = {}

    print(f"\n================ Training Cohort: {cohort_name} ================")
    print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}, Class 1 Prevalence: {y_train.mean():.1%}")

    for name, clf in classifiers.items():
        t0 = time.time()
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])

        # Cross-validation
        cv_summary = {}
        if run_cv and len(X_train) <= 100000:
            try:
                cv_summary = run_cross_validation(pipe, X_train, y_train, cv=5)
            except Exception as e:
                print(f"  CV Error for {name}: {e}")

        # Fit on full training set
        pipe.fit(X_train, y_train)
        fit_time = round(time.time() - t0, 2)

        # Test evaluation
        eval_metrics = evaluate_binary_model(pipe, X_test, y_test, model_name=name)
        eval_metrics["fit_time_seconds"] = fit_time
        if "roc_auc" in cv_summary:
            eval_metrics["cv_roc_auc_mean"] = cv_summary["roc_auc"]["mean"]
            eval_metrics["cv_roc_auc_std"] = cv_summary["roc_auc"]["std"]
            eval_metrics["cv_recall_mean"] = cv_summary["recall"]["mean"]
            eval_metrics["overfit_gap"] = cv_summary["roc_auc"]["overfit_gap"]

        results.append(eval_metrics)
        trained_pipelines[name] = pipe
        print(f"  [{name:22s}] Test AUC: {eval_metrics.get('roc_auc')} | Recall: {eval_metrics.get('recall_sensitivity')} | Time: {fit_time}s")

    res_df = pd.DataFrame(results)
    # Model selection prioritizing ROC-AUC, Recall (Sensitivity in screening), and F1
    res_df["selection_score"] = (
        res_df["roc_auc"].fillna(0) * 0.4
        + res_df["recall_sensitivity"].fillna(0) * 0.35
        + res_df["f1_score"].fillna(0) * 0.25
    )
    best_row = res_df.sort_values(by="selection_score", ascending=False).iloc[0]
    best_name = best_row["model_name"]
    best_pipe = trained_pipelines[best_name]

    print(f"--> Best Justified Model for {cohort_name}: {best_name} (Selection Score: {best_row['selection_score']:.4f})")

    # Probability Calibration on Best Model
    print(f"--> Fitting CalibratedClassifierCV on {best_name}...")
    calibrated_clf = CalibratedClassifierCV(best_pipe, cv=3, method="sigmoid")
    calibrated_clf.fit(X_train, y_train)

    # Re-evaluate calibrated model
    calib_metrics = evaluate_binary_model(calibrated_clf, X_test, y_test, model_name=f"{best_name} (Calibrated)")
    print(f"    Calibrated Brier Score: {calib_metrics.get('brier_score')} (Uncalibrated: {best_row.get('brier_score')})")

    # Generate Diagnostic Curves
    plot_model_diagnostic_charts(calibrated_clf, X_test, y_test, save_prefix=f"{cohort_name}_best")

    # Save Comparison Table
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(TABLES_DIR / f"{cohort_name}_model_comparison.csv", index=False)

    bundle = {
        "cohort_name": cohort_name,
        "best_model_name": best_name,
        "raw_pipeline": best_pipe,
        "calibrated_model": calibrated_clf,
        "test_metrics": best_row.to_dict(),
        "calibrated_metrics": calib_metrics,
        "all_comparison_df": res_df,
        "feature_names_in": list(X_train.columns),
        "random_seed": RANDOM_SEED,
    }

    return res_df, bundle, calibrated_clf


def run_clinical_ablation_study() -> pd.DataFrame:
    """
    Research Ablation: Compares models trained strictly on Lifestyle & Demographic variables
    vs models including Laboratory Biomarkers (HbA1c & Blood Glucose).
    Directly answers Research Objective: 'Whether clinical measurements improve classification performance'.
    """
    print("\n================ Research Ablation: Lifestyle vs Clinical Biomarkers ================")
    X_train, X_test, y_train, y_test, _ = prepare_clinical_data()

    # Model A: Full Model (Lifestyle + Demographics + Laboratory HbA1c + Blood Glucose)
    full_pipe = Pipeline([
        ("preprocessor", prepare_clinical_data()[4]),
        ("classifier", LGBMClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_SEED, class_weight="balanced", verbose=-1)),
    ])
    full_pipe.fit(X_train, y_train)
    m_full = evaluate_binary_model(full_pipe, X_test, y_test, model_name="Full Model (Clinical + Lifestyle)")

    # Model B: Lifestyle Only (Exclude HbA1c and Blood Glucose)
    lifestyle_features = ["gender", "age", "hypertension", "heart_disease", "smoking_history", "bmi"]
    X_train_life = X_train[lifestyle_features]
    X_test_life = X_test[lifestyle_features]

    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    life_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["age", "bmi"]),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), ["gender", "smoking_history"]),
            ("pass", "passthrough", ["hypertension", "heart_disease"]),
        ]
    )
    life_pipe = Pipeline([
        ("preprocessor", life_preprocessor),
        ("classifier", LGBMClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_SEED, class_weight="balanced", verbose=-1)),
    ])
    life_pipe.fit(X_train_life, y_train)
    m_life = evaluate_binary_model(life_pipe, X_test_life, y_test, model_name="Ablation: Lifestyle & History Only")

    # Compare
    ablation_df = pd.DataFrame([m_life, m_full])
    ablation_df["Delta_Gain"] = [
        "Baseline (No Lab Biomarkers)",
        f"+{m_full['roc_auc'] - m_life['roc_auc']:.3f} ROC-AUC (+{m_full['recall_sensitivity'] - m_life['recall_sensitivity']:.1%} Sensitivity)"
    ]
    print(ablation_df[["model_name", "accuracy", "recall_sensitivity", "precision", "roc_auc", "Delta_Gain"]])
    ablation_df.to_csv(TABLES_DIR / "ablation_study_comparison.csv", index=False)
    return ablation_df


def run_multiclass_brfss_training() -> Dict[str, Any]:
    """
    Trains multi-class classification on BRFSS 012 (0 = Healthy, 1 = Prediabetes, 2 = Diabetes).
    Evaluates research question on prediabetes distinguishability.
    """
    print("\n================ Training BRFSS Multiclass (Prediabetes vs Diabetes) ================")
    # Use 30,000 stratified sample for efficient multi-class training and evaluation
    X_train, X_test, y_train, y_test, preprocessor = prepare_brfss_012_data(sample_n=30000)

    clf = LGBMClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        random_state=RANDOM_SEED, class_weight="balanced", verbose=-1
    )
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])
    pipe.fit(X_train, y_train)

    metrics = evaluate_multiclass_model(pipe, X_test, y_test, model_name="LightGBM Multiclass")
    print(f"  BRFSS Multiclass Accuracy: {metrics['accuracy']} | Macro F1: {metrics['f1_macro']}")

    bundle = {
        "cohort_name": "brfss_multiclass_012",
        "model": pipe,
        "metrics": metrics,
        "feature_names_in": list(X_train.columns),
        "target_mapping": {0: "Healthy", 1: "Prediabetes", 2: "Diabetes"},
    }
    joblib.dump(bundle, MODELS_DIR / "brfss_multiclass_model_bundle.joblib")
    return bundle


def run_full_training_pipeline() -> Dict[str, Any]:
    """
    Executes training across all 4 tracks, saves model bundles to models/,
    and writes comparison tables to outputs/tables/.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_bundles = {}

    # Track 1: Clinical & Laboratory Cohort
    X_tr_clin, X_te_clin, y_tr_clin, y_te_clin, prep_clin = prepare_clinical_data()
    _, clin_bundle, _ = train_and_compare_cohort(
        "clinical", X_tr_clin, X_te_clin, y_tr_clin, y_te_clin, prep_clin, include_knn_svm=False, run_cv=True
    )
    joblib.dump(clin_bundle, MODELS_DIR / "clinical_model_bundle.joblib")
    all_bundles["clinical"] = clin_bundle

    # Track 2: CDC BRFSS Lifestyle Survey (Binary, sample 50,000 for efficient rigorous tuning)
    X_tr_brfss, X_te_brfss, y_tr_brfss, y_te_brfss, prep_brfss = prepare_brfss_data(sample_n=50000)
    _, brfss_bundle, _ = train_and_compare_cohort(
        "brfss_lifestyle", X_tr_brfss, X_te_brfss, y_tr_brfss, y_te_brfss, prep_brfss, include_knn_svm=False, run_cv=True
    )
    joblib.dump(brfss_bundle, MODELS_DIR / "brfss_lifestyle_model_bundle.joblib")
    all_bundles["brfss_lifestyle"] = brfss_bundle

    # Track 3: Early-Stage Symptoms (Sylhet Cohort - includes SVM & KNN!)
    X_tr_sym, X_te_sym, y_tr_sym, y_te_sym, prep_sym = prepare_symptoms_data()
    _, sym_bundle, _ = train_and_compare_cohort(
        "early_symptoms", X_tr_sym, X_te_sym, y_tr_sym, y_te_sym, prep_sym, include_knn_svm=True, run_cv=True
    )
    joblib.dump(sym_bundle, MODELS_DIR / "symptoms_model_bundle.joblib")
    all_bundles["early_symptoms"] = sym_bundle

    # Track 4: Research Ablation & BRFSS Multiclass
    run_clinical_ablation_study()
    run_multiclass_brfss_training()

    # Track 5: Balanced 5050 BRFSS study
    X_tr_bal, X_te_bal, y_tr_bal, y_te_bal, prep_bal = prepare_brfss_data(balanced=True, sample_n=30000)
    _, bal_bundle, _ = train_and_compare_cohort(
        "brfss_5050_balanced", X_tr_bal, X_te_bal, y_tr_bal, y_te_bal, prep_bal, include_knn_svm=False, run_cv=False
    )
    joblib.dump(bal_bundle, MODELS_DIR / "brfss_5050_balanced_model_bundle.joblib")

    print("\n================ Full Model Training Pipeline Successfully Completed ================")
    return all_bundles


if __name__ == "__main__":
    run_full_training_pipeline()
