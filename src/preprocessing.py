"""
Data Preprocessing and Pipeline Construction Module.
Builds scikit-learn ColumnTransformers and Pipelines ensuring strict train/test separation,
absence of target leakage, and reproducible stratified splitting.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED
from src.data_loader import (
    load_brfss_012_data,
    load_brfss_5050_data,
    load_brfss_lifestyle_data,
    load_clinical_data,
    load_symptoms_data,
)
from src.feature_engineering import (
    BRFSSLifestyleFeatureEngineer,
    ClinicalFeatureEngineer,
    SymptomsFeatureEngineer,
)


def get_clinical_preprocessor() -> Tuple[Pipeline, List[str]]:
    """
    Constructs the preprocessor for the Clinical & Laboratory Measurement Cohort.
    Includes feature engineering, one-hot encoding for categorical variables,
    standard scaling for numerical variables, and passthrough for binary flags.
    """
    num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level", "glycemic_composite"]
    cat_cols = ["gender", "smoking_history"]
    passthrough_cols = ["hypertension", "heart_disease", "cvd_comorbidity", "age_ge_45", "bmi_obese", "bmi_overweight"]

    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
            ("pass", "passthrough", passthrough_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline([
        ("feat_eng", ClinicalFeatureEngineer()),
        ("col_transform", ct),
    ])
    return pipe


def get_brfss_preprocessor() -> Tuple[Pipeline, List[str]]:
    """
    Constructs the preprocessor for CDC BRFSS Lifestyle & Health Indicators.
    """
    scale_cols = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income", "total_distress_days"]
    binary_cols = [
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack",
        "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
        "NoDocbcCost", "DiffWalk", "Sex", "lifestyle_risk_score",
        "vascular_comorbidity_count", "healthcare_access_barrier"
    ]

    ct = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scale_cols),
            ("pass", "passthrough", binary_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline([
        ("feat_eng", BRFSSLifestyleFeatureEngineer()),
        ("col_transform", ct),
    ])
    return pipe


def get_symptoms_preprocessor() -> Tuple[Pipeline, List[str]]:
    """
    Constructs the preprocessor for Early-Stage Symptoms (Sylhet Cohort).
    """
    scale_cols = ["Age", "classic_triad_count", "symptom_burden_score"]
    binary_cols = [
        "Gender", "Polyuria", "Polydipsia", "sudden weight loss", "weakness",
        "Polyphagia", "Genital thrush", "visual blurring", "Itching", "Irritability",
        "delayed healing", "partial paresis", "muscle stiffness", "Alopecia", "Obesity"
    ]

    ct = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), scale_cols),
            ("pass", "passthrough", binary_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipe = Pipeline([
        ("feat_eng", SymptomsFeatureEngineer()),
        ("col_transform", ct),
    ])
    return pipe


def prepare_clinical_data(
    deduplicate: bool = True, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """
    Prepares the Clinical dataset for modeling with strict stratified splitting.
    """
    df = load_clinical_data()
    if deduplicate:
        df = df.drop_duplicates().reset_index(drop=True)

    # Filter out rare "Other" gender to keep standard binary encoding clean if necessary,
    # or keep and handle gracefully
    df = df[df["gender"] != "Other"].reset_index(drop=True)

    X = df.drop(columns=["diabetes"])
    y = df["diabetes"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = get_clinical_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor


def prepare_brfss_data(
    balanced: bool = False, deduplicate: bool = True, test_size: float = 0.2, sample_n: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """
    Prepares the BRFSS Lifestyle dataset (Binary full or 50-50 balanced).
    """
    if balanced:
        df = load_brfss_5050_data()
    else:
        df = load_brfss_lifestyle_data()

    if deduplicate:
        df = df.drop_duplicates().reset_index(drop=True)

    if sample_n and sample_n < len(df):
        # Stratified sample if requested for fast benchmarking
        df, _ = train_test_split(
            df, train_size=sample_n, random_state=RANDOM_SEED, stratify=df["Diabetes_binary"]
        )
        df = df.reset_index(drop=True)

    X = df.drop(columns=["Diabetes_binary"])
    y = df["Diabetes_binary"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = get_brfss_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor


def prepare_brfss_012_data(
    deduplicate: bool = True, test_size: float = 0.2, sample_n: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """
    Prepares the BRFSS 3-class dataset (0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes).
    """
    df = load_brfss_012_data()
    if deduplicate:
        df = df.drop_duplicates().reset_index(drop=True)

    if sample_n and sample_n < len(df):
        df, _ = train_test_split(
            df, train_size=sample_n, random_state=RANDOM_SEED, stratify=df["Diabetes_012"]
        )
        df = df.reset_index(drop=True)

    X = df.drop(columns=["Diabetes_012"])
    y = df["Diabetes_012"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = get_brfss_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor


def prepare_symptoms_data(
    deduplicate: bool = False, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline]:
    """
    Prepares the Sylhet Early-Stage Symptoms dataset.
    Note: Sylhet has 520 rows where identical symptoms can be answered by different patients.
    Deduplicating blindly would drop half the cohort, so we maintain records while stratifying.
    """
    df = load_symptoms_data()
    if deduplicate:
        df = df.drop_duplicates().reset_index(drop=True)

    # Encode binary columns: 'Yes' -> 1, 'No' -> 0; 'Male' -> 1, 'Female' -> 0
    df_clean = df.copy()
    for col in df_clean.columns:
        if col == "Gender":
            df_clean[col] = (df_clean[col] == "Male").astype(int)
        elif col == "class":
            df_clean[col] = (df_clean[col] == "Positive").astype(int)
        elif col != "Age":
            df_clean[col] = (df_clean[col] == "Yes").astype(int)

    X = df_clean.drop(columns=["class"])
    y = df_clean["class"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = get_symptoms_preprocessor()
    return X_train, X_test, y_train, y_test, preprocessor
