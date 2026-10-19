"""
Data loading module with fallback resolution, caching, and integrity validation.
Loads raw datasets without mutating original files on disk.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import pandas as pd

from src.config import (
    DATASET_FILES,
    FAMILY_CLINICAL,
    FAMILY_LIFESTYLE,
    FAMILY_LIFESTYLE_BALANCED,
    FAMILY_MULTICLASS,
    FAMILY_SYMPTOMS,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)

_DATA_CACHE: Dict[str, pd.DataFrame] = {}


def resolve_file_path(filename: str) -> Path:
    """
    Resolves the absolute path of a dataset file, checking both data/raw/ and PROJECT_ROOT.
    """
    candidates = [
        RAW_DATA_DIR / filename,
        PROJECT_ROOT / filename,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    raise FileNotFoundError(
        f"Dataset file '{filename}' was not found in {RAW_DATA_DIR} or {PROJECT_ROOT}."
    )


def load_dataset_by_filename(filename: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Loads a dataset by filename with in-memory caching.
    """
    if use_cache and filename in _DATA_CACHE:
        return _DATA_CACHE[filename].copy()

    file_path = resolve_file_path(filename)
    df = pd.read_csv(file_path)
    if use_cache:
        _DATA_CACHE[filename] = df
    return df.copy()


def load_clinical_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Loads the clinical & laboratory measurement dataset.
    Features: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level, diabetes.
    """
    return load_dataset_by_filename(DATASET_FILES["diabetes_prediction_primary"], use_cache=use_cache)


def load_clinical_duplicate_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Loads the secondary diabetes prediction dataset file to verify duplicate status.
    """
    return load_dataset_by_filename(DATASET_FILES["diabetes_prediction_copy"], use_cache=use_cache)


def load_brfss_lifestyle_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Loads the full CDC BRFSS 2015 binary dataset (253,680 rows).
    Target: Diabetes_binary (0.0 vs 1.0)
    """
    return load_dataset_by_filename(DATASET_FILES["brfss_binary"], use_cache=use_cache)


def load_brfss_5050_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Loads the CDC BRFSS 2015 50-50 balanced dataset (70,692 rows).
    Target: Diabetes_binary (0.0 vs 1.0)
    """
    return load_dataset_by_filename(DATASET_FILES["brfss_5050"], use_cache=use_cache)


def load_brfss_012_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Loads the CDC BRFSS 2015 3-class dataset (253,680 rows).
    Target: Diabetes_012 (0.0 = Healthy, 1.0 = Prediabetes, 2.0 = Diabetes)
    """
    return load_dataset_by_filename(DATASET_FILES["brfss_012"], use_cache=use_cache)


def load_symptoms_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Loads the Sylhet hospital early-stage symptom dataset (520 rows).
    Target: class ('Positive' vs 'Negative')
    """
    return load_dataset_by_filename(DATASET_FILES["diabetes_symptoms"], use_cache=use_cache)


def load_all_datasets() -> Dict[str, pd.DataFrame]:
    """
    Loads all six dataset files into a dictionary keyed by filename.
    """
    datasets = {}
    for key, filename in DATASET_FILES.items():
        try:
            datasets[filename] = load_dataset_by_filename(filename)
        except FileNotFoundError as e:
            datasets[filename] = None
    return datasets
