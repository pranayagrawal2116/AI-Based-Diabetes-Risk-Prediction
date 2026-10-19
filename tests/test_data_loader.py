"""
Tests for Data Loader Module.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.data_loader import (
    resolve_file_path,
    load_clinical_data,
    load_clinical_duplicate_data,
    load_brfss_lifestyle_data,
    load_brfss_5050_data,
    load_brfss_012_data,
    load_symptoms_data,
    load_all_datasets,
)


def test_resolve_file_path():
    path = resolve_file_path("diabetes_prediction_dataset.csv")
    assert path.exists()
    assert path.is_file()


def test_resolve_file_path_missing():
    with pytest.raises(FileNotFoundError):
        resolve_file_path("non_existent_dataset_12345.csv")


def test_load_clinical_data():
    df = load_clinical_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (100000, 9)
    assert "diabetes" in df.columns
    assert "HbA1c_level" in df.columns
    assert "blood_glucose_level" in df.columns


def test_load_clinical_duplicate_data():
    df1 = load_clinical_data()
    df2 = load_clinical_duplicate_data()
    assert df1.shape == df2.shape
    assert list(df1.columns) == list(df2.columns)


def test_load_brfss_lifestyle_data():
    df = load_brfss_lifestyle_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (253680, 22)
    assert "Diabetes_binary" in df.columns
    assert "PhysActivity" in df.columns
    assert "Fruits" in df.columns


def test_load_symptoms_data():
    df = load_symptoms_data()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (520, 17)
    assert "class" in df.columns
    assert "Polyuria" in df.columns


def test_load_all_datasets():
    datasets = load_all_datasets()
    assert len(datasets) >= 5
    for name, df in datasets.items():
        assert df is not None
        assert len(df) > 0
