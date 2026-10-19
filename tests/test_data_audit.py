"""
Tests for Data Audit and Cryptographic Integrity Module.
"""

from pathlib import Path
import pytest
import pandas as pd

from src.data_audit import (
    compute_file_md5,
    audit_single_dataset,
    audit_cross_dataset_integrity,
    run_full_data_audit,
)
from src.data_loader import resolve_file_path, load_all_datasets


def test_compute_file_md5():
    p1 = resolve_file_path("diabetes_prediction_dataset.csv")
    md5 = compute_file_md5(p1)
    assert isinstance(md5, str)
    assert len(md5) == 32
    assert md5 == "cb243b99d81d96e9aeb08944ce9ac066"


def test_audit_single_dataset():
    p = resolve_file_path("diabetes_data_upload.csv")
    audit = audit_single_dataset(p)
    assert audit["n_rows"] == 520
    assert audit["n_cols"] == 17
    assert audit["total_nulls"] == 0
    assert "class" in audit["targets"]


def test_clinical_duplicate_verification():
    dfs = load_all_datasets()
    cross = audit_cross_dataset_integrity(dfs)
    assert "clinical_duplicate_check" in cross
    dup_info = cross["clinical_duplicate_check"]
    assert dup_info["is_identical"] is True
    assert dup_info["file_1_md5"] == dup_info["file_2_md5"]


def test_brfss_binary_vs_012_relationship():
    dfs = load_all_datasets()
    cross = audit_cross_dataset_integrity(dfs)
    assert "brfss_binary_vs_012_check" in cross
    rel = cross["brfss_binary_vs_012_check"]
    assert rel["same_row_count"] is True
    assert rel["features_identical"] is True
    assert rel["target_mapping_confirmed"] is True


def test_run_full_data_audit():
    report = run_full_data_audit()
    assert "dataset_audits" in report
    assert "cross_dataset_integrity" in report
    assert len(report["dataset_audits"]) >= 5
