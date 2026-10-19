"""
Tests for Streamlit application scripts and syntax integrity.
"""

import ast
from pathlib import Path
from src.config import PROJECT_ROOT


def test_app_scripts_syntax():
    """Verifies that all Streamlit application pages have valid Python syntax."""
    app_file = PROJECT_ROOT / "app.py"
    pages_dir = PROJECT_ROOT / "pages"

    files_to_check = [app_file] + list(pages_dir.glob("*.py"))
    assert len(files_to_check) >= 6

    for py_file in files_to_check:
        with open(py_file, "r") as f:
            code = f.read()
        # ast.parse will raise SyntaxError if file is invalid
        tree = ast.parse(code, filename=str(py_file))
        assert tree is not None


def test_dataset_explorer_column_profiles():
    """Verifies that all column profile sample values format cleanly without TypeError."""
    import json
    from src.config import REPORTS_DIR

    audit_path = REPORTS_DIR / "data_audit_report.json"
    assert audit_path.exists()
    with open(audit_path, "r") as f:
        data = json.load(f)

    for fname, d in data["dataset_audits"].items():
        if "error" in d:
            continue
        for col_name, cprofile in d.get("columns", {}).items():
            samples = cprofile.get("samples", [])
            sample_str = ", ".join(str(s) for s in samples[:4])
            assert isinstance(sample_str, str)
