"""
Tests for Report Generator and Document Exports.
"""

from pathlib import Path
from src.config import REPORTS_DIR
from src.report_generator import (
    generate_research_report_markdown,
    generate_research_report_pdf,
)


def test_generate_research_report_markdown():
    content = generate_research_report_markdown()
    assert isinstance(content, str)
    assert len(content) > 500
    assert "AI-Based Diabetes Risk Prediction" in content
    assert "Ablation" in content
    md_file = REPORTS_DIR / "comprehensive_research_report.md"
    assert md_file.exists()


def test_generate_research_report_pdf():
    pdf_path = generate_research_report_pdf()
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000  # Non-empty valid PDF


def test_20_page_monograph_generation():
    import pypdf
    from src.monograph_generator import (
        generate_20_page_monograph_pdf,
        generate_matching_markdown_monograph,
    )

    pdf_out = REPORTS_DIR / "diabetes_risk_comprehensive_monograph_20_pages.pdf"
    md_out = REPORTS_DIR / "diabetes_risk_comprehensive_monograph_20_pages.md"

    generate_20_page_monograph_pdf(pdf_out)
    generate_matching_markdown_monograph(md_out)

    assert pdf_out.exists()
    assert md_out.exists()

    reader = pypdf.PdfReader(str(pdf_out))
    assert len(reader.pages) == 20
    assert md_out.stat().st_size > 5000
