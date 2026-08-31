"""Focused tests for the final OCR evaluation runner.

These tests validate orchestration and report contracts without rerunning
expensive OCR inference or the 90-case robustness experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_evaluation


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


def test_mean_returns_expected_average() -> None:
    assert run_evaluation._mean([1.0, 2.0, 3.0]) == 2.0


def test_mean_returns_none_for_empty_input() -> None:
    assert run_evaluation._mean([]) is None


def test_format_handles_none() -> None:
    assert run_evaluation._format(None) == "n/a"


def test_format_handles_float() -> None:
    assert run_evaluation._format(0.024936, digits=4) == "0.0249"


def test_git_revision_is_available() -> None:
    revision = run_evaluation._git_revision()

    assert revision is not None
    assert revision.strip()
    assert len(revision) >= 7


# ---------------------------------------------------------------------------
# Synthetic-document discovery
# ---------------------------------------------------------------------------


def test_discover_synthetic_documents_finds_three_documents() -> None:
    documents = run_evaluation.discover_synthetic_documents(
        run_evaluation.DEFAULT_SYNTHETIC_ROOT
    )

    assert len(documents) == 3
    assert [path.name for path in documents] == [
        "doc_001",
        "doc_002",
        "doc_003",
    ]


def test_discover_synthetic_documents_requires_existing_directory(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError):
        run_evaluation.discover_synthetic_documents(missing)


# ---------------------------------------------------------------------------
# Existing artifact validation
# ---------------------------------------------------------------------------


def test_validate_robustness_report_accepts_current_artifact() -> None:
    validated = run_evaluation.validate_robustness_report(
        run_evaluation.DEFAULT_ROBUSTNESS_REPORT
    )

    assert validated["experiment"]["case_count"] == 90
    assert validated["expected_case_count"] == 90


def test_validate_robustness_report_requires_existing_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing_robustness.json"

    with pytest.raises(FileNotFoundError):
        run_evaluation.validate_robustness_report(missing)


# ---------------------------------------------------------------------------
# Real-data boundary
# ---------------------------------------------------------------------------


def test_real_data_evidence_preserves_ground_truth_boundary() -> None:
    evidence = run_evaluation.load_real_data_evidence(
        run_evaluation.DEFAULT_REAL_DATA_REPORT
    )

    assert evidence is not None
    assert evidence["ground_truth_available"] is False
    assert evidence["document_count"] == 2


def test_missing_real_data_report_is_handled() -> None:
    missing = Path("this-file-does-not-exist.json")

    evidence = run_evaluation.load_real_data_evidence(missing)

    assert evidence is None


# ---------------------------------------------------------------------------
# Final report contract
# ---------------------------------------------------------------------------


def test_render_text_report_contains_major_sections() -> None:
    report = run_evaluation.load_json(
        run_evaluation.DEFAULT_JSON_OUTPUT
    )

    text = run_evaluation.render_text_report(report)

    required_sections = [
        "EVALUATION SCOPE",
        "SYNTHETIC BASELINE",
        "PER-DOCUMENT RESULTS",
        "CRITICAL-FIELD CONSEQUENCES",
        "ROBUSTNESS",
        "REAL-DATA OBSERVATIONS",
        "ERROR ANALYSIS",
        "EVALUATOR FINDINGS",
        "PRODUCTION-RELEVANT INTERPRETATION",
        "LIMITATIONS",
    ]

    for section in required_sections:
        assert section in text


def test_render_text_report_contains_git_revision() -> None:
    report = run_evaluation.load_json(
        run_evaluation.DEFAULT_JSON_OUTPUT
    )

    text = run_evaluation.render_text_report(report)

    revision = run_evaluation._git_revision()

    assert revision is not None
    assert f"Git revision: {revision}" in text


def test_render_text_report_contains_robustness_count() -> None:
    report = run_evaluation.load_json(
        run_evaluation.DEFAULT_JSON_OUTPUT
    )

    text = run_evaluation.render_text_report(report)

    assert "Cases evaluated: 90" in text


def test_render_text_report_uses_readable_title() -> None:
    report = run_evaluation.load_json(
        run_evaluation.DEFAULT_JSON_OUTPUT
    )

    text = run_evaluation.render_text_report(report)

    assert "OCR DOCUMENT EVALUATION — FINAL REPORT" in text
    assert "â€”" not in text

def test_render_text_report_formats_evaluator_findings_readably() -> None:
    report = run_evaluation.load_json(
        run_evaluation.DEFAULT_JSON_OUTPUT
    )

    text = run_evaluation.render_text_report(report)

    assert (
        "- Aggregate metrics are insufficient: "
        "Controlled degradation shows cases where relatively small "
        "text-level changes still affect field accuracy."
    ) in text

    assert (
        "- Identifier errors require special attention: "
        "Identifier corruption is explicitly measured separately "
        "from general character-level corruption."
    ) in text

    assert "{'finding':" not in text


# ---------------------------------------------------------------------------
# JSON/report writing
# ---------------------------------------------------------------------------


def test_write_json_creates_parent_directory(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "report.json"

    data = {
        "report_version": "1.0",
        "documents": 3,
    }

    run_evaluation.write_json(output, data)

    assert output.is_file()

    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == data


def test_load_json_requires_existing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        run_evaluation.load_json(missing)


def test_load_json_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError):
        run_evaluation.load_json(path)


# ---------------------------------------------------------------------------
# Current materialized final evidence
# ---------------------------------------------------------------------------


def test_current_final_evaluation_report_exists() -> None:
    report_path = run_evaluation.DEFAULT_JSON_OUTPUT

    assert report_path.is_file()


def test_current_final_evaluation_report_is_valid_json() -> None:
    report = run_evaluation.load_json(
        run_evaluation.DEFAULT_JSON_OUTPUT
    )

    assert isinstance(report, dict)


def test_current_final_evaluation_report_contains_expected_evidence() -> None:
    report = run_evaluation.load_json(
        run_evaluation.DEFAULT_JSON_OUTPUT
    )

    synthetic = report["synthetic_baseline"]
    robustness = report["robustness"]
    real_data = report["real_data"]
    error_analysis = report["error_analysis"]

    assert synthetic["summary"]["document_count"] == 3
    assert robustness["case_count"] == 90
    assert real_data["ground_truth_available"] is False
    assert error_analysis["report"]["robustness"]["case_count"] == 90


def test_current_final_text_report_exists() -> None:
    assert run_evaluation.DEFAULT_TEXT_OUTPUT.is_file()


def test_current_final_text_report_is_nonempty() -> None:
    text = run_evaluation.DEFAULT_TEXT_OUTPUT.read_text(
        encoding="utf-8"
    )

    assert len(text.strip()) > 1000
    assert "OCR DOCUMENT EVALUATION" in text