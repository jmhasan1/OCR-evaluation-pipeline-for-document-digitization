from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_errors import (
    ERROR_CATEGORIES,
    analyze_errors,
    build_report,
    category_for_degradation,
    render_text_report,
)


def make_robustness_report() -> dict:
    return {
        "results": [
            {
                "document_id": "doc_001",
                "degradation_type": "character_substitution",
                "severity": 0.25,
                "metrics": {
                    "cer": 0.10,
                    "wer": 0.20,
                    "field_accuracy": 0.75,
                },
                "faithfulness": {"cer": 0.10},
                "changed_positions": 3,
            },
            {
                "document_id": "doc_001",
                "degradation_type": "whitespace_corruption",
                "severity": 0.50,
                "metrics": {
                    "cer": 0.05,
                    "wer": 0.40,
                    "field_accuracy": 0.80,
                },
                "faithfulness": {"cer": 0.05},
                "changed_positions": 5,
            },
            {
                "document_id": "doc_002",
                "degradation_type": "punctuation_corruption",
                "severity": 0.75,
                "metrics": {
                    "cer": 0.02,
                    "wer": 0.10,
                    "field_accuracy": 0.60,
                },
                "faithfulness": {"cer": 0.02},
                "changed_positions": 7,
            },
            {
                "document_id": "doc_003",
                "degradation_type": "identifier_corruption",
                "severity": 1.0,
                "metrics": {
                    "cer": 0.01,
                    "wer": 0.02,
                    "field_accuracy": 0.50,
                },
                "faithfulness": {"cer": 0.01},
                "changed_positions": 4,
            },
        ],
        "summary": [
            {
                "degradation_type": "character_substitution",
                "mean_cer": 0.10,
                "mean_wer": 0.20,
                "field_accuracy": 0.75,
            }
        ],
    }


def make_real_data_report() -> dict:
    return {
        "ground_truth_available": False,
        "documents": [
            {
                "source": "assignment.pdf",
                "page_count": 2,
                "text": {
                    "characters": 100,
                    "words": 20,
                },
                "confidence": {
                    "mean": 0.55,
                },
                "anomalies": {
                    "low_confidence_pages": [1],
                    "low_confidence_regions": 4,
                    "near_empty_pages": [],
                },
            }
        ],
    }


def test_degradation_categories_are_stable():
    assert (
        category_for_degradation("character_substitution")
        == "character_level"
    )
    assert (
        category_for_degradation("character_deletion")
        == "character_level"
    )
    assert (
        category_for_degradation("character_insertion")
        == "character_level"
    )
    assert category_for_degradation("whitespace_corruption") == "whitespace"
    assert (
        category_for_degradation("punctuation_corruption")
        == "punctuation"
    )
    assert category_for_degradation("identifier_corruption") == "identifier"


def test_all_error_categories_have_descriptions():
    expected = {
        "character_level",
        "whitespace",
        "punctuation",
        "identifier",
        "field_level",
        "faithfulness",
        "real_data_observation",
    }

    assert set(ERROR_CATEGORIES) == expected
    assert all(ERROR_CATEGORIES.values())


def test_report_contains_required_top_level_sections():
    report = build_report(
        make_robustness_report(),
        make_real_data_report(),
    )

    assert report["report_type"] == "ocr_error_analysis"
    assert report["version"] == "1.0"
    assert "scope" in report
    assert "categories" in report
    assert "robustness" in report
    assert "real_data" in report
    assert "evaluator_findings" in report
    assert "limitations" in report


def test_report_counts_robustness_cases():
    report = build_report(
        make_robustness_report(),
        make_real_data_report(),
    )

    assert report["robustness"]["case_count"] == 4
    assert report["robustness"]["summary_count"] == 1


def test_report_categorizes_robustness_cases():
    report = build_report(
        make_robustness_report(),
        make_real_data_report(),
    )

    counts = report["robustness"]["category_counts"]

    assert counts["character_level"] == 1
    assert counts["whitespace"] == 1
    assert counts["punctuation"] == 1
    assert counts["identifier"] == 1

def test_nested_degradation_metadata_is_categorized():
    robustness = make_robustness_report()

    robustness["results"][0].pop("degradation_type")
    robustness["results"][0]["degradation"] = {
        "type": "character_substitution"
    }

    report = build_report(
        robustness,
        make_real_data_report(),
    )

    assert report["robustness"]["category_counts"]["character_level"] == 1
    assert (
        report["robustness"]["degradation_case_counts"]
        ["character_substitution"]
        == 1
    )

def test_nested_degradation_severity_is_used_for_examples():
    robustness = make_robustness_report()

    case = robustness["results"][0]
    case.pop("severity")
    case["degradation"] = {
        "type": "character_substitution",
        "severity": 0.25,
    }

    report = build_report(
        robustness,
        make_real_data_report(),
    )

    examples = report["robustness"]["representative_examples"]

    assert any(
        example["degradation_type"] == "character_substitution"
        and example["severity"] == 0.25
        for example in examples
    )


def test_report_contains_representative_examples():
    report = build_report(
        make_robustness_report(),
        make_real_data_report(),
    )

    examples = report["robustness"]["representative_examples"]

    assert len(examples) == 4

    degradation_types = {
        example["degradation_type"]
        for example in examples
    }

    assert degradation_types == {
        "character_substitution",
        "whitespace_corruption",
        "punctuation_corruption",
        "identifier_corruption",
    }


def test_representative_example_contains_consequences():
    report = build_report(
        make_robustness_report(),
        make_real_data_report(),
    )

    example = report["robustness"]["representative_examples"][0]

    assert "cer" in example
    assert "wer" in example
    assert "field_accuracy" in example
    assert "faithfulness_cer" in example
    assert "changed_positions" in example
    assert example["interpretation"]


def test_real_data_is_explicitly_ground_truth_limited():
    report = build_report(
        make_robustness_report(),
        make_real_data_report(),
    )

    assert report["real_data"]["available"] is True
    assert report["real_data"]["ground_truth_available"] is False
    assert "ground truth" in report["real_data"]["limitation"].lower()

def test_real_data_low_confidence_region_count_is_preserved():
    real_data = make_real_data_report()

    report = build_report(
        make_robustness_report(),
        real_data,
    )

    document = report["real_data"]["documents"][0]

    assert document["low_confidence_regions"] == 4

def test_real_data_low_confidence_region_list_is_counted():
    real_data = make_real_data_report()
    real_data["documents"][0]["anomalies"]["low_confidence_regions"] = [
        {"page": 1},
        {"page": 1},
        {"page": 2},
        {"page": 2},
        {"page": 3},
    ]

    report = build_report(
        make_robustness_report(),
        real_data,
    )

    assert (
        report["real_data"]["documents"][0]["low_confidence_regions"]
        == 5
    )

def test_assignment_documents_are_not_claimed_as_ground_truth():
    report = build_report(
        make_robustness_report(),
        make_real_data_report(),
    )

    assert (
        report["scope"]["assignment_documents_used_as_ground_truth"]
        is False
    )


def test_report_has_production_relevant_findings():
    report = build_report(
        make_robustness_report(),
        make_real_data_report(),
    )

    findings = " ".join(
        finding["finding"]
        for finding in report["evaluator_findings"]
    )

    assert "Aggregate metrics are insufficient" in findings
    assert "Identifier errors require special attention" in findings
    assert "Critical fields provide consequence-oriented evidence" in findings


def test_text_report_is_human_readable():
    report = build_report(
        make_robustness_report(),
        make_real_data_report(),
    )

    text = render_text_report(report)

    assert "OCR ERROR ANALYSIS REPORT" in text
    assert "ERROR CATEGORIES" in text
    assert "REPRESENTATIVE ROBUSTNESS FAILURES" in text
    assert "REAL-DATA OBSERVATIONS" in text
    assert "EVALUATOR FINDINGS" in text
    assert "LIMITATIONS" in text
    assert "identifier_corruption" in text


def test_analysis_writes_json_and_text_reports(tmp_path: Path):
    robustness_path = tmp_path / "robustness.json"
    real_data_path = tmp_path / "real_data.json"
    json_output = tmp_path / "reports" / "error_analysis.json"
    text_output = tmp_path / "reports" / "error_analysis.txt"

    robustness_path.write_text(
        json.dumps(make_robustness_report()),
        encoding="utf-8",
    )

    real_data_path.write_text(
        json.dumps(make_real_data_report()),
        encoding="utf-8",
    )

    report = analyze_errors(
        robustness_path=robustness_path,
        real_data_path=real_data_path,
        json_output=json_output,
        text_output=text_output,
    )

    assert json_output.exists()
    assert text_output.exists()
    assert report["robustness"]["case_count"] == 4

    saved = json.loads(json_output.read_text(encoding="utf-8"))

    assert saved["report_type"] == "ocr_error_analysis"


def test_analysis_works_without_real_data_report(tmp_path: Path):
    robustness_path = tmp_path / "robustness.json"
    json_output = tmp_path / "error_analysis.json"
    text_output = tmp_path / "error_analysis.txt"

    robustness_path.write_text(
        json.dumps(make_robustness_report()),
        encoding="utf-8",
    )

    report = analyze_errors(
        robustness_path=robustness_path,
        real_data_path=tmp_path / "missing.json",
        json_output=json_output,
        text_output=text_output,
    )

    assert report["real_data"]["available"] is False
    assert json_output.exists()
    assert text_output.exists()


def test_report_is_json_serializable():
    report = build_report(
        make_robustness_report(),
        make_real_data_report(),
    )

    encoded = json.dumps(report, ensure_ascii=False)
    decoded = json.loads(encoded)

    assert decoded["report_type"] == "ocr_error_analysis"


def test_limitations_are_present():
    report = build_report(
        make_robustness_report(),
        make_real_data_report(),
    )

    assert len(report["limitations"]) >= 3


@pytest.mark.parametrize(
    "degradation,expected_category",
    [
        ("character_substitution", "character_level"),
        ("character_deletion", "character_level"),
        ("character_insertion", "character_level"),
        ("whitespace_corruption", "whitespace"),
        ("punctuation_corruption", "punctuation"),
        ("identifier_corruption", "identifier"),
    ],
)
def test_all_supported_degradations_are_categorized(
    degradation: str,
    expected_category: str,
):
    assert category_for_degradation(degradation) == expected_category