from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocr_eval.degradation import DEGRADATIONS
from scripts.run_robustness_experiment import (
    DEFAULT_DEGRADATION_TYPES,
    DEFAULT_SEVERITIES,
    _build_summary,
    _format_text_report,
    run_experiment,
)


ROOT = Path(__file__).resolve().parents[1]

SYNTHETIC_ROOT = (
    ROOT
    / "data"
    / "development"
    / "synthetic_documents"
)


@pytest.fixture(scope="module")
def experiment_report():
    """Run the expensive 90-case experiment once for this test module."""

    return run_experiment(
        synthetic_root=SYNTHETIC_ROOT
    )


def test_default_degradation_types_match_framework():
    assert (
        DEFAULT_DEGRADATION_TYPES
        == tuple(DEGRADATIONS.keys())
    )


def test_experiment_discovers_three_synthetic_documents(
    experiment_report,
):
    assert experiment_report["experiment"]["documents"] == [
        "doc_001",
        "doc_002",
        "doc_003",
    ]


def test_experiment_contains_expected_90_case_matrix(
    experiment_report,
):
    expected = (
        3
        * len(DEFAULT_DEGRADATION_TYPES)
        * len(DEFAULT_SEVERITIES)
    )

    assert expected == 90
    assert len(experiment_report["results"]) == expected
    assert (
        experiment_report["experiment"]["case_count"]
        == expected
    )


def test_all_degradation_types_are_present(
    experiment_report,
):
    actual = {
        result["degradation"]["type"]
        for result in experiment_report["results"]
    }

    assert (
        actual
        == set(DEFAULT_DEGRADATION_TYPES)
    )


def test_all_severities_are_present(
    experiment_report,
):
    actual = {
        result["degradation"]["severity"]
        for result in experiment_report["results"]
    }

    assert (
        actual
        == set(DEFAULT_SEVERITIES)
    )


def test_zero_severity_is_a_perfect_control(
    experiment_report,
):
    controls = [
        result
        for result in experiment_report["results"]
        if result["degradation"]["severity"]
        == 0.0
    ]

    expected_controls = (
        3 * len(DEFAULT_DEGRADATION_TYPES)
    )

    assert len(controls) == expected_controls

    for result in controls:
        assert result["text_metrics"]["cer"] == 0.0
        assert result["text_metrics"]["wer"] == 0.0

        assert (
            result["field_evaluation"]["accuracy"]
            == 1.0
        )

        assert (
            result["faithfulness"].get("cer")
            == 0.0
        )

        assert (
            result["degradation"][
                "changed_positions"
            ]
            == []
        )


def test_nonzero_degradation_changes_text(
    experiment_report,
):
    degraded = [
        result
        for result in experiment_report["results"]
        if result["degradation"]["severity"]
        > 0.0
    ]

    assert degraded

    assert any(
        result["text"]["reference"]
        != result["text"]["degraded"]
        for result in degraded
    )


def test_nonzero_degradation_has_positive_text_error(
    experiment_report,
):
    degraded = [
        result
        for result in experiment_report["results"]
        if result["degradation"]["severity"]
        > 0.0
    ]

    assert any(
        result["text_metrics"]["cer"] > 0.0
        for result in degraded
    )


def test_changed_positions_are_recorded(
    experiment_report,
):
    degraded = [
        result
        for result in experiment_report["results"]
        if result["degradation"]["severity"]
        > 0.0
    ]

    assert any(
        result["degradation"][
            "changed_positions"
        ]
        for result in degraded
    )


def test_every_result_contains_required_layers(
    experiment_report,
):
    required_keys = {
        "document_id",
        "degradation",
        "text_metrics",
        "field_evaluation",
        "faithfulness",
        "text",
    }

    for result in experiment_report["results"]:
        assert required_keys <= result.keys()

        assert {
            "type",
            "severity",
            "changed_positions",
        } <= result["degradation"].keys()

        assert "cer" in result["text_metrics"]
        assert "wer" in result["text_metrics"]

        assert (
            "accuracy"
            in result["field_evaluation"]
        )

        assert "reference" in result["text"]
        assert "degraded" in result["text"]


# def test_experiment_is_deterministic():
#     first = run_experiment(
#         synthetic_root=SYNTHETIC_ROOT
#     )

#     second = run_experiment(
#         synthetic_root=SYNTHETIC_ROOT
#     )

#     assert first == second


def test_summary_matches_experiment_matrix(
    experiment_report,
):
    summary = _build_summary(
        experiment_report
    )

    assert summary["case_count"] == 90
    assert (
        summary["expected_case_count"]
        == 90
    )

    assert (
        summary["control_cases"]["count"]
        == 18
    )

    assert (
        summary["control_cases"][
            "all_cer_zero"
        ]
    )

    assert (
        summary["control_cases"][
            "all_wer_zero"
        ]
    )

    assert (
        summary["control_cases"][
            "all_fields_perfect"
        ]
    )

    assert (
        summary["control_cases"][
            "all_faithfulness_cer_zero"
        ]
    )


def test_summary_contains_every_degradation_type(
    experiment_report,
):
    summary = _build_summary(
        experiment_report
    )

    assert (
        set(summary["by_degradation"])
        == set(DEFAULT_DEGRADATION_TYPES)
    )

    for values in summary[
        "by_degradation"
    ].values():
        assert values["cases"] == 15

        assert (
            values["mean_cer"]
            is not None
        )

        assert (
            values["mean_wer"]
            is not None
        )

        assert (
            values["mean_field_accuracy"]
            is not None
        )

        assert (
            values["mean_changed_positions"]
            is not None
        )


def test_each_degradation_has_five_severity_cases_per_document(
    experiment_report,
):
    for degradation_type in (
        DEFAULT_DEGRADATION_TYPES
    ):
        cases = [
            result
            for result in experiment_report["results"]
            if result["degradation"]["type"]
            == degradation_type
        ]

        assert len(cases) == 15

        severities = {
            result["degradation"]["severity"]
            for result in cases
        }

        assert (
            severities
            == set(DEFAULT_SEVERITIES)
        )


def test_text_report_is_human_readable(
    experiment_report,
):
    report = dict(experiment_report)
    report["summary"] = _build_summary(
        report
    )

    text = _format_text_report(report)

    assert (
        "SYNTHETIC OCR ROBUSTNESS REPORT"
        in text
    )

    assert "CONTROL CASES" in text

    assert (
        "AGGREGATE IMPACT BY DEGRADATION"
        in text
    )

    assert "INTERPRETATION" in text

    for degradation_type in (
        DEFAULT_DEGRADATION_TYPES
    ):
        assert degradation_type in text


def test_results_are_json_serializable(
    experiment_report,
):
    serialized = json.dumps(
        experiment_report,
        ensure_ascii=False,
    )

    restored = json.loads(
        serialized
    )

    assert (
        len(restored["results"])
        == 90
    )


def test_assignment_documents_are_not_used(
    experiment_report,
):
    document_ids = {
        result["document_id"]
        for result in experiment_report["results"]
    }

    assert document_ids == {
        "doc_001",
        "doc_002",
        "doc_003",
    }

    for result in experiment_report["results"]:
        assert (
            "assignment"
            not in result["text"]["reference"]
        )