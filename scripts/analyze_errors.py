from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROBUSTNESS_REPORT = (
    ROOT / "outputs" / "robustness" / "robustness_results.json"
)
DEFAULT_REAL_DATA_REPORT = (
    ROOT / "outputs" / "assignment" / "real_data_ocr_quality_report.json"
)

DEFAULT_JSON_OUTPUT = ROOT / "outputs" / "reports" / "error_analysis.json"
DEFAULT_TEXT_OUTPUT = ROOT / "outputs" / "reports" / "error_analysis.txt"


ERROR_CATEGORIES = {
    "character_level": "Character substitutions, insertions, and deletions",
    "whitespace": "Whitespace corruption affecting token boundaries",
    "punctuation": "Punctuation corruption affecting text or identifiers",
    "identifier": "Corruption affecting structured identifiers",
    "field_level": "Critical-field extraction or evaluation failures",
    "faithfulness": "Downstream output differences from source OCR",
    "real_data_observation": "Observable real-data OCR quality signals",
}


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")

    return data


def find_case_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Extract robustness cases from the existing robustness report.

    The function intentionally tolerates the report being wrapped in a
    top-level `results` field or exposing the cases directly.
    """
    results = report.get("results")

    if isinstance(results, list):
        return [item for item in results if isinstance(item, dict)]

    cases = report.get("cases")

    if isinstance(cases, list):
        return [item for item in cases if isinstance(item, dict)]

    return []


def find_summary(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract aggregate degradation summaries from the robustness report."""
    summary = report.get("summary")

    if isinstance(summary, list):
        return [item for item in summary if isinstance(item, dict)]

    return []

def extract_degradation_type(case: dict[str, Any]) -> str:
    """Extract a normalized degradation type from a robustness case."""
    value = case.get("degradation_type")

    if isinstance(value, str) and value:
        return value

    degradation = case.get("degradation")

    if isinstance(degradation, str) and degradation:
        return degradation

    if isinstance(degradation, dict):
        for key in (
            "type",
            "degradation_type",
            "name",
            "method",
        ):
            nested_value = degradation.get(key)
            if isinstance(nested_value, str) and nested_value:
                return nested_value

    return "unknown"

def extract_severity(case: dict[str, Any]) -> float:
    """Extract degradation severity from a robustness case."""
    value = case.get("severity")

    if isinstance(value, (int, float)):
        return float(value)

    degradation = case.get("degradation")

    if isinstance(degradation, dict):
        for key in ("severity", "level"):
            nested_value = degradation.get(key)
            if isinstance(nested_value, (int, float)):
                return float(nested_value)

    return 0.0

def category_for_degradation(degradation_type: str) -> str:
    """Map a controlled degradation to an evaluator-facing category."""
    if degradation_type in {
        "character_substitution",
        "character_deletion",
        "character_insertion",
    }:
        return "character_level"

    if degradation_type == "whitespace_corruption":
        return "whitespace"

    if degradation_type == "punctuation_corruption":
        return "punctuation"

    if degradation_type == "identifier_corruption":
        return "identifier"

    return "character_level"


def extract_metric(case: dict[str, Any], name: str) -> float | None:
    """
    Read a metric from a robustness case.

    Supports both the current nested metric layout and a flat fallback
    so the reporting script is resilient to small report-schema changes.
    """
    metrics = case.get("metrics")

    if isinstance(metrics, dict):
        value = metrics.get(name)
        if isinstance(value, (int, float)):
            return float(value)

    value = case.get(name)

    if isinstance(value, (int, float)):
        return float(value)

    return None


def extract_field_accuracy(case: dict[str, Any]) -> float | None:
    """Extract field accuracy from a robustness case."""
    for key in ("field_accuracy", "critical_field_accuracy"):
        value = case.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    metrics = case.get("metrics")
    if isinstance(metrics, dict):
        for key in ("field_accuracy", "critical_field_accuracy"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    field_result = case.get("field_evaluation")

    if isinstance(field_result, dict):
        value = field_result.get("accuracy")
        if isinstance(value, (int, float)):
            return float(value)

    return None


def extract_faithfulness_cer(case: dict[str, Any]) -> float | None:
    """Extract faithfulness CER from a robustness case."""
    for key in ("faithfulness_cer", "faithfulness_error"):
        value = case.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    faithfulness = case.get("faithfulness")

    if isinstance(faithfulness, dict):
        for key in ("cer", "faithfulness_cer", "error_rate"):
            value = faithfulness.get(key)
            if isinstance(value, (int, float)):
                return float(value)

    return None


def extract_changed_positions(case: dict[str, Any]) -> int | None:
    """Extract the number of changed positions from a case."""
    for key in ("changed_positions", "change_count"):
        value = case.get(key)
        if isinstance(value, int):
            return value

    changes = case.get("changes")

    if isinstance(changes, list):
        return len(changes)

    return None


def build_robustness_analysis(report: dict[str, Any]) -> dict[str, Any]:
    """Build evaluator-facing robustness error analysis."""
    cases = find_case_results(report)
    summary = find_summary(report)

    category_counts: Counter[str] = Counter()
    category_cases: dict[str, int] = Counter()

    for case in cases:

        degradation = extract_degradation_type(case)

        category = category_for_degradation(degradation)
        category_counts[category] += 1
        category_cases[degradation] += 1

    representative_examples: list[dict[str, Any]] = []

    nonzero_cases = [
        case
        for case in cases
        if extract_severity(case) > 0
    ]

    # Prefer one representative case per degradation type.
    selected: set[str] = set()

    for case in nonzero_cases:

        degradation = extract_degradation_type(case)

        if degradation in selected:
            continue

        selected.add(degradation)

        representative_examples.append(
            {
                "category": category_for_degradation(degradation),
                "degradation_type": degradation,
                "document_id": case.get("document_id"),
                "severity": extract_severity(case),
                "cer": extract_metric(case, "cer"),
                "wer": extract_metric(case, "wer"),
                "field_accuracy": extract_field_accuracy(case),
                "faithfulness_cer": extract_faithfulness_cer(case),
                "changed_positions": extract_changed_positions(case),
                "interpretation": interpretation_for_case(
                    degradation,
                    extract_metric(case, "cer"),
                    extract_metric(case, "wer"),
                    extract_field_accuracy(case),
                ),
            }
        )

    return {
        "case_count": len(cases),
        "summary_count": len(summary),
        "category_counts": dict(category_counts),
        "degradation_case_counts": dict(category_cases),
        "representative_examples": representative_examples,
        "aggregate_summary": summary,
    }


def interpretation_for_case(
    degradation: str,
    cer: float | None,
    wer: float | None,
    field_accuracy: float | None,
) -> str:
    """Generate a concise interpretation without inventing causal claims."""
    if degradation == "whitespace_corruption":
        return (
            "Whitespace corruption can produce a relatively modest CER while "
            "causing larger word-level disruption."
        )

    if degradation == "punctuation_corruption":
        return (
            "Punctuation changes may produce relatively small text-level "
            "error while still affecting structured field evaluation."
        )

    if degradation == "identifier_corruption":
        return (
            "Identifier corruption is especially important because small "
            "text changes can affect structured document information."
        )

    if degradation in {
        "character_substitution",
        "character_deletion",
        "character_insertion",
    }:
        return (
            "Character-level corruption can propagate from text recognition "
            "errors into downstream field evaluation."
        )

    return "Controlled corruption produced measurable evaluation changes."

def extract_low_confidence_region_count(
    anomalies: dict[str, Any],
) -> int:
    """Extract a low-confidence region count from anomaly metadata."""
    for key in (
        "low_confidence_regions",
        "low_confidence_region_count",
        "low_confidence_regions_count",
    ):
        value = anomalies.get(key)

        if isinstance(value, int):
            return value

        if isinstance(value, list):
            return len(value)

    return 0

def build_real_data_analysis(
    report: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Summarize real-data observations.

    No claim of transcription accuracy is made because assignment ground
    truth is unavailable.
    """
    if not report:
        return {
            "available": False,
            "ground_truth_available": False,
            "documents": [],
        }

    documents = report.get("documents")

    if not isinstance(documents, list):
        documents = []

    observations = []

    for document in documents:
        if not isinstance(document, dict):
            continue

        confidence = document.get("confidence", {})
        anomalies = document.get("anomalies", {})
        text = document.get("text", {})

        observations.append(
            {
                "source": document.get("source"),
                "page_count": document.get("page_count"),
                "characters": text.get("characters"),
                "words": text.get("words"),
                "mean_confidence": confidence.get("mean"),
                "low_confidence_pages": anomalies.get(
                    "low_confidence_pages", []
                ),
                "low_confidence_regions": extract_low_confidence_region_count(
                    anomalies
                ),
                "near_empty_pages": anomalies.get("near_empty_pages", []),
            }
        )

    return {
        "available": True,
        "ground_truth_available": bool(
            report.get("ground_truth_available", False)
        ),
        "documents": observations,
        "limitation": (
            "Assignment documents do not have authoritative ground truth "
            "in this evaluation, so these observations must not be treated "
            "as CER/WER or field-accuracy measurements."
        ),
    }


def build_report(
    robustness_report: dict[str, Any],
    real_data_report: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the consolidated machine-readable error-analysis report."""
    robustness = build_robustness_analysis(robustness_report)
    real_data = build_real_data_analysis(real_data_report)

    category_descriptions = {
        name: description
        for name, description in ERROR_CATEGORIES.items()
    }

    return {
        "report_type": "ocr_error_analysis",
        "version": "1.0",
        "scope": {
            "synthetic_robustness": True,
            "real_data_observations": real_data["available"],
            "assignment_documents_used_as_ground_truth": False,
        },
        "categories": category_descriptions,
        "robustness": robustness,
        "real_data": real_data,
        "evaluator_findings": [
            {
                "finding": "Aggregate metrics are insufficient",
                "evidence": (
                    "Controlled degradation shows cases where relatively "
                    "small text-level changes still affect field accuracy."
                ),
            },
            {
                "finding": "Identifier errors require special attention",
                "evidence": (
                    "Identifier corruption is explicitly measured separately "
                    "from general character-level corruption."
                ),
            },
            {
                "finding": "Real-data confidence signals require context",
                "evidence": (
                    "Assignment OCR outputs show low-confidence regions, "
                    "but lack authoritative reference text for accuracy claims."
                ),
            },
            {
                "finding": "Critical fields provide consequence-oriented evidence",
                "evidence": (
                    "Field-level evaluation exposes structured-information "
                    "impact that CER/WER alone cannot describe."
                ),
            },
        ],
        "limitations": [
            (
                "Synthetic robustness results measure controlled corruption "
                "of known development references, not real OCR error rates."
            ),
            (
                "Assignment documents are evaluated using observable OCR "
                "quality signals because authoritative ground truth is unavailable."
            ),
            (
                "Representative examples are selected from existing evaluation "
                "artifacts and do not constitute a statistically representative "
                "sample of all possible document failures."
            ),
        ],
    }


def format_float(value: Any) -> str:
    if value is None:
        return "n/a"

    if isinstance(value, (int, float)):
        return f"{value:.4f}"

    return str(value)


def render_text_report(report: dict[str, Any]) -> str:
    """Render the consolidated report as human-readable text."""
    lines: list[str] = []

    lines.extend(
        [
            "OCR ERROR ANALYSIS REPORT",
            "=" * 80,
            "",
            "This report consolidates evidence from the existing OCR evaluation",
            "layers. It does not introduce a new accuracy metric.",
            "",
            "SCOPE",
            "-" * 80,
            f"Synthetic robustness cases: "
            f"{report['robustness']['case_count']}",
            f"Real-data observations available: "
            f"{report['real_data']['available']}",
            f"Assignment documents used as ground truth: "
            f"{report['scope']['assignment_documents_used_as_ground_truth']}",
            "",
            "ERROR CATEGORIES",
            "-" * 80,
        ]
    )

    for name, description in report["categories"].items():
        count = report["robustness"]["category_counts"].get(name, 0)
        lines.append(f"{name}: {description} | robustness cases={count}")

    lines.extend(
        [
            "",
            "REPRESENTATIVE ROBUSTNESS FAILURES",
            "-" * 80,
        ]
    )

    for example in report["robustness"]["representative_examples"]:
        lines.extend(
            [
                f"Degradation: {example['degradation_type']}",
                f"Category: {example['category']}",
                f"Document: {example.get('document_id')}",
                f"Severity: {format_float(example.get('severity'))}",
                f"CER: {format_float(example.get('cer'))}",
                f"WER: {format_float(example.get('wer'))}",
                "Field accuracy: "
                f"{format_float(example.get('field_accuracy'))}",
                "Faithfulness CER: "
                f"{format_float(example.get('faithfulness_cer'))}",
                "Changed positions: "
                f"{format_float(example.get('changed_positions'))}",
                f"Interpretation: {example['interpretation']}",
                "",
            ]
        )

    lines.extend(
        [
            "REAL-DATA OBSERVATIONS",
            "-" * 80,
        ]
    )

    if not report["real_data"]["documents"]:
        lines.append("No real-data report available.")
    else:
        for document in report["real_data"]["documents"]:
            lines.extend(
                [
                    f"Document: {document.get('source')}",
                    f"Pages: {document.get('page_count')}",
                    f"Characters: {document.get('characters')}",
                    f"Words: {document.get('words')}",
                    "Mean confidence: "
                    f"{format_float(document.get('mean_confidence'))}",
                    "Low-confidence pages: "
                    f"{document.get('low_confidence_pages')}",
                    "Low-confidence regions: "
                    f"{document.get('low_confidence_regions')}",
                    "Near-empty pages: "
                    f"{document.get('near_empty_pages')}",
                    "",
                ]
            )

    lines.extend(
        [
            "EVALUATOR FINDINGS",
            "-" * 80,
        ]
    )

    for finding in report["evaluator_findings"]:
        lines.append(f"- {finding['finding']}: {finding['evidence']}")

    lines.extend(
        [
            "",
            "LIMITATIONS",
            "-" * 80,
        ]
    )

    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")

    lines.append("")

    return "\n".join(lines)


def analyze_errors(
    robustness_path: Path = DEFAULT_ROBUSTNESS_REPORT,
    real_data_path: Path = DEFAULT_REAL_DATA_REPORT,
    json_output: Path = DEFAULT_JSON_OUTPUT,
    text_output: Path = DEFAULT_TEXT_OUTPUT,
) -> dict[str, Any]:
    """Load existing reports and write the consolidated analysis."""
    robustness_report = load_json(robustness_path)

    real_data_report = None
    if real_data_path.exists():
        real_data_report = load_json(real_data_path)

    report = build_report(
        robustness_report=robustness_report,
        real_data_report=real_data_report,
    )

    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)

    json_output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    text_output.write_text(
        render_text_report(report),
        encoding="utf-8",
    )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build consolidated OCR error-analysis reports."
    )

    parser.add_argument(
        "--robustness-report",
        type=Path,
        default=DEFAULT_ROBUSTNESS_REPORT,
    )

    parser.add_argument(
        "--real-data-report",
        type=Path,
        default=DEFAULT_REAL_DATA_REPORT,
    )

    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
    )

    parser.add_argument(
        "--text-output",
        type=Path,
        default=DEFAULT_TEXT_OUTPUT,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report = analyze_errors(
        robustness_path=args.robustness_report,
        real_data_path=args.real_data_report,
        json_output=args.json_output,
        text_output=args.text_output,
    )

    print(
        "Error analysis generated:",
        report["robustness"]["case_count"],
        "robustness cases",
    )
    print(f"JSON report: {args.json_output}")
    print(f"Text report: {args.text_output}")


if __name__ == "__main__":
    main()