"""Run the complete OCR evaluation pipeline and consolidate final evidence.

This runner intentionally orchestrates the existing evaluation layers rather
than reimplementing their metrics or modifying the core evaluation
architecture.

Default behavior:
    1. Load existing synthetic OCR outputs.
    2. Evaluate CER/WER against synthetic ground truth.
    3. Extract and evaluate critical fields.
    4. Run the existing faithfulness comparison.
    5. Load the existing robustness experiment artifact.
    6. Load the existing real-data quality artifact when available.
    7. Reuse the existing error-analysis layer.
    8. Produce one consolidated JSON + text report.

Expensive operations such as OCR inference and the 90-case robustness
experiment are not rerun by default. They have dedicated scripts and their
materialized reports are treated as evaluation artifacts.

Use --run-robustness when a fresh robustness experiment is explicitly wanted.
Use --require-real-data when assignment OCR evidence must be present.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

DEFAULT_SYNTHETIC_ROOT = (
    ROOT / "data" / "development" / "synthetic_documents"
)

DEFAULT_RAW_OUTPUT_DIR = ROOT / "outputs" / "raw"

DEFAULT_ROBUSTNESS_REPORT = (
    ROOT / "outputs" / "robustness" / "robustness_results.json"
)

DEFAULT_REAL_DATA_REPORT = (
    ROOT / "outputs" / "assignment" / "real_data_ocr_quality_report.json"
)

DEFAULT_ERROR_ANALYSIS_JSON = (
    ROOT / "outputs" / "reports" / "error_analysis.json"
)

DEFAULT_ERROR_ANALYSIS_TEXT = (
    ROOT / "outputs" / "reports" / "error_analysis.txt"
)

DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "evaluation"

DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "evaluation_report.json"
DEFAULT_TEXT_OUTPUT = DEFAULT_OUTPUT_DIR / "evaluation_report.txt"


# ---------------------------------------------------------------------------
# Existing project interfaces
# ---------------------------------------------------------------------------

from ocr_eval.critical_fields import evaluate_fields
from ocr_eval.extraction import extract_critical_fields
from ocr_eval.faithfulness import compare_faithfulness
from ocr_eval.metrics import text_metrics


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")

    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON report with stable UTF-8 formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _mean(values: list[float]) -> float | None:
    """Return the arithmetic mean for a non-empty list."""
    if not values:
        return None

    return sum(values) / len(values)


def _format(value: Any, digits: int = 4) -> str:
    """Format values for the human-readable report."""
    if value is None:
        return "n/a"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def _git_revision() -> str | None:
    """Return the current Git revision when available."""
    try:
        result = subprocess.run(
            [
                "git",
                "rev-parse",
                "--short",
                "HEAD",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        revision = result.stdout.strip()

        return revision or None

    except OSError:
        return None


# ---------------------------------------------------------------------------
# Synthetic OCR-output discovery
# ---------------------------------------------------------------------------


def discover_synthetic_documents(
    synthetic_root: Path,
) -> list[Path]:
    """Discover synthetic documents with authoritative ground truth."""
    if not synthetic_root.is_dir():
        raise FileNotFoundError(
            f"Synthetic document directory does not exist: "
            f"{synthetic_root}"
        )

    documents = sorted(
        path
        for path in synthetic_root.iterdir()
        if (
            path.is_dir()
            and (
                path / "ground_truth" / "full_text.txt"
            ).is_file()
            and (
                path / "ground_truth" / "fields.json"
            ).is_file()
        )
    )

    if not documents:
        raise FileNotFoundError(
            "No synthetic documents with complete ground truth "
            f"were found under {synthetic_root}"
        )

    return documents


def _load_ground_truth(
    document_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """Load synthetic reference text and fields."""
    ground_truth = document_dir / "ground_truth"

    text_path = ground_truth / "full_text.txt"
    fields_path = ground_truth / "fields.json"

    reference_text = text_path.read_text(
        encoding="utf-8"
    )

    metadata = load_json(fields_path)

    reference_fields = metadata.get("fields")

    if not isinstance(reference_fields, dict):
        raise ValueError(
            f"Ground-truth file does not contain a "
            f"'fields' object: {fields_path}"
        )

    return reference_text, reference_fields


def _candidate_raw_outputs(
    document_id: str,
) -> list[Path]:
    """Return preferred existing OCR output paths for a document.

    Preference order deliberately favors the latest explicit development
    outputs while retaining compatibility with earlier benchmark/output
    naming conventions.
    """
    return [
        DEFAULT_RAW_OUTPUT_DIR
        / f"{document_id}_auto_final.json",
        DEFAULT_RAW_OUTPUT_DIR
        / f"{document_id}_gpu.json",
        DEFAULT_RAW_OUTPUT_DIR
        / f"{document_id}_auto.json",
        DEFAULT_RAW_OUTPUT_DIR
        / f"{document_id}_cpu.json",
    ]


def find_ocr_output(
    document_id: str,
    raw_output_dir: Path,
) -> Path | None:
    """Find the preferred existing OCR output for a document."""
    candidates = [
        raw_output_dir / f"{document_id}_auto_final.json",
        raw_output_dir / f"{document_id}_gpu.json",
        raw_output_dir / f"{document_id}_auto.json",
        raw_output_dir / f"{document_id}_cpu.json",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


# ---------------------------------------------------------------------------
# Synthetic baseline evaluation
# ---------------------------------------------------------------------------


def evaluate_synthetic_document(
    *,
    document_dir: Path,
    ocr_output_path: Path,
) -> dict[str, Any]:
    """Evaluate one existing synthetic OCR output."""
    document_id = document_dir.name

    reference_text, reference_fields = _load_ground_truth(
        document_dir
    )

    ocr_output = load_json(ocr_output_path)

    hypothesis_text = ocr_output.get("full_text")

    if not isinstance(hypothesis_text, str):
        raise ValueError(
            f"OCR output has no valid 'full_text': "
            f"{ocr_output_path}"
        )

    metrics = text_metrics(
        reference_text,
        hypothesis_text,
    )

    extracted_fields = extract_critical_fields(
        hypothesis_text
    )

    field_evaluation = evaluate_fields(
        reference_fields,
        extracted_fields,
    )

    faithfulness = compare_faithfulness(
        reference_text,
        hypothesis_text,
    )

    timing = ocr_output.get("timing", {})

    if not isinstance(timing, dict):
        timing = {}

    config = ocr_output.get("config", {})

    if not isinstance(config, dict):
        config = {}

    runtime = config.get("runtime", {})

    if not isinstance(runtime, dict):
        runtime = {}

    pages = ocr_output.get("pages", [])

    if not isinstance(pages, list):
        pages = []

    return {
        "document_id": document_id,
        "ocr_output": str(
            ocr_output_path.relative_to(ROOT)
        ),
        "reference": {
            "source": str(
                (
                    document_dir
                    / "ground_truth"
                    / "full_text.txt"
                ).relative_to(ROOT)
            ),
            "authoritative_ground_truth": True,
        },
        "ocr": {
            "engine": ocr_output.get("engine"),
            "engine_version": ocr_output.get(
                "engine_version"
            ),
            "device_requested": runtime.get(
                "device_requested"
            ),
            "device_resolved": runtime.get(
                "device_resolved"
            ),
            "gpu_name": runtime.get("gpu_name"),
            "page_count": len(pages),
            "initialization_seconds": timing.get(
                "initialization_seconds"
            ),
            "document_inference_seconds": timing.get(
                "document_inference_seconds"
            ),
        },
        "text_metrics": metrics,
        "critical_fields": {
            "accuracy": field_evaluation.get(
                "accuracy"
            ),
            "total_fields": field_evaluation.get(
                "total_fields"
            ),
            "correct_fields": field_evaluation.get(
                "correct_fields"
            ),
            "exact_matches": field_evaluation.get(
                "exact_matches"
            ),
            "normalized_matches": field_evaluation.get(
                "normalized_matches"
            ),
            "mismatches": field_evaluation.get(
                "mismatches"
            ),
            "missing": field_evaluation.get(
                "missing"
            ),
            "fields": field_evaluation.get(
                "fields",
                [],
            ),
            "failures": field_evaluation.get(
                "failures",
                [],
            ),
            "extracted": extracted_fields,
        },
        "faithfulness": faithfulness.to_dict(),
    }


def build_synthetic_summary(
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build overall synthetic evaluation statistics."""
    cer_values = [
        item["text_metrics"]["cer"]
        for item in documents
        if item["text_metrics"].get("cer") is not None
    ]

    wer_values = [
        item["text_metrics"]["wer"]
        for item in documents
        if item["text_metrics"].get("wer") is not None
    ]

    field_values = [
        item["critical_fields"]["accuracy"]
        for item in documents
        if item["critical_fields"].get("accuracy")
        is not None
    ]

    faithfulness_cer_values = [
        item["faithfulness"]["cer"]
        for item in documents
        if item["faithfulness"].get("cer") is not None
    ]

    return {
        "document_count": len(documents),
        "mean_cer": _mean(cer_values),
        "mean_wer": _mean(wer_values),
        "mean_field_accuracy": _mean(field_values),
        "mean_faithfulness_cer": _mean(
            faithfulness_cer_values
        ),
        "all_documents_exact_text_match": all(
            item["faithfulness"]["exact_match"]
            for item in documents
        ),
        "all_documents_normalized_match": all(
            item["faithfulness"]["normalized_match"]
            for item in documents
        ),
    }


# ---------------------------------------------------------------------------
# Existing robustness artifact
# ---------------------------------------------------------------------------


def validate_robustness_report(
    path: Path,
) -> dict[str, Any]:
    """Load and validate the materialized robustness experiment."""
    report = load_json(path)

    experiment = report.get("experiment")

    if not isinstance(experiment, dict):
        raise ValueError(
            "Robustness report is missing the "
            "'experiment' section."
        )

    results = report.get("results")

    if not isinstance(results, list):
        raise ValueError(
            "Robustness report is missing the "
            "'results' list."
        )

    case_count = experiment.get(
        "case_count",
        len(results),
    )

    degradation_types = experiment.get(
        "degradation_types",
        [],
    )

    severities = experiment.get(
        "severities",
        [],
    )

    documents = experiment.get(
        "documents",
        [],
    )

    expected_count = (
        len(documents)
        * len(degradation_types)
        * len(severities)
    )

    if expected_count != len(results):
        raise ValueError(
            "Robustness report matrix is inconsistent: "
            f"expected {expected_count} cases, "
            f"found {len(results)}."
        )

    return {
        "artifact": str(
            path.relative_to(ROOT)
        ),
        "experiment": experiment,
        "case_count": len(results),
        "expected_case_count": expected_count,
        "validated": (
            case_count == len(results)
            and expected_count == len(results)
        ),
        "summary": report.get("summary"),
    }


# ---------------------------------------------------------------------------
# Existing real-data artifact
# ---------------------------------------------------------------------------


def load_real_data_evidence(
    path: Path,
) -> dict[str, Any] | None:
    """Load the existing real-data quality report if available."""
    if not path.is_file():
        return None

    report = load_json(path)

    return {
        "artifact": str(
            path.relative_to(ROOT)
        ),
        "ground_truth_available": report.get(
            "ground_truth_available"
        ),
        "document_count": report.get(
            "document_count"
        ),
        "interpretation": report.get(
            "interpretation"
        ),
        "documents": report.get(
            "documents",
            [],
        ),
    }


# ---------------------------------------------------------------------------
# Error-analysis integration
# ---------------------------------------------------------------------------


def run_error_analysis(
    *,
    robustness_path: Path,
    real_data_path: Path,
    json_output: Path,
    text_output: Path,
) -> dict[str, Any]:
    """Run the existing error-analysis script through its CLI."""

    script_path = ROOT / "scripts" / "analyze_errors.py"

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Error-analysis script not found: {script_path}"
        )

    command = [
        sys.executable,
        str(script_path),
        "--robustness-report",
        str(robustness_path),
        "--real-data-report",
        str(real_data_path),
        "--json-output",
        str(json_output),
        "--text-output",
        str(text_output),
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Error-analysis script failed.\n\n"
            f"Command: {' '.join(command)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    if not json_output.is_file():
        raise RuntimeError(
            "Error-analysis completed successfully but did not "
            f"produce JSON output: {json_output}"
        )

    if not text_output.is_file():
        raise RuntimeError(
            "Error-analysis completed successfully but did not "
            f"produce text output: {text_output}"
        )

    return load_json(json_output)


# ---------------------------------------------------------------------------
# Optional robustness execution
# ---------------------------------------------------------------------------


def run_robustness_experiment(
    *,
    synthetic_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Explicitly rerun the existing robustness experiment."""
    from scripts.run_robustness_experiment import (
        DEFAULT_DEGRADATION_TYPES,
        DEFAULT_SEVERITIES,
        _build_summary,
        _format_text_report,
        run_experiment,
    )

    report = run_experiment(
        synthetic_root=synthetic_root,
        degradation_types=DEFAULT_DEGRADATION_TYPES,
        severities=DEFAULT_SEVERITIES,
    )

    report["summary"] = _build_summary(report)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = output_dir / "robustness_results.json"
    text_path = output_dir / "robustness_results.txt"

    write_json(
        json_path,
        report,
    )

    text_path.write_text(
        _format_text_report(report),
        encoding="utf-8",
    )

    return report


# ---------------------------------------------------------------------------
# Final report construction
# ---------------------------------------------------------------------------


def build_final_report(
    *,
    synthetic_documents: list[dict[str, Any]],
    robustness: dict[str, Any],
    real_data: dict[str, Any] | None,
    error_analysis: dict[str, Any],
    git_revision: str | None,
) -> dict[str, Any]:
    """Build the evaluator-facing consolidated report."""
    synthetic_summary = build_synthetic_summary(
        synthetic_documents
    )

    return {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "project": {
            "name": "OCR Document Evaluation Pipeline",
            "git_revision": git_revision,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "scope": {
            "synthetic_documents": len(
                synthetic_documents
            ),
            "synthetic_ground_truth_authoritative": True,
            "assignment_documents_used_as_ground_truth": False,
            "real_data_ground_truth_available": (
                real_data.get("ground_truth_available")
                if real_data
                else False
            ),
        },
        "synthetic_baseline": {
            "summary": synthetic_summary,
            "documents": synthetic_documents,
        },
        "robustness": robustness,
        "real_data": real_data,
        "error_analysis": {
            "artifact": str(
                DEFAULT_ERROR_ANALYSIS_JSON.relative_to(
                    ROOT
                )
            ),
            "report": error_analysis,
        },
        "evaluation_interpretation": {
            "text_metrics": (
                "CER and WER quantify full-text "
                "transcription error against "
                "authoritative synthetic references."
            ),
            "critical_fields": (
                "Field-level evaluation measures "
                "whether important structured values "
                "survive OCR and extraction."
            ),
            "faithfulness": (
                "Faithfulness comparison identifies "
                "substantive differences between the "
                "reference text and evaluated output."
            ),
            "robustness": (
                "Controlled degradation measures how "
                "known OCR-like corruptions affect "
                "text metrics, critical fields, and "
                "faithfulness."
            ),
            "real_data": (
                "Assignment documents are reported using "
                "observable OCR-quality signals only when "
                "authoritative ground truth is unavailable."
            ),
        },
        "limitations": [
            (
                "Synthetic accuracy metrics describe the "
                "available development documents and do "
                "not estimate real-world OCR accuracy."
            ),
            (
                "Controlled degradation is an experimental "
                "proxy for robustness, not a substitute for "
                "a statistically representative real-document "
                "error corpus."
            ),
            (
                "Assignment documents are not treated as "
                "ground truth without authoritative "
                "transcriptions or field annotations."
            ),
            (
                "The final report reuses materialized "
                "robustness and error-analysis artifacts "
                "unless explicit recomputation is requested."
            ),
        ],
    }


# ---------------------------------------------------------------------------
# Human-readable final report
# ---------------------------------------------------------------------------


def render_text_report(
    report: dict[str, Any],
) -> str:
    """Render the consolidated report for evaluator review."""
    synthetic = report["synthetic_baseline"]
    summary = synthetic["summary"]
    robustness = report["robustness"]
    real_data = report.get("real_data")
    error_analysis = report["error_analysis"]["report"]

    lines = [
        "OCR DOCUMENT EVALUATION — FINAL REPORT",
        "=" * 80,
        "",
        f"Report version: {report['report_version']}",
        f"Generated: {report['generated_at_utc']}",
        f"Git revision: "
        f"{report['project'].get('git_revision') or 'n/a'}",
        "",
        "EVALUATION SCOPE",
        "-" * 80,
        (
            "Synthetic ground truth authoritative: "
            f"{report['scope']['synthetic_ground_truth_authoritative']}"
        ),
        (
            "Assignment documents used as ground truth: "
            f"{report['scope']['assignment_documents_used_as_ground_truth']}"
        ),
        (
            "Real-data ground truth available: "
            f"{report['scope']['real_data_ground_truth_available']}"
        ),
        "",
        "SYNTHETIC BASELINE",
        "-" * 80,
        f"Documents: {summary['document_count']}",
        f"Mean CER: {_format(summary['mean_cer'])}",
        f"Mean WER: {_format(summary['mean_wer'])}",
        (
            "Mean critical-field accuracy: "
            f"{_format(summary['mean_field_accuracy'])}"
        ),
        (
            "Mean faithfulness CER: "
            f"{_format(summary['mean_faithfulness_cer'])}"
        ),
        (
            "All documents exact text match: "
            f"{summary['all_documents_exact_text_match']}"
        ),
        (
            "All documents normalized match: "
            f"{summary['all_documents_normalized_match']}"
        ),
        "",
        "PER-DOCUMENT RESULTS",
        "-" * 80,
        (
            f"{'Document':<12}"
            f"{'CER':>10}"
            f"{'WER':>10}"
            f"{'Field Acc.':>14}"
            f"{'Faith CER':>14}"
        ),
    ]

    for document in synthetic["documents"]:
        lines.append(
            f"{document['document_id']:<12}"
            f"{_format(document['text_metrics'].get('cer')):>10}"
            f"{_format(document['text_metrics'].get('wer')):>10}"
            f"{_format(document['critical_fields'].get('accuracy')):>14}"
            f"{_format(document['faithfulness'].get('cer')):>14}"
        )

    lines.extend(
        [
            "",
            "CRITICAL-FIELD CONSEQUENCES",
            "-" * 80,
        ]
    )

    for document in synthetic["documents"]:
        failures = document["critical_fields"].get(
            "failures",
            [],
        )

        lines.append(
            f"{document['document_id']}: "
            f"{len(failures)} field failure(s)"
        )

        for failure in failures:
            lines.append(
                "  - "
                f"{failure.get('field')}: "
                f"{failure.get('status')} | "
                f"reference={failure.get('reference')!r} | "
                f"hypothesis={failure.get('hypothesis')!r}"
            )

    lines.extend(
        [
            "",
            "ROBUSTNESS",
            "-" * 80,
            (
                "Cases evaluated: "
                f"{robustness['case_count']}"
            ),
            (
                "Expected cases: "
                f"{robustness['expected_case_count']}"
            ),
            (
                "Matrix validated: "
                f"{robustness['validated']}"
            ),
            (
                "Degradation types: "
                f"{len(robustness['experiment'].get('degradation_types', []))}"
            ),
            (
                "Severities: "
                f"{robustness['experiment'].get('severities', [])}"
            ),
        ]
    )

    robustness_summary = robustness.get(
        "summary"
    )

    if isinstance(robustness_summary, dict):
        lines.extend(
            [
                "",
                "ROBUSTNESS CONTROL CASES",
                "-" * 80,
            ]
        )

        control = robustness_summary.get(
            "control_cases",
            {},
        )

        lines.extend(
            [
                (
                    "Control cases: "
                    f"{control.get('count', 'n/a')}"
                ),
                (
                    "CER zero: "
                    f"{control.get('all_cer_zero', 'n/a')}"
                ),
                (
                    "WER zero: "
                    f"{control.get('all_wer_zero', 'n/a')}"
                ),
                (
                    "Field accuracy 1.0: "
                    f"{control.get('all_fields_perfect', 'n/a')}"
                ),
                (
                    "Faithfulness CER zero: "
                    f"{control.get('all_faithfulness_cer_zero', 'n/a')}"
                ),
            ]
        )

        by_degradation = robustness_summary.get(
            "by_degradation",
            {},
        )

        if isinstance(by_degradation, dict):
            lines.extend(
                [
                    "",
                    "ROBUSTNESS IMPACT BY DEGRADATION",
                    "-" * 80,
                    (
                        f"{'Degradation':<28}"
                        f"{'CER':>10}"
                        f"{'WER':>10}"
                        f"{'Field Acc.':>14}"
                        f"{'Changed':>12}"
                    ),
                ]
            )

            for name, values in by_degradation.items():
                lines.append(
                    f"{name:<28}"
                    f"{_format(values.get('mean_cer')):>10}"
                    f"{_format(values.get('mean_wer')):>10}"
                    f"{_format(values.get('mean_field_accuracy')):>14}"
                    f"{_format(values.get('mean_changed_positions')):>12}"
                )

    lines.extend(
        [
            "",
            "REAL-DATA OBSERVATIONS",
            "-" * 80,
        ]
    )

    if real_data is None:
        lines.append(
            "No real-data OCR quality report was available."
        )
    else:
        lines.append(
            "Authoritative ground truth available: "
            f"{real_data.get('ground_truth_available')}"
        )
        lines.append(
            "Documents evaluated: "
            f"{real_data.get('document_count')}"
        )
        lines.append(
            "Interpretation: "
            f"{real_data.get('interpretation')}"
        )

        for document in real_data.get(
            "documents",
            [],
        ):
            lines.extend(
                [
                    "",
                    f"Document: {document.get('source')}",
                    (
                        f"  Pages: "
                        f"{document.get('page_count')}"
                    ),
                    (
                        "  Characters: "
                        f"{document.get('text', {}).get('characters')}"
                    ),
                    (
                        "  Words: "
                        f"{document.get('text', {}).get('words')}"
                    ),
                    (
                        "  Mean confidence: "
                        f"{_format(document.get('confidence', {}).get('mean'))}"
                    ),
                    (
                        "  Near-empty pages: "
                        f"{document.get('anomalies', {}).get('near_empty_pages')}"
                    ),
                    (
                        "  Low-confidence pages: "
                        f"{document.get('anomalies', {}).get('low_confidence_pages')}"
                    ),
                ]
            )

    lines.extend(
        [
            "",
            "ERROR ANALYSIS",
            "-" * 80,
            (
                "Robustness cases analyzed: "
                f"{error_analysis.get('robustness', {}).get('case_count', 'n/a')}"
            ),
        ]
    )

    categories = error_analysis.get(
        "error_categories",
        {},
    )

    if isinstance(categories, dict):
        for name, category in categories.items():
            if isinstance(category, dict):
                count = category.get(
                    "case_count",
                    category.get(
                        "robustness_cases",
                        0,
                    ),
                )
            else:
                count = 0

            lines.append(
                f"  {name}: {count}"
            )

    findings = error_analysis.get(
        "findings",
        error_analysis.get(
            "evaluator_findings",
            [],
        ),
    )

    if isinstance(findings, list) and findings:
        lines.extend(
            [
                "",
                "EVALUATOR FINDINGS",
                "-" * 80,
            ]
        )

        for finding in findings:
            if isinstance(finding, dict):
                finding_name = finding.get("finding")
                evidence = finding.get("evidence")

                if finding_name and evidence:
                    lines.append(
                        f"- {finding_name}: {evidence}"
                    )
                elif finding_name:
                    lines.append(f"- {finding_name}")
                elif evidence:
                    lines.append(f"- {evidence}")
            else:
                lines.append(f"- {finding}")

    lines.extend(
        [
            "",
            "PRODUCTION-RELEVANT INTERPRETATION",
            "-" * 80,
            (
                "1. Aggregate CER/WER should not be used as the "
                "sole acceptance criterion because structured "
                "field failures can have disproportionate "
                "consequences."
            ),
            (
                "2. Identifier-like fields require explicit "
                "evaluation because punctuation and small "
                "character changes can alter meaning."
            ),
            (
                "3. Confidence signals are useful for routing "
                "human review, but confidence alone does not "
                "establish transcription correctness."
            ),
            (
                "4. Controlled degradation provides a repeatable "
                "regression mechanism for testing sensitivity "
                "to known corruption patterns."
            ),
            (
                "5. A production system should maintain a versioned "
                "golden set, monitor field-level regressions, and "
                "route low-confidence or high-risk cases for review."
            ),
            "",
            "LIMITATIONS",
            "-" * 80,
        ]
    )

    for limitation in report["limitations"]:
        lines.append(
            f"- {limitation}"
        )

    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_evaluation(
    *,
    synthetic_root: Path = DEFAULT_SYNTHETIC_ROOT,
    raw_output_dir: Path = DEFAULT_RAW_OUTPUT_DIR,
    robustness_report: Path = DEFAULT_ROBUSTNESS_REPORT,
    real_data_report: Path = DEFAULT_REAL_DATA_REPORT,
    output_json: Path = DEFAULT_JSON_OUTPUT,
    output_text: Path = DEFAULT_TEXT_OUTPUT,
    run_robustness: bool = False,
    require_real_data: bool = False,
) -> dict[str, Any]:
    """Run the complete evaluation orchestration."""
    documents = discover_synthetic_documents(
        synthetic_root
    )

    synthetic_results: list[dict[str, Any]] = []

    missing_outputs: list[str] = []

    for document_dir in documents:
        document_id = document_dir.name

        output_path = find_ocr_output(
            document_id,
            raw_output_dir,
        )

        if output_path is None:
            missing_outputs.append(document_id)
            continue

        synthetic_results.append(
            evaluate_synthetic_document(
                document_dir=document_dir,
                ocr_output_path=output_path,
            )
        )

    if missing_outputs:
        expected = "\n".join(
            str(path)
            for document_id in missing_outputs
            for path in _candidate_raw_outputs(
                document_id
            )
        )

        raise FileNotFoundError(
            "Missing OCR output(s) for synthetic documents: "
            f"{', '.join(missing_outputs)}.\n\n"
            "Generate the OCR outputs first. Preferred candidates "
            "include:\n"
            f"{expected}"
        )

    if run_robustness:
        generated_robustness = run_robustness_experiment(
            synthetic_root=synthetic_root,
            output_dir=robustness_report.parent,
        )

        robustness = validate_robustness_report(
            robustness_report
        )

        # Preserve the freshly generated report summary.
        robustness["summary"] = generated_robustness.get(
            "summary"
        )
    else:
        robustness = validate_robustness_report(
            robustness_report
        )

    real_data = load_real_data_evidence(
        real_data_report
    )

    if require_real_data and real_data is None:
        raise FileNotFoundError(
            "Real-data evaluation report is required but was "
            f"not found: {real_data_report}"
        )

    error_analysis = run_error_analysis(
        robustness_path=robustness_report,
        real_data_path=real_data_report,
        json_output=DEFAULT_ERROR_ANALYSIS_JSON,
        text_output=DEFAULT_ERROR_ANALYSIS_TEXT,
    )

    report = build_final_report(
        synthetic_documents=synthetic_results,
        robustness=robustness,
        real_data=real_data,
        error_analysis=error_analysis,
        git_revision=_git_revision(),
    )

    write_json(
        output_json,
        report,
    )

    output_text.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_text.write_text(
        render_text_report(report),
        encoding="utf-8",
    )

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete OCR evaluation orchestration "
            "and produce a consolidated evaluator-facing report."
        )
    )

    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=DEFAULT_SYNTHETIC_ROOT,
        help=(
            "Root directory containing synthetic documents "
            "and authoritative ground truth."
        ),
    )

    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        default=DEFAULT_RAW_OUTPUT_DIR,
        help=(
            "Directory containing previously generated "
            "synthetic OCR JSON outputs."
        ),
    )

    parser.add_argument(
        "--robustness-report",
        type=Path,
        default=DEFAULT_ROBUSTNESS_REPORT,
        help=(
            "Existing robustness experiment JSON artifact."
        ),
    )

    parser.add_argument(
        "--real-data-report",
        type=Path,
        default=DEFAULT_REAL_DATA_REPORT,
        help=(
            "Existing real-data OCR quality report. "
            "Optional unless --require-real-data is supplied."
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help="Final machine-readable evaluation report.",
    )

    parser.add_argument(
        "--output-text",
        type=Path,
        default=DEFAULT_TEXT_OUTPUT,
        help="Final human-readable evaluation report.",
    )

    parser.add_argument(
        "--run-robustness",
        action="store_true",
        help=(
            "Explicitly rerun the expensive 90-case synthetic "
            "robustness experiment before building the final report."
        ),
    )

    parser.add_argument(
        "--require-real-data",
        action="store_true",
        help=(
            "Fail if the real-data OCR quality report is unavailable."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()

    report = run_evaluation(
        synthetic_root=args.synthetic_root,
        raw_output_dir=args.raw_output_dir,
        robustness_report=args.robustness_report,
        real_data_report=args.real_data_report,
        output_json=args.output_json,
        output_text=args.output_text,
        run_robustness=args.run_robustness,
        require_real_data=args.require_real_data,
    )

    summary = report["synthetic_baseline"]["summary"]

    print("=" * 80)
    print("OCR DOCUMENT EVALUATION")
    print("=" * 80)
    print(
        "Synthetic documents: "
        f"{summary['document_count']}"
    )
    print(
        "Mean CER: "
        f"{_format(summary['mean_cer'])}"
    )
    print(
        "Mean WER: "
        f"{_format(summary['mean_wer'])}"
    )
    print(
        "Mean field accuracy: "
        f"{_format(summary['mean_field_accuracy'])}"
    )
    print(
        "Mean faithfulness CER: "
        f"{_format(summary['mean_faithfulness_cer'])}"
    )
    print(
        "Robustness cases: "
        f"{report['robustness']['case_count']}"
    )
    print(
        "Real-data evidence: "
        f"{'available' if report['real_data'] else 'not available'}"
    )
    print()
    print(
        f"JSON report: {args.output_json}"
    )
    print(
        f"Text report: {args.output_text}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())