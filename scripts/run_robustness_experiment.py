"""Run a controlled OCR robustness experiment over synthetic documents.

The experiment applies every supported deterministic degradation at a fixed
set of severity levels to the three synthetic documents, then evaluates the
result with the project's existing text, field, and faithfulness metrics.

This script intentionally operates only on synthetic development data.
Employer-provided assignment documents are excluded from the experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow direct execution from the repository root without requiring an
# editable package installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_eval.critical_fields import evaluate_fields
from ocr_eval.degradation import DEGRADATIONS, apply_degradation
from ocr_eval.faithfulness import compare_faithfulness
from ocr_eval.metrics import text_metrics


DEFAULT_DEGRADATION_TYPES = tuple(DEGRADATIONS.keys())
DEFAULT_SEVERITIES = (0.0, 0.25, 0.50, 0.75, 1.0)


def _load_document(
    document_dir: Path,
) -> tuple[str, dict[str, Any]]:
    """Load synthetic ground-truth text and critical fields."""

    ground_truth_dir = document_dir / "ground_truth"

    text_path = ground_truth_dir / "full_text.txt"
    fields_path = ground_truth_dir / "fields.json"

    if not text_path.is_file():
        raise FileNotFoundError(
            f"Missing ground-truth text: {text_path}"
        )

    if not fields_path.is_file():
        raise FileNotFoundError(
            f"Missing ground-truth fields: {fields_path}"
        )

    text = text_path.read_text(encoding="utf-8")
    metadata = json.loads(
        fields_path.read_text(encoding="utf-8")
    )

    if "fields" not in metadata:
        raise ValueError(
            f"Missing 'fields' in {fields_path}"
        )

    return text, metadata


def _extract_fields(text: str) -> dict[str, Any]:
    """Extract critical fields using the project's production extractor."""

    from ocr_eval.extraction import extract_critical_fields

    return extract_critical_fields(text)


def _serialize_result(result: Any) -> dict[str, Any]:
    """Convert a project result object to a JSON-compatible dictionary."""

    if hasattr(result, "to_dict"):
        return result.to_dict()

    if isinstance(result, dict):
        return result

    if hasattr(result, "__dict__"):
        return dict(result.__dict__)

    raise TypeError(
        f"Unsupported result type: {type(result).__name__}"
    )


def _run_case(
    *,
    document_id: str,
    reference_text: str,
    reference_fields: dict[str, Any],
    degradation_type: str,
    severity: float,
) -> dict[str, Any]:
    """Run one degradation/evaluation case."""

    degradation_result = apply_degradation(
        reference_text,
        degradation=degradation_type,
        severity=severity,
    )

    degraded_text = degradation_result.degraded_text

    metrics = text_metrics(
        reference_text,
        degraded_text,
    )

    extracted_fields = _extract_fields(
        degraded_text
    )

    field_result = evaluate_fields(
        reference_fields,
        extracted_fields,
    )

    faithfulness_result = compare_faithfulness(
    reference_text,
    degraded_text,
    )

    return {
        "document_id": document_id,
        "degradation": {
            "type": degradation_type,
            "severity": severity,
            "changed_positions": list(
                degradation_result.changed_positions
            ),
        },
        "text_metrics": _serialize_result(metrics),
        "field_evaluation": _serialize_result(field_result),
        "faithfulness": _serialize_result(
            faithfulness_result
        ),
        "text": {
            "reference": reference_text,
            "degraded": degraded_text,
        },
    }


def run_experiment(
    *,
    synthetic_root: Path,
    degradation_types: tuple[str, ...] = DEFAULT_DEGRADATION_TYPES,
    severities: tuple[float, ...] = DEFAULT_SEVERITIES,
) -> dict[str, Any]:
    """Run the complete synthetic robustness experiment."""

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
            f"No synthetic documents found under "
            f"{synthetic_root}"
        )

    results: list[dict[str, Any]] = []

    for document_dir in documents:
        reference_text, metadata = _load_document(
            document_dir
        )

        document_id = str(
            metadata.get(
                "document_id",
                document_dir.name,
            )
        )

        for degradation_type in degradation_types:
            if degradation_type not in DEGRADATIONS:
                raise ValueError(
                    f"Unknown degradation type: "
                    f"{degradation_type}"
                )

            for severity in severities:
                result = _run_case(
                    document_id=document_id,
                    reference_text=reference_text,
                    reference_fields=metadata["fields"],
                    degradation_type=degradation_type,
                    severity=severity,
                )

                results.append(result)

    return {
        "experiment": {
            "name": "synthetic_ocr_robustness",
            "documents": [
                path.name for path in documents
            ],
            "degradation_types": list(
                degradation_types
            ),
            "severities": list(severities),
            "case_count": len(results),
        },
        "results": results,
    }


def _mean(
    values: list[float],
) -> float | None:
    """Return the arithmetic mean, or None for no values."""

    return (
        sum(values) / len(values)
        if values
        else None
    )


def _build_summary(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Build compact aggregate statistics."""

    results = report["results"]

    by_degradation: dict[str, dict[str, Any]] = {}

    for degradation_type in report[
        "experiment"
    ]["degradation_types"]:

        subset = [
            result
            for result in results
            if result["degradation"]["type"]
            == degradation_type
        ]

        by_degradation[degradation_type] = {
            "cases": len(subset),

            "mean_cer": _mean(
                [
                    result["text_metrics"]["cer"]
                    for result in subset
                    if result["text_metrics"].get(
                        "cer"
                    ) is not None
                ]
            ),

            "mean_wer": _mean(
                [
                    result["text_metrics"]["wer"]
                    for result in subset
                    if result["text_metrics"].get(
                        "wer"
                    ) is not None
                ]
            ),

            "mean_field_accuracy": _mean(
                [
                    result[
                        "field_evaluation"
                    ]["accuracy"]
                    for result in subset
                ]
            ),

            "mean_faithfulness_cer": _mean(
                [
                    result[
                        "faithfulness"
                    ].get("cer")
                    for result in subset
                    if result[
                        "faithfulness"
                    ].get("cer") is not None
                ]
            ),

            "mean_changed_positions": _mean(
                [
                    float(
                        len(
                            result[
                                "degradation"
                            ]["changed_positions"]
                        )
                    )
                    for result in subset
                ]
            ),
        }

    control_cases = [
        result
        for result in results
        if result["degradation"]["severity"]
        == 0.0
    ]

    expected_case_count = (
        len(
            report["experiment"][
                "documents"
            ]
        )
        * len(
            report["experiment"][
                "degradation_types"
            ]
        )
        * len(
            report["experiment"][
                "severities"
            ]
        )
    )

    return {
        "case_count": len(results),
        "expected_case_count": expected_case_count,

        "control_cases": {
            "count": len(control_cases),

            "all_cer_zero": all(
                result["text_metrics"]["cer"]
                == 0.0
                for result in control_cases
            ),

            "all_wer_zero": all(
                result["text_metrics"]["wer"]
                == 0.0
                for result in control_cases
            ),

            "all_fields_perfect": all(
                result[
                    "field_evaluation"
                ]["accuracy"]
                == 1.0
                for result in control_cases
            ),

            "all_faithfulness_cer_zero": all(
                result[
                    "faithfulness"
                ].get("cer") == 0.0
                for result in control_cases
            ),
        },

        "by_degradation": by_degradation,
    }


def _format_number(
    value: float | None,
) -> str:
    """Format optional numeric values for CLI output."""

    if value is None:
        return "n/a"

    return f"{value:.4f}"


def _format_text_report(
    report: dict[str, Any],
) -> str:
    """Create a concise human-readable robustness report."""

    experiment = report["experiment"]
    summary = report["summary"]

    lines = [
        "SYNTHETIC OCR ROBUSTNESS REPORT",
        "=" * 80,
        "",
        f"Documents: "
        f"{len(experiment['documents'])}",
        f"Degradation types: "
        f"{len(experiment['degradation_types'])}",
        f"Severities: "
        f"{', '.join(str(x) for x in experiment['severities'])}",
        f"Cases: {summary['case_count']}",
        f"Expected cases: "
        f"{summary['expected_case_count']}",
        "",
        "CONTROL CASES",
        "-" * 80,
        f"Cases: "
        f"{summary['control_cases']['count']}",
        f"CER zero: "
        f"{summary['control_cases']['all_cer_zero']}",
        f"WER zero: "
        f"{summary['control_cases']['all_wer_zero']}",
        f"Field accuracy 1.0: "
        f"{summary['control_cases']['all_fields_perfect']}",
        f"Faithfulness CER zero: "
        f"{summary['control_cases']['all_faithfulness_cer_zero']}",
        "",
        "AGGREGATE IMPACT BY DEGRADATION",
        "-" * 80,
        (
            f"{'Degradation':<28}"
            f"{'Mean CER':>12}"
            f"{'Mean WER':>12}"
            f"{'Field Acc.':>14}"
            f"{'Changed Pos.':>15}"
        ),
    ]

    for degradation_type, values in (
        summary["by_degradation"].items()
    ):
        lines.append(
            f"{degradation_type:<28}"
            f"{_format_number(values['mean_cer']):>12}"
            f"{_format_number(values['mean_wer']):>12}"
            f"{_format_number(values['mean_field_accuracy']):>14}"
            f"{_format_number(values['mean_changed_positions']):>15}"
        )

    lines.extend(
        [
            "",
            "INTERPRETATION",
            "-" * 80,
            (
                "The experiment measures how controlled "
                "OCR-like textual corruption affects "
                "CER/WER, critical-field accuracy, "
                "faithfulness, and the amount of "
                "changed text."
            ),
            (
                "Severity 0.0 is the deterministic control "
                "condition and should produce zero text "
                "error and perfect field extraction."
            ),
            (
                "The experiment is intentionally limited "
                "to synthetic development documents so "
                "the employer-provided assignment data "
                "is never treated as synthetic ground truth."
            ),
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the synthetic OCR robustness "
            "experiment."
        )
    )

    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=(
            ROOT
            / "data"
            / "development"
            / "synthetic_documents"
        ),
        help="Root directory containing synthetic documents.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "robustness"
        ),
        help="Directory for generated reports.",
    )

    args = parser.parse_args()

    report = run_experiment(
        synthetic_root=args.synthetic_root,
    )

    report["summary"] = _build_summary(
        report
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        args.output_dir
        / "robustness_results.json"
    )

    text_path = (
        args.output_dir
        / "robustness_results.txt"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    text_path.write_text(
        _format_text_report(report),
        encoding="utf-8",
    )

    print(
        f"Cases evaluated: "
        f"{report['summary']['case_count']}"
    )

    print(
        f"JSON report: {json_path}"
    )

    print(
        f"Text report: {text_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())