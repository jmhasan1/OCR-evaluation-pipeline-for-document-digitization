"""Generate objective OCR quality reports for real assignment documents.

This script evaluates OCR outputs without requiring ground-truth
transcriptions. It focuses on observable OCR-quality signals:

- document/page counts
- character and word counts
- empty/near-empty pages
- OCR confidence statistics
- low-confidence regions
- per-page anomalies
- document-level throughput and processing metadata

The employer-provided source PDFs and generated OCR outputs are intentionally
kept outside version control.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("outputs/assignment")
DEFAULT_OUTPUT_JSON = DEFAULT_INPUT_DIR / "real_data_ocr_quality_report.json"
DEFAULT_OUTPUT_TEXT = DEFAULT_INPUT_DIR / "real_data_ocr_quality_report.txt"

EXCLUDED_FILES = {
    "ocr_quality_report.json",
    "real_data_ocr_quality_report.json",
}

# These thresholds are diagnostic heuristics, not accuracy claims.
NEAR_EMPTY_CHARACTER_THRESHOLD = 20
LOW_CONFIDENCE_THRESHOLD = 0.80
LOW_CONFIDENCE_REGION_THRESHOLD = 0.70


def _safe_mean(values: list[float]) -> float | None:
    """Return the arithmetic mean or None for an empty sequence."""
    if not values:
        return None
    return statistics.mean(values)


def _safe_min(values: list[float]) -> float | None:
    """Return the minimum or None for an empty sequence."""
    if not values:
        return None
    return min(values)


def _safe_max(values: list[float]) -> float | None:
    """Return the maximum or None for an empty sequence."""
    if not values:
        return None
    return max(values)


def _percentile(values: list[float], percentile: float) -> float | None:
    """Calculate a percentile using linear interpolation."""
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _region_confidences(page: dict[str, Any]) -> list[float]:
    """Extract valid OCR-region confidence values from a page."""
    values: list[float] = []

    for region in page.get("regions", []):
        confidence = region.get("confidence")

        if isinstance(confidence, (int, float)):
            values.append(float(confidence))

    return values


def _page_metrics(page: dict[str, Any]) -> dict[str, Any]:
    """Calculate objective diagnostic metrics for one OCR page."""
    text = page.get("text") or ""

    words = text.split()
    characters = len(text)

    confidences = _region_confidences(page)

    low_confidence_regions = [
        confidence
        for confidence in confidences
        if confidence < LOW_CONFIDENCE_REGION_THRESHOLD
    ]

    return {
        "page_number": page.get("page_number"),
        "characters": characters,
        "words": len(words),
        "regions": len(page.get("regions", [])),
        "confidence": {
            "regions_with_confidence": len(confidences),
            "mean": _safe_mean(confidences),
            "min": _safe_min(confidences),
            "p10": _percentile(confidences, 0.10),
            "median": _percentile(confidences, 0.50),
            "p90": _percentile(confidences, 0.90),
            "max": _safe_max(confidences),
        },
        "anomalies": {
            "empty_page": characters == 0,
            "near_empty_page": 0 < characters < NEAR_EMPTY_CHARACTER_THRESHOLD,
            "low_mean_confidence": (
                bool(confidences)
                and _safe_mean(confidences) is not None
                and _safe_mean(confidences) < LOW_CONFIDENCE_THRESHOLD
            ),
            "low_confidence_regions": len(low_confidence_regions),
        },
    }


def evaluate_document(path: Path) -> dict[str, Any]:
    """Evaluate one OCR JSON document."""
    data = json.loads(path.read_text(encoding="utf-8"))

    pages = data.get("pages", [])
    page_metrics = [_page_metrics(page) for page in pages]

    all_confidences: list[float] = []

    for page in pages:
        all_confidences.extend(_region_confidences(page))

    total_characters = sum(
        metric["characters"] for metric in page_metrics
    )
    total_words = sum(metric["words"] for metric in page_metrics)

    empty_pages = [
        metric["page_number"]
        for metric in page_metrics
        if metric["anomalies"]["empty_page"]
    ]

    near_empty_pages = [
        metric["page_number"]
        for metric in page_metrics
        if metric["anomalies"]["near_empty_page"]
    ]

    low_confidence_pages = [
        metric["page_number"]
        for metric in page_metrics
        if metric["anomalies"]["low_mean_confidence"]
    ]

    low_confidence_region_count = sum(
        metric["anomalies"]["low_confidence_regions"]
        for metric in page_metrics
    )

    inference_seconds = data.get("timing", {}).get(
        "document_inference_seconds"
    )

    if inference_seconds is None:
        inference_seconds = data.get("document_inference_seconds")

    pages_per_second = None

    if (
        isinstance(inference_seconds, (int, float))
        and inference_seconds > 0
        and pages
    ):
        pages_per_second = len(pages) / inference_seconds

    return {
        "source": path.name,
        "document_id": data.get("document_id"),
        "input_path": data.get("input_path"),
        "engine": data.get("engine"),
        "engine_version": data.get("engine_version"),
        "config": data.get("config", {}),
        "page_count": len(pages),
        "text": {
            "characters": total_characters,
            "words": total_words,
            "average_characters_per_page": (
                total_characters / len(pages) if pages else None
            ),
            "average_words_per_page": (
                total_words / len(pages) if pages else None
            ),
        },
        "confidence": {
            "regions_with_confidence": len(all_confidences),
            "mean": _safe_mean(all_confidences),
            "min": _safe_min(all_confidences),
            "p10": _percentile(all_confidences, 0.10),
            "median": _percentile(all_confidences, 0.50),
            "p90": _percentile(all_confidences, 0.90),
            "max": _safe_max(all_confidences),
        },
        "timing": {
            "initialization_seconds": data.get("timing", {}).get(
                "initialization_seconds"
            ),
            "document_inference_seconds": inference_seconds,
            "pages_per_second": pages_per_second,
        },
        "anomalies": {
            "empty_pages": empty_pages,
            "near_empty_pages": near_empty_pages,
            "low_confidence_pages": low_confidence_pages,
            "low_confidence_region_count": low_confidence_region_count,
        },
        "pages": page_metrics,
    }


def build_report(input_dir: Path) -> dict[str, Any]:
    """Build a report for all OCR JSON files in an assignment directory."""

    files = sorted(
    path
    for path in input_dir.glob("*.json")
    if path.name not in EXCLUDED_FILES
    )

    documents = [evaluate_document(path) for path in files]

    return {
        "report_version": "1.0",
        "evaluation_type": "real_data_ocr_quality",
        "ground_truth_available": False,
        "interpretation": (
            "This report measures observable OCR quality signals only. "
            "It does not calculate CER/WER or field accuracy because "
            "authoritative ground-truth transcriptions/fields are unavailable "
            "for the assignment documents."
        ),
        "thresholds": {
            "near_empty_character_threshold": (
                NEAR_EMPTY_CHARACTER_THRESHOLD
            ),
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "low_confidence_region_threshold": (
                LOW_CONFIDENCE_REGION_THRESHOLD
            ),
        },
        "document_count": len(documents),
        "documents": documents,
    }


def _format_value(value: Any) -> str:
    """Format report values for human-readable CLI output."""
    if value is None:
        return "n/a"

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


def render_text_report(report: dict[str, Any]) -> str:
    """Render the machine-readable report as concise human-readable text."""
    lines = [
        "REAL-DATA OCR QUALITY REPORT",
        "=" * 80,
        "",
        "Evaluation type: real-data OCR quality",
        f"Ground truth available: {report['ground_truth_available']}",
        f"Documents evaluated: {report['document_count']}",
        "",
        "IMPORTANT",
        "-" * 80,
        report["interpretation"],
        "",
    ]

    for document in report["documents"]:
        lines.extend(
            [
                "=" * 80,
                document["source"],
                "=" * 80,
                f"Document ID: {document['document_id']}",
                f"Engine: {document['engine']}",
                f"Engine version: {document['engine_version']}",
                f"Pages: {document['page_count']}",
                "",
                "TEXT",
                "-" * 80,
                f"Characters: {document['text']['characters']}",
                f"Words: {document['text']['words']}",
                (
                    "Average characters/page: "
                    f"{_format_value(document['text']['average_characters_per_page'])}"
                ),
                (
                    "Average words/page: "
                    f"{_format_value(document['text']['average_words_per_page'])}"
                ),
                "",
                "CONFIDENCE",
                "-" * 80,
                (
                    "Regions with confidence: "
                    f"{document['confidence']['regions_with_confidence']}"
                ),
                f"Mean: {_format_value(document['confidence']['mean'])}",
                f"Minimum: {_format_value(document['confidence']['min'])}",
                f"P10: {_format_value(document['confidence']['p10'])}",
                f"Median: {_format_value(document['confidence']['median'])}",
                f"P90: {_format_value(document['confidence']['p90'])}",
                f"Maximum: {_format_value(document['confidence']['max'])}",
                "",
                "TIMING",
                "-" * 80,
                (
                    "Initialization: "
                    f"{_format_value(document['timing']['initialization_seconds'])} s"
                ),
                (
                    "Inference: "
                    f"{_format_value(document['timing']['document_inference_seconds'])} s"
                ),
                (
                    "Pages/second: "
                    f"{_format_value(document['timing']['pages_per_second'])}"
                ),
                "",
                "ANOMALIES",
                "-" * 80,
                (
                    "Empty pages: "
                    f"{document['anomalies']['empty_pages'] or 'none'}"
                ),
                (
                    "Near-empty pages: "
                    f"{document['anomalies']['near_empty_pages'] or 'none'}"
                ),
                (
                    "Low-confidence pages: "
                    f"{document['anomalies']['low_confidence_pages'] or 'none'}"
                ),
                (
                    "Low-confidence regions: "
                    f"{document['anomalies']['low_confidence_region_count']}"
                ),
                "",
            ]
        )

    return "\n".join(lines)


def main() -> int:
    """Run the real-data OCR quality evaluation."""
    parser = argparse.ArgumentParser(
        description="Generate a real-data OCR quality report."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing OCR JSON outputs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Machine-readable report path.",
    )
    parser.add_argument(
        "--output-text",
        type=Path,
        default=DEFAULT_OUTPUT_TEXT,
        help="Human-readable report path.",
    )

    args = parser.parse_args()

    if not args.input_dir.exists():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    report = build_report(args.input_dir)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    args.output_text.write_text(
        render_text_report(report),
        encoding="utf-8",
    )

    print(f"Evaluated documents: {report['document_count']}")
    print(f"JSON report: {args.output_json}")
    print(f"Text report: {args.output_text}")

    for document in report["documents"]:
        print(
            f"{document['source']}: "
            f"{document['page_count']} pages, "
            f"{document['text']['characters']} chars, "
            f"{document['text']['words']} words"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())