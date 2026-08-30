"""Analyze OCR quality characteristics for employer-provided documents.

This script intentionally performs descriptive analysis only. It does not
claim field-level accuracy because the employer-provided documents do not
currently have corresponding ground-truth field annotations.
"""

from __future__ import annotations

import json
import statistics
import unicodedata
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("outputs/assignment")
DEFAULT_OUTPUT_JSON = DEFAULT_INPUT_DIR / "ocr_quality_report.json"
DEFAULT_OUTPUT_TEXT = DEFAULT_INPUT_DIR / "ocr_quality_report.txt"

NEAR_EMPTY_THRESHOLD = 20
NUMERIC_HEAVY_THRESHOLD = 0.30
CORRUPTION_SYMBOL_THRESHOLD = 0.05


def is_latin(character: str) -> bool:
    """Return whether a character belongs to the Latin script."""
    name = unicodedata.name(character, "")
    return "LATIN" in name


def is_devanagari(character: str) -> bool:
    """Return whether a character belongs to the Devanagari script."""
    codepoint = ord(character)
    return 0x0900 <= codepoint <= 0x097F


def is_digit(character: str) -> bool:
    """Return whether a character is a Unicode decimal digit."""
    return character.isdigit()


def script_counts(text: str) -> dict[str, int]:
    """Count broad script categories in OCR text."""
    latin = 0
    devanagari = 0
    digits = 0
    other = 0

    for character in text:
        if character.isspace():
            continue

        if is_digit(character):
            digits += 1
        elif is_latin(character):
            latin += 1
        elif is_devanagari(character):
            devanagari += 1
        else:
            other += 1

    return {
        "latin": latin,
        "devanagari": devanagari,
        "digits": digits,
        "other": other,
    }


def classify_page(text: str) -> dict[str, Any]:
    """Produce descriptive quality indicators for one OCR page."""
    stripped = text.strip()
    counts = script_counts(stripped)
    non_whitespace = sum(not character.isspace() for character in stripped)

    numeric_ratio = (
        counts["digits"] / non_whitespace
        if non_whitespace
        else 0.0
    )

    latin_present = counts["latin"] > 0
    devanagari_present = counts["devanagari"] > 0

    # These characters are useful as cautious indicators of possible OCR
    # corruption because they are common in malformed mixed-script output.
    suspicious_characters = set("�")
    suspicious_count = sum(
        character in suspicious_characters for character in stripped
    )

    suspicious_ratio = (
        suspicious_count / non_whitespace
        if non_whitespace
        else 0.0
    )

    flags: list[str] = []

    if len(stripped) <= NEAR_EMPTY_THRESHOLD:
        flags.append("low_text_volume")

    if numeric_ratio >= NUMERIC_HEAVY_THRESHOLD:
        flags.append("numeric_heavy")

    if latin_present and devanagari_present:
        flags.append("mixed_script")

    if suspicious_ratio >= CORRUPTION_SYMBOL_THRESHOLD:
        flags.append("possible_ocr_corruption")

    return {
        "characters": len(stripped),
        "latin_characters": counts["latin"],
        "devanagari_characters": counts["devanagari"],
        "digit_characters": counts["digits"],
        "other_characters": counts["other"],
        "numeric_ratio": round(numeric_ratio, 4),
        "suspicious_character_count": suspicious_count,
        "suspicious_character_ratio": round(suspicious_ratio, 4),
        "flags": flags,
    }


def analyze_document(path: Path) -> dict[str, Any]:
    """Analyze one normalized OCR JSON output."""
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("pages", [])

    page_reports: list[dict[str, Any]] = []

    for page in pages:
        text = page.get("text", "")
        report = classify_page(text)
        report["page_number"] = page.get("page_number")
        page_reports.append(report)

    character_counts = [
        report["characters"]
        for report in page_reports
    ]

    if character_counts:
        mean_characters = statistics.mean(character_counts)
        median_characters = statistics.median(character_counts)
        minimum_characters = min(character_counts)
        maximum_characters = max(character_counts)
    else:
        mean_characters = 0.0
        median_characters = 0.0
        minimum_characters = 0
        maximum_characters = 0

    page_count = len(page_reports)

    low_text_pages = [
        report["page_number"]
        for report in page_reports
        if "low_text_volume" in report["flags"]
    ]

    latin_pages = [
        report["page_number"]
        for report in page_reports
        if report["latin_characters"] > 0
    ]

    devanagari_pages = [
        report["page_number"]
        for report in page_reports
        if report["devanagari_characters"] > 0
    ]

    numeric_pages = [
        report["page_number"]
        for report in page_reports
        if report["digit_characters"] > 0
    ]

    numeric_heavy_pages = [
        report["page_number"]
        for report in page_reports
        if "numeric_heavy" in report["flags"]
    ]

    corruption_pages = [
        report["page_number"]
        for report in page_reports
        if "possible_ocr_corruption" in report["flags"]
    ]

    mixed_script_pages = [
        report["page_number"]
        for report in page_reports
        if "mixed_script" in report["flags"]
    ]

    total_characters = sum(character_counts)

    return {
        "source": str(path),
        "document_id": data.get("document_id"),
        "engine": data.get("engine"),
        "engine_version": data.get("engine_version"),
        "pages": page_count,
        "total_characters": total_characters,
        "mean_characters_per_page": round(mean_characters, 2),
        "median_characters_per_page": round(median_characters, 2),
        "min_characters_per_page": minimum_characters,
        "max_characters_per_page": maximum_characters,
        "near_empty_page_ratio": round(
            len(low_text_pages) / page_count,
            4,
        )
        if page_count
        else 0.0,
        "latin_page_ratio": round(
            len(latin_pages) / page_count,
            4,
        )
        if page_count
        else 0.0,
        "devanagari_page_ratio": round(
            len(devanagari_pages) / page_count,
            4,
        )
        if page_count
        else 0.0,
        "numeric_page_ratio": round(
            len(numeric_pages) / page_count,
            4,
        )
        if page_count
        else 0.0,
        "low_text_pages": low_text_pages,
        "latin_pages": latin_pages,
        "devanagari_pages": devanagari_pages,
        "numeric_pages": numeric_pages,
        "numeric_heavy_pages": numeric_heavy_pages,
        "possible_corruption_pages": corruption_pages,
        "mixed_script_pages": mixed_script_pages,
        "page_reports": page_reports,
    }


def build_report(input_dir: Path) -> dict[str, Any]:
    """Analyze every OCR JSON in an assignment output directory."""
    files = sorted(
        path
        for path in input_dir.glob("*.json")
        if path.name != DEFAULT_OUTPUT_JSON.name
    )

    documents = [
        analyze_document(path)
        for path in files
    ]

    return {
        "report_type": "real_data_ocr_quality",
        "ground_truth_available": False,
        "documents": documents,
    }


def format_document_summary(document: dict[str, Any]) -> str:
    """Format one document summary for CLI output."""
    return "\n".join(
        [
            f"Document: {document['source']}",
            f"Pages: {document['pages']}",
            f"Total characters: {document['total_characters']}",
            (
                "Characters/page: "
                f"mean={document['mean_characters_per_page']}, "
                f"median={document['median_characters_per_page']}, "
                f"min={document['min_characters_per_page']}, "
                f"max={document['max_characters_per_page']}"
            ),
            (
                "Near-empty pages: "
                f"{len(document['low_text_pages'])}/"
                f"{document['pages']} "
                f"({document['near_empty_page_ratio']:.1%})"
            ),
            (
                "Latin pages: "
                f"{len(document['latin_pages'])}/"
                f"{document['pages']}"
            ),
            (
                "Devanagari pages: "
                f"{len(document['devanagari_pages'])}/"
                f"{document['pages']}"
            ),
            (
                "Numeric pages: "
                f"{len(document['numeric_pages'])}/"
                f"{document['pages']}"
            ),
            (
                "Numeric-heavy pages: "
                f"{document['numeric_heavy_pages']}"
            ),
            (
                "Possible corruption pages: "
                f"{document['possible_corruption_pages']}"
            ),
            (
                "Mixed-script pages: "
                f"{document['mixed_script_pages']}"
            ),
        ]
    )


def main() -> None:
    """Run real-data OCR quality analysis."""
    report = build_report(DEFAULT_INPUT_DIR)

    DEFAULT_OUTPUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summaries = [
        format_document_summary(document)
        for document in report["documents"]
    ]

    text_output = (
        "REAL-DATA OCR QUALITY REPORT\n"
        + "=" * 80
        + "\n\n"
        + "\n\n".join(summaries)
        + "\n"
    )

    DEFAULT_OUTPUT_TEXT.write_text(
        text_output,
        encoding="utf-8",
    )

    print(text_output)
    print(f"JSON report: {DEFAULT_OUTPUT_JSON}")
    print(f"Text report: {DEFAULT_OUTPUT_TEXT}")


if __name__ == "__main__":
    main()