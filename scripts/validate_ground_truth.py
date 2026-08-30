"""Validate the quality and traceability of synthetic OCR ground truth."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "development"
    / "synthetic_documents"
)


@dataclass
class FieldCheck:
    """Result for one ground-truth field."""

    field: str
    field_type: str
    status: str
    values_checked: int
    values_found: int
    missing_values: list[str]


@dataclass
class DocumentAudit:
    """Ground-truth audit result for one document."""

    document_id: str
    difficulty: str
    difficulties: list[str]
    reference_text_exists: bool
    fields_file_exists: bool
    json_valid: bool
    field_checks: list[FieldCheck]
    passed: bool


def _display_value(value: Any) -> str:
    """Convert a ground-truth value to a readable string."""
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    return str(value)


def _check_text_value(
    field: str,
    value: Any,
    reference_text: str,
    field_type: str,
) -> FieldCheck:
    """Check whether a scalar value is traceable to reference text."""
    text_value = _display_value(value)

    if not text_value:
        return FieldCheck(
            field=field,
            field_type=field_type,
            status="SKIP",
            values_checked=0,
            values_found=0,
            missing_values=[],
        )

    found = text_value in reference_text

    return FieldCheck(
        field=field,
        field_type=field_type,
        status="PASS" if found else "FAIL",
        values_checked=1,
        values_found=1 if found else 0,
        missing_values=[] if found else [text_value],
    )


def _check_list(
    field: str,
    values: list[Any],
    reference_text: str,
) -> FieldCheck:
    """Check whether every list value is traceable to reference text."""
    missing: list[str] = []

    for value in values:
        text_value = _display_value(value)

        if text_value and text_value not in reference_text:
            missing.append(text_value)

    checked = len(values)
    found = checked - len(missing)

    return FieldCheck(
        field=field,
        field_type="list",
        status="PASS" if not missing else "FAIL",
        values_checked=checked,
        values_found=found,
        missing_values=missing,
    )


def _check_area(
    field: str,
    value: Any,
    reference_text: str,
) -> list[FieldCheck]:
    """Check the structured area field."""
    if not isinstance(value, dict):
        return [
            FieldCheck(
                field=field,
                field_type="nested_object",
                status="FAIL",
                values_checked=1,
                values_found=0,
                missing_values=["expected object with value and unit"],
            )
        ]

    checks: list[FieldCheck] = []

    for key in ("value", "unit"):
        child_value = value.get(key)

        if child_value is None:
            checks.append(
                FieldCheck(
                    field=f"{field}.{key}",
                    field_type="nested_scalar",
                    status="FAIL",
                    values_checked=1,
                    values_found=0,
                    missing_values=["missing annotation"],
                )
            )
            continue

        checks.append(
            _check_text_value(
                field=f"{field}.{key}",
                value=child_value,
                reference_text=reference_text,
                field_type="nested_scalar",
            )
        )

    return checks


def _audit_fields(
    fields: dict[str, Any],
    reference_text: str,
) -> list[FieldCheck]:
    """Audit all annotated critical fields."""
    checks: list[FieldCheck] = []

    for field, value in fields.items():
        if isinstance(value, list):
            checks.append(
                _check_list(
                    field=field,
                    values=value,
                    reference_text=reference_text,
                )
            )

        elif isinstance(value, dict):
            if field == "area":
                checks.extend(
                    _check_area(
                        field=field,
                        value=value,
                        reference_text=reference_text,
                    )
                )
            else:
                checks.append(
                    FieldCheck(
                        field=field,
                        field_type="nested_object",
                        status="SKIP",
                        values_checked=0,
                        values_found=0,
                        missing_values=[],
                    )
                )
        else:
            checks.append(
                _check_text_value(
                    field=field,
                    value=value,
                    reference_text=reference_text,
                    field_type="scalar",
                )
            )

    return checks


def audit_document(document_dir: Path) -> DocumentAudit:
    """Audit one synthetic document's ground truth."""
    document_id = document_dir.name

    ground_truth_dir = document_dir / "ground_truth"
    text_path = ground_truth_dir / "full_text.txt"
    fields_path = ground_truth_dir / "fields.json"

    reference_text_exists = text_path.is_file()
    fields_file_exists = fields_path.is_file()

    if not reference_text_exists or not fields_file_exists:
        return DocumentAudit(
            document_id=document_id,
            difficulty="unknown",
            difficulties=[],
            reference_text_exists=reference_text_exists,
            fields_file_exists=fields_file_exists,
            json_valid=False,
            field_checks=[],
            passed=False,
        )

    try:
        reference_text = text_path.read_text(encoding="utf-8")
        data = json.loads(fields_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return DocumentAudit(
            document_id=document_id,
            difficulty="unknown",
            difficulties=[],
            reference_text_exists=True,
            fields_file_exists=True,
            json_valid=False,
            field_checks=[],
            passed=False,
        )

    fields = data.get("fields")

    if not isinstance(fields, dict):
        return DocumentAudit(
            document_id=document_id,
            difficulty=str(data.get("difficulty", "unknown")),
            difficulties=data.get("difficulties", []),
            reference_text_exists=True,
            fields_file_exists=True,
            json_valid=True,
            field_checks=[],
            passed=False,
        )

    checks = _audit_fields(fields, reference_text)

    passed = all(check.status in {"PASS", "SKIP"} for check in checks)

    return DocumentAudit(
        document_id=document_id,
        difficulty=str(data.get("difficulty", "unknown")),
        difficulties=list(data.get("difficulties", [])),
        reference_text_exists=True,
        fields_file_exists=True,
        json_valid=True,
        field_checks=checks,
        passed=passed,
    )


def _print_document_report(audit: DocumentAudit) -> None:
    """Print a human-readable audit report."""
    print()
    print("=" * 80)
    print(audit.document_id)
    print("=" * 80)

    print(f"Difficulty: {audit.difficulty}")
    print(
        "Declared difficulties: "
        + (", ".join(audit.difficulties) if audit.difficulties else "none")
    )

    print()
    print("STRUCTURE")
    print("-" * 80)
    print(
        f"Reference text: {'PASS' if audit.reference_text_exists else 'FAIL'}"
    )
    print(
        f"Fields JSON:     "
        f"{'PASS' if audit.fields_file_exists and audit.json_valid else 'FAIL'}"
    )

    if not audit.field_checks:
        print("\nField coverage: unavailable")
        print(f"\nResult: {'PASS' if audit.passed else 'FAIL'}")
        return

    print()
    print("FIELD TRACEABILITY")
    print("-" * 80)

    for check in audit.field_checks:
        print(
            f"{check.field:<30} "
            f"{check.status:<4} "
            f"({check.field_type}) "
            f"{check.values_found}/{check.values_checked}"
        )

        for missing in check.missing_values:
            print(f"  MISSING: {missing!r}")

    print()
    print(f"Result: {'PASS' if audit.passed else 'FAIL'}")


def _print_dataset_summary(audits: list[DocumentAudit]) -> None:
    """Print aggregate dataset-level results."""
    total_fields = sum(
        len(audit.field_checks)
        for audit in audits
    )

    passed_fields = sum(
        sum(check.status == "PASS" for check in audit.field_checks)
        for audit in audits
    )

    print()
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(f"Documents:          {len(audits)}")
    print(
        f"Documents passing:  "
        f"{sum(audit.passed for audit in audits)}"
    )
    print(
        f"Documents failing:  "
        f"{sum(not audit.passed for audit in audits)}"
    )
    print(f"Field checks:        {total_fields}")
    print(f"Field checks passed: {passed_fields}")

    difficulties: dict[str, int] = {}

    for audit in audits:
        difficulties[audit.difficulty] = (
            difficulties.get(audit.difficulty, 0) + 1
        )

    print()
    print("Difficulty levels:")

    for difficulty, count in sorted(difficulties.items()):
        print(f"  {difficulty}: {count}")

    print()
    print(
        "GROUND TRUTH QUALITY: "
        + ("PASS" if all(audit.passed for audit in audits) else "FAIL")
    )


def main() -> int:
    """Run the synthetic ground-truth audit."""
    if not DATASET_ROOT.is_dir():
        print(f"Dataset directory not found: {DATASET_ROOT}")
        return 1

    document_dirs = sorted(
        path
        for path in DATASET_ROOT.iterdir()
        if path.is_dir()
    )

    if not document_dirs:
        print(f"No synthetic documents found in: {DATASET_ROOT}")
        return 1

    audits = [
        audit_document(document_dir)
        for document_dir in document_dirs
    ]

    for audit in audits:
        _print_document_report(audit)

    _print_dataset_summary(audits)

    return 0 if all(audit.passed for audit in audits) else 1


if __name__ == "__main__":
    sys.exit(main())