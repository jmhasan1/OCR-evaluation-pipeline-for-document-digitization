"""Critical-field evaluation for OCR output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .normalization import normalize_text


@dataclass(frozen=True)
class FieldEvaluation:
    field: str
    reference: Any
    hypothesis: Any
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_value(value: Any) -> Any:
    """Normalize scalar or list-valued field content."""
    if isinstance(value, str):
        return normalize_text(value)

    if isinstance(value, list):
        return [_normalize_value(item) for item in value]

    if isinstance(value, dict):
        return {
            key: _normalize_value(item)
            for key, item in value.items()
        }

    return value


def evaluate_field(
    field: str,
    reference: Any,
    hypothesis: Any,
) -> FieldEvaluation:
    """Evaluate one field while preserving meaningful punctuation."""
    if hypothesis is None:
        status = "missing"
    elif reference == hypothesis:
        status = "exact_match"
    elif _normalize_value(reference) == _normalize_value(hypothesis):
        status = "normalized_match"
    else:
        status = "mismatch"

    return FieldEvaluation(
        field=field,
        reference=reference,
        hypothesis=hypothesis,
        status=status,
    )


def evaluate_fields(
    reference_fields: dict[str, Any],
    hypothesis_fields: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate all reference fields and summarize the results."""
    evaluations = [
        evaluate_field(
            field=field,
            reference=reference,
            hypothesis=hypothesis_fields.get(field),
        )
        for field, reference in reference_fields.items()
    ]

    total = len(evaluations)
    exact_matches = sum(
        result.status == "exact_match"
        for result in evaluations
    )
    normalized_matches = sum(
        result.status == "normalized_match"
        for result in evaluations
    )
    mismatches = sum(
        result.status == "mismatch"
        for result in evaluations
    )
    missing = sum(
        result.status == "missing"
        for result in evaluations
    )

    correct = exact_matches + normalized_matches

    return {
        "total_fields": total,
        "correct_fields": correct,
        "exact_matches": exact_matches,
        "normalized_matches": normalized_matches,
        "mismatches": mismatches,
        "missing": missing,
        "accuracy": correct / total if total else None,
        "fields": [result.to_dict() for result in evaluations],
        "failures": [
            result.to_dict()
            for result in evaluations
            if result.status in {"mismatch", "missing"}
        ],
    }