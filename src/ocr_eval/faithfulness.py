"""Deterministic OCR faithfulness comparison primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from difflib import SequenceMatcher

from .metrics import text_metrics
from .normalization import normalize_text


@dataclass(frozen=True)
class TextDifference:
    """A deterministic difference between reference and hypothesis text."""

    operation: str
    reference: str
    hypothesis: str
    reference_start: int
    reference_end: int
    hypothesis_start: int
    hypothesis_end: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)

def _structured_diff(
    reference: str,
    hypothesis: str,
) -> list[TextDifference]:
    """Return deterministic character-level differences.

    The comparison is performed on normalized text because formatting-only
    differences should not produce substantive diff evidence.

    SequenceMatcher operations are converted into a stable public contract:
    equal blocks are omitted; insert/delete/replace blocks are retained.
    """
    matcher = SequenceMatcher(
        a=reference,
        b=hypothesis,
        autojunk=False,
    )

    differences: list[TextDifference] = []

    for tag, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
        if tag == "equal":
            continue

        differences.append(
            TextDifference(
                operation=tag,
                reference=reference[ref_start:ref_end],
                hypothesis=hypothesis[hyp_start:hyp_end],
                reference_start=ref_start,
                reference_end=ref_end,
                hypothesis_start=hyp_start,
                hypothesis_end=hyp_end,
            )
        )

    return differences


@dataclass(frozen=True)
class FaithfulnessResult:
    """Result of comparing reference text with OCR hypothesis text.

    The result deliberately separates:
    - exact equality,
    - equality after permitted normalization,
    - substantive differences,
    - OCR error metrics.

    Later steps can extend the result with structured differences and
    risk/consequence flags without changing the basic comparison contract.
    """

    status: str
    reference_text: str
    hypothesis_text: str
    normalized_reference: str
    normalized_hypothesis: str
    exact_match: bool
    normalized_match: bool
    cer: float | None
    wer: float | None
    character_distance: int
    word_distance: int
    differences: tuple[TextDifference, ...] = ()
    risk_flags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def compare_faithfulness(
    reference: str,
    hypothesis: str,
) -> FaithfulnessResult:
    """Compare OCR text against reference text.

    Status semantics:

    ``exact_match``
        Reference and hypothesis are byte-for-byte equal as Python strings.

    ``normalized_match``
        The raw strings differ, but the configured normalization produces
        identical text.

    ``mismatch``
        The normalized texts differ.

    Empty-reference behavior follows the existing CER/WER contract:
    a non-empty hypothesis is a mismatch with undefined CER/WER.
    """
    if not isinstance(reference, str):
        raise TypeError("reference must be a string")

    if not isinstance(hypothesis, str):
        raise TypeError("hypothesis must be a string")

    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)

    exact_match = reference == hypothesis
    normalized_match = normalized_reference == normalized_hypothesis

    if exact_match:
        status = "exact_match"
    elif normalized_match:
        status = "normalized_match"
    else:
        status = "mismatch"

    metrics = text_metrics(reference, hypothesis)

    differences = tuple(
        _structured_diff(
            normalized_reference,
            normalized_hypothesis,
        )   
    )

    return FaithfulnessResult(
        status=status,
        reference_text=reference,
        hypothesis_text=hypothesis,
        normalized_reference=normalized_reference,
        normalized_hypothesis=normalized_hypothesis,
        exact_match=exact_match,
        normalized_match=normalized_match,
        cer=metrics["cer"],
        wer=metrics["wer"],
        character_distance=metrics["characters"]["distance"],
        word_distance=metrics["words"]["distance"],
        differences=differences,
    )