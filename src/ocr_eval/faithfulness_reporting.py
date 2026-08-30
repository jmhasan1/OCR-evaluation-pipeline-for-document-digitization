"""Token-level reporting for OCR faithfulness comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

from .normalization import normalize_text


@dataclass(frozen=True)
class TokenDifference:
    """A deterministic token-level difference."""

    operation: str
    reference: tuple[str, ...]
    hypothesis: tuple[str, ...]
    reference_start: int
    reference_end: int
    hypothesis_start: int
    hypothesis_end: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class FaithfulnessReport:
    """Human-oriented structured report for a faithfulness comparison."""

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
    differences: tuple[TokenDifference, ...]

    @property
    def difference_count(self) -> int:
        """Return the number of substantive token-level differences."""
        return len(self.differences)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _tokenize(text: str) -> list[str]:
    """Normalize and tokenize text using whitespace boundaries."""
    return normalize_text(text).split()


def _token_differences(
    reference_tokens: list[str],
    hypothesis_tokens: list[str],
) -> tuple[TokenDifference, ...]:
    """Return deterministic token-level differences."""
    matcher = SequenceMatcher(
        a=reference_tokens,
        b=hypothesis_tokens,
        autojunk=False,
    )

    differences: list[TokenDifference] = []

    for (
        operation,
        reference_start,
        reference_end,
        hypothesis_start,
        hypothesis_end,
    ) in matcher.get_opcodes():
        if operation == "equal":
            continue

        differences.append(
            TokenDifference(
                operation=operation,
                reference=tuple(
                    reference_tokens[reference_start:reference_end]
                ),
                hypothesis=tuple(
                    hypothesis_tokens[hypothesis_start:hypothesis_end]
                ),
                reference_start=reference_start,
                reference_end=reference_end,
                hypothesis_start=hypothesis_start,
                hypothesis_end=hypothesis_end,
            )
        )

    return tuple(differences)


def build_faithfulness_report(
    reference: str,
    hypothesis: str,
) -> FaithfulnessReport:
    """Build a token-level report from a faithfulness comparison."""
    if not isinstance(reference, str):
        raise TypeError("reference must be a string")

    if not isinstance(hypothesis, str):
        raise TypeError("hypothesis must be a string")

    from .faithfulness import compare_faithfulness

    comparison = compare_faithfulness(reference, hypothesis)

    reference_tokens = _tokenize(reference)
    hypothesis_tokens = _tokenize(hypothesis)

    return FaithfulnessReport(
        status=comparison.status,
        reference_text=reference,
        hypothesis_text=hypothesis,
        normalized_reference=comparison.normalized_reference,
        normalized_hypothesis=comparison.normalized_hypothesis,
        exact_match=comparison.exact_match,
        normalized_match=comparison.normalized_match,
        cer=comparison.cer,
        wer=comparison.wer,
        character_distance=comparison.character_distance,
        word_distance=comparison.word_distance,
        differences=_token_differences(
            reference_tokens,
            hypothesis_tokens,
        ),
    )