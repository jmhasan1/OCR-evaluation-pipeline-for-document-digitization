"""OCR quality metrics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .edit_distance import EditCounts, levenshtein_counts
from .normalization import normalize_text


def character_error_counts(
    reference: str,
    hypothesis: str,
) -> EditCounts:
    """Return character-level edit counts after normalization."""
    reference_normalized = normalize_text(reference)
    hypothesis_normalized = normalize_text(hypothesis)

    return levenshtein_counts(
        reference_normalized,
        hypothesis_normalized,
    )


def word_error_counts(
    reference: str,
    hypothesis: str,
) -> EditCounts:
    """Return word-level edit counts after normalization."""
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()

    return levenshtein_counts(
        reference_words,
        hypothesis_words,
    )


def _error_rate(
    counts: EditCounts,
    reference_length: int,
) -> float | None:
    """Calculate an error rate.

    Returns None when the reference denominator is zero and the hypothesis
    is non-empty, because the conventional error rate is undefined.
    """
    if reference_length == 0:
        return 0.0 if counts.distance == 0 else None

    return counts.distance / reference_length


def cer(
    reference: str,
    hypothesis: str,
) -> float | None:
    """Calculate Character Error Rate (CER)."""
    reference_normalized = normalize_text(reference)
    counts = character_error_counts(reference, hypothesis)

    return _error_rate(counts, len(reference_normalized))


def wer(
    reference: str,
    hypothesis: str,
) -> float | None:
    """Calculate Word Error Rate (WER)."""
    reference_words = normalize_text(reference).split()
    counts = word_error_counts(reference, hypothesis)

    return _error_rate(counts, len(reference_words))


def text_metrics(
    reference: str,
    hypothesis: str,
) -> dict[str, Any]:
    """Calculate CER, WER and their underlying edit counts."""
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)

    character_counts = levenshtein_counts(
        normalized_reference,
        normalized_hypothesis,
    )

    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()

    word_counts = levenshtein_counts(
        reference_words,
        hypothesis_words,
    )

    return {
        "cer": _error_rate(
            character_counts,
            len(normalized_reference),
        ),
        "wer": _error_rate(
            word_counts,
            len(reference_words),
        ),
        "characters": {
            "reference_count": len(normalized_reference),
            "hypothesis_count": len(normalized_hypothesis),
            **asdict(character_counts),
            "distance": character_counts.distance,
        },
        "words": {
            "reference_count": len(reference_words),
            "hypothesis_count": len(hypothesis_words),
            **asdict(word_counts),
            "distance": word_counts.distance,
        },
    }