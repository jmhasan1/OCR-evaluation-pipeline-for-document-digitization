"""Levenshtein edit-distance utilities for OCR evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class EditCounts:
    """Counts of Levenshtein operations."""

    substitutions: int
    insertions: int
    deletions: int

    @property
    def distance(self) -> int:
        """Total edit distance."""
        return self.substitutions + self.insertions + self.deletions


def levenshtein_counts(
    reference: Sequence[T],
    hypothesis: Sequence[T],
) -> EditCounts:
    """Calculate substitution, insertion and deletion counts.

    A dynamic-programming Levenshtein alignment is used. Ties are resolved
    deterministically, preferring substitutions, then deletions, then
    insertions.

    The function works for strings as well as arbitrary sequences.
    """

    n = len(reference)
    m = len(hypothesis)

    # DP cells contain:
    # (cost, substitutions, insertions, deletions)
    dp: list[list[tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(m + 1)]
        for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, 0, i)

    for j in range(1, m + 1):
        dp[0][j] = (j, 0, j, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue

            substitution = dp[i - 1][j - 1]
            deletion = dp[i - 1][j]
            insertion = dp[i][j - 1]

            candidates = [
                (
                    substitution[0] + 1,
                    substitution[1] + 1,
                    substitution[2],
                    substitution[3],
                ),
                (
                    deletion[0] + 1,
                    deletion[1],
                    deletion[2],
                    deletion[3] + 1,
                ),
                (
                    insertion[0] + 1,
                    insertion[1],
                    insertion[2] + 1,
                    insertion[3],
                ),
            ]

            # Stable ordering gives deterministic results when multiple
            # alignments have the same minimum cost.
            dp[i][j] = min(candidates, key=lambda item: item[0])

    _, substitutions, insertions, deletions = dp[n][m]

    return EditCounts(
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
    )