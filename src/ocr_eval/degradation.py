"""Deterministic text degradations for controlled OCR robustness testing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Callable


MIN_SEVERITY = 0.0
MAX_SEVERITY = 1.0


@dataclass(frozen=True)
class DegradationResult:
    """Result of applying a controlled text degradation."""

    original_text: str
    degraded_text: str
    degradation: str
    severity: float
    changed_positions: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(frozen=True)
class DegradationSpec:
    """Metadata and implementation for one degradation strategy."""

    name: str
    description: str
    function: Callable[[str, float], str]


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")


def _validate_severity(severity: float) -> float:
    if isinstance(severity, bool) or not isinstance(severity, (int, float)):
        raise TypeError("severity must be a number")

    severity = float(severity)

    if not MIN_SEVERITY <= severity <= MAX_SEVERITY:
        raise ValueError("severity must be between 0.0 and 1.0")

    return severity


def _changed_positions(original: str, degraded: str) -> tuple[int, ...]:
    """Return positions affected by aligned character changes.

    Insertions/deletions are represented by the differing region. The result
    is intentionally conservative: it identifies positions in the original
    text that participate in a changed alignment.
    """
    positions: set[int] = set()

    common_length = min(len(original), len(degraded))

    for index in range(common_length):
        if original[index] != degraded[index]:
            positions.add(index)

    if len(original) > common_length:
        positions.update(range(common_length, len(original)))

    return tuple(sorted(positions))


def _replacement_count(length: int, severity: float) -> int:
    if length == 0 or severity == 0:
        return 0

    return max(1, min(length, round(length * severity)))


def _character_substitution(text: str, severity: float) -> str:
    """Replace deterministic non-whitespace characters with OCR-like variants."""
    if not text or severity == 0:
        return text

    candidates = [
        index
        for index, char in enumerate(text)
        if not char.isspace()
    ]

    count = _replacement_count(len(candidates), severity)
    selected = candidates[:count]

    replacements = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "3": "8",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
        "9": "g",
    }

    chars = list(text)

    for index in selected:
        original = chars[index]
        chars[index] = replacements.get(
            original,
            "X" if original != "X" else "Y",
        )

    return "".join(chars)


def _character_deletion(text: str, severity: float) -> str:
    """Deterministically remove non-whitespace characters."""
    if not text or severity == 0:
        return text

    candidates = [
        index
        for index, char in enumerate(text)
        if not char.isspace()
    ]

    count = _replacement_count(len(candidates), severity)
    remove = set(candidates[:count])

    return "".join(
        char
        for index, char in enumerate(text)
        if index not in remove
    )


def _character_insertion(text: str, severity: float) -> str:
    """Deterministically insert OCR-like noise after selected characters."""
    if not text or severity == 0:
        return text

    count = _replacement_count(len(text), severity)

    insertions = {
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
    }

    chars: list[str] = []

    for index, char in enumerate(text):
        chars.append(char)

        if index < count and not char.isspace():
            chars.append(insertions.get(char, "X"))

    return "".join(chars)


def _whitespace_corruption(text: str, severity: float) -> str:
    """Apply deterministic spacing corruption."""
    if not text or severity == 0:
        return text

    result = re.sub(r"[ \t]+", " ", text)

    if severity >= 0.5:
        result = result.replace(" ", "")

    if severity >= 0.75:
        result = result.replace("\n", " ")

    return result


def _punctuation_corruption(text: str, severity: float) -> str:
    """Remove selected punctuation characters."""
    if not text or severity == 0:
        return text

    punctuation = set("/-.,:;()")

    candidates = [
        index
        for index, char in enumerate(text)
        if char in punctuation
    ]

    count = _replacement_count(len(candidates), severity)
    remove = set(candidates[:count])

    return "".join(
        char
        for index, char in enumerate(text)
        if index not in remove
    )


def _identifier_corruption(text: str, severity: float) -> str:
    """Apply OCR-like corruption specifically inside identifiers/numbers."""
    if not text or severity == 0:
        return text

    chars = list(text)

    identifier_pattern = re.compile(
        r"[A-Za-z]{1,}[A-Za-z0-9./-]*\d[A-Za-z0-9./-]*"
    )

    matches = list(identifier_pattern.finditer(text))

    if not matches:
        return _character_substitution(text, severity)

    target_budget = max(1, round(len(matches) * severity))

    replacements = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "3": "E",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
        "9": "g",
    }

    changed = 0

    for match in matches:
        if changed >= target_budget:
            break

        start = match.start()
        end = match.end()

        for index in range(start, end):
            char = chars[index]

            if char.isdigit():
                chars[index] = replacements.get(char, "X")
                changed += 1
                break

            if char in "/-.":
                chars[index] = ""
                changed += 1
                break

    return "".join(chars)


DEGRADATIONS: dict[str, DegradationSpec] = {
    "character_substitution": DegradationSpec(
        name="character_substitution",
        description="Replace characters with deterministic OCR-like confusions.",
        function=_character_substitution,
    ),
    "character_deletion": DegradationSpec(
        name="character_deletion",
        description="Remove characters to simulate missed OCR characters.",
        function=_character_deletion,
    ),
    "character_insertion": DegradationSpec(
        name="character_insertion",
        description="Insert spurious characters to simulate OCR noise.",
        function=_character_insertion,
    ),
    "whitespace_corruption": DegradationSpec(
        name="whitespace_corruption",
        description="Alter spacing and line boundaries.",
        function=_whitespace_corruption,
    ),
    "punctuation_corruption": DegradationSpec(
        name="punctuation_corruption",
        description="Remove punctuation used by OCR-sensitive text.",
        function=_punctuation_corruption,
    ),
    "identifier_corruption": DegradationSpec(
        name="identifier_corruption",
        description="Corrupt digits or separators inside identifiers.",
        function=_identifier_corruption,
    ),
}


def list_degradations() -> list[dict[str, str]]:
    """Return available degradation strategies."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
        }
        for spec in DEGRADATIONS.values()
    ]


def apply_degradation(
    text: str,
    degradation: str,
    severity: float = 0.5,
) -> DegradationResult:
    """Apply a deterministic degradation to text.

    Severity is normalized to [0.0, 1.0]. It controls transformation
    intensity and is not itself an OCR error rate.
    """
    _validate_text(text)
    severity = _validate_severity(severity)

    if degradation not in DEGRADATIONS:
        available = ", ".join(sorted(DEGRADATIONS))
        raise ValueError(
            f"Unknown degradation '{degradation}'. "
            f"Available degradations: {available}"
        )

    degraded = DEGRADATIONS[degradation].function(text, severity)

    return DegradationResult(
        original_text=text,
        degraded_text=degraded,
        degradation=degradation,
        severity=severity,
        changed_positions=_changed_positions(text, degraded),
    )