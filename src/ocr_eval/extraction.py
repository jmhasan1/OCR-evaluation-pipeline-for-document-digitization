"""Deterministic extraction of critical fields from OCR text.

The extractor intentionally performs no fuzzy correction or semantic guessing.
Its responsibility is to convert OCR text into a structured hypothesis that can
subsequently be evaluated against ground truth by ``critical_fields.py``.

This design is particularly important for identifiers such as survey numbers,
registration numbers, and mutation numbers: an OCR error must remain visible
to the evaluation layer rather than being silently "corrected".
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Common patterns
# ---------------------------------------------------------------------------

_NUMBER = r"[0-9]+(?:[.,][0-9]+)?"

# Keep identifier characters deliberately conservative. In particular, do not
# normalize or remove "/" or "-" because those characters can be meaningful.
_IDENTIFIER = r"[A-Za-z0-9][A-Za-z0-9./_-]*"

_AREA_UNITS = (
    "Acre",
    "Acres",
    "Bigha",
    "Bighas",
    "Hectare",
    "Hectares",
)

_AREA_PATTERN = re.compile(
    rf"\bArea\s*:\s*(?P<value>{_NUMBER})\s+"
    rf"(?P<unit>{'|'.join(_AREA_UNITS)})\b",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _clean_value(value: str) -> str:
    """Normalize extraction whitespace without changing substantive content."""
    return " ".join(value.strip().split())


def _extract_label_value(
    text: str,
    labels: list[str],
) -> str | None:
    """Extract a single-line value following one of the supplied labels.

    Matching is case-insensitive and accepts either ``:`` or whitespace after
    the label. The extracted value is otherwise preserved.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    label_pattern = "|".join(re.escape(label) for label in labels)

    pattern = re.compile(
        rf"(?im)^[ \t]*(?:{label_pattern})[ \t]*(?::[ \t]*)?"
        rf"(?P<value>[^\r\n]+?)\s*$"
    )

    match = pattern.search(text)
    if match is None:
        return None

    value = _clean_value(match.group("value"))
    return value or None


def _extract_identifier(
    text: str,
    labels: list[str],
) -> str | None:
    """Extract an identifier following one of the supplied labels.

    Only the first contiguous identifier token is returned. No OCR correction
    is attempted.
    """
    value = _extract_label_value(text, labels)
    if value is None:
        return None

    match = re.match(_IDENTIFIER, value)
    return match.group(0) if match else None


def _extract_area(text: str) -> dict[str, str] | None:
    """Extract an area value and unit from an ``Area:`` line."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    match = _AREA_PATTERN.search(text)
    if match is None:
        return None

    return {
        "value": match.group("value"),
        "unit": match.group("unit"),
    }


# ---------------------------------------------------------------------------
# Party extraction
# ---------------------------------------------------------------------------


_PARTY_LINE_PATTERN = re.compile(
    r"(?im)^[ \t]*(?P<name>.+?),[ \t]*"
    r"(?:(?:son|daughter|wife|husband)[ \t]+of[ \t]+"
    r"(?P<relation>.+?))"
    r"(?:\.[ \t]*|,[ \t]*)$"
)


def _extract_party_names(
    text: str,
) -> tuple[list[str], list[str]]:
    """Extract purchaser/owner and father-husband names.

    The synthetic documents use several role headings:

    - ``PURCHASER``
    - ``TRANSFEREE / PURCHASER``

    Seller/transferor lines are deliberately ignored.

    Returns:
        ``(owner_names, father_husband_names)``
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    owner_names: list[str] = []
    father_husband_names: list[str] = []

    lines = text.splitlines()

    purchaser_heading = re.compile(
        r"^\s*(?:PURCHASER|TRANSFEREE\s*/\s*PURCHASER)\s*$",
        flags=re.IGNORECASE,
    )

    for index, line in enumerate(lines):
        if not purchaser_heading.match(line):
            continue

        # Inspect the next few lines only. This keeps seller information and
        # unrelated later sections outside the extraction scope.
        for candidate in lines[index + 1 : index + 4]:
            candidate = candidate.strip()

            if not candidate:
                continue

            # Stop if another section heading is encountered.
            if candidate.isupper() and len(candidate) < 80:
                break

            # Handle:
            #   Mohammad Hasan, son of Abdul Majid, residing at Rampur.
            #
            # and:
            #   Md. Imran Hossain, son of Jalal Hossain
            match = re.match(
                r"^(?P<name>.+?),\s*"
                r"(?P<relation_type>son|daughter|wife|husband)"
                r"\s+of\s+(?P<relation>[^,.]+)"
                r"(?:,|\.)?",
                candidate,
                flags=re.IGNORECASE,
            )

            if match:
                owner = _clean_value(match.group("name"))
                relation = _clean_value(match.group("relation"))

                if owner and owner not in owner_names:
                    owner_names.append(owner)

                if relation and relation not in father_husband_names:
                    father_husband_names.append(relation)

                continue

            # Support a purchaser line without a relationship clause.
            # Do not treat arbitrary following prose as a person's name.
            if "," not in candidate and not candidate.endswith(":"):
                if (
                    not candidate.startswith("[")
                    and not candidate.lower().startswith(
                        ("the ", "address:", "handwritten", "boundary")
                    )
                ):
                    owner = _clean_value(candidate)
                    if owner and owner not in owner_names:
                        owner_names.append(owner)

    return owner_names, father_husband_names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_critical_fields(text: str) -> dict[str, Any]:
    """Extract the critical fields used by the evaluation pipeline.

    The returned structure mirrors the ``fields`` object in the development
    ground-truth files.

    Missing scalar fields are represented by ``None``.
    Missing list fields are represented by ``[]``.

    No fuzzy matching, OCR correction, inference, or hallucinated values are
    performed.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    owner_names, father_husband_names = _extract_party_names(text)

    return {
        "owner_names": owner_names,
        "father_husband_names": father_husband_names,
        "survey_plot_khasra_number": _extract_identifier(
            text,
            [
                "Survey Number",
                "Survey / Plot No.",
                "Survey/Plot No.",
                "Khasra / Survey No.",
                "Khasra/Survey No.",
            ],
        ),
        "area": _extract_area(text),
        "village": _extract_label_value(text, ["Village"]),
        "tehsil": _extract_label_value(text, ["Tehsil"]),
        "district": _extract_label_value(text, ["District"]),
        "registration_number": _extract_identifier(
            text,
            [
                "Registration Number",
                "Registration No.",
            ],
        ),
        "registration_date": _extract_label_value(
            text,
            [
                "Registration Date",
            ],
        ),
        "mutation_number": _extract_identifier(
            text,
            [
                "Mutation Number",
                "Mutation No.",
            ],
        ),
        "mutation_date": _extract_label_value(
            text,
            [
                "Mutation Date",
            ],
        ),
    }