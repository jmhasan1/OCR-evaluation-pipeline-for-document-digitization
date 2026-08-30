"""Text normalization utilities for OCR evaluation.

The normalization policy is intentionally language-neutral. It reduces
formatting noise without transliterating or otherwise altering the semantic
content of multilingual OCR text.
"""

from __future__ import annotations

import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize OCR/reference text for fair comparison.

    The transformation:
    - applies Unicode NFKC normalization
    - normalizes line endings
    - converts NBSP to a regular space
    - collapses consecutive whitespace
    - strips surrounding whitespace

    Important: this does not transliterate scripts, remove punctuation,
    remove digits, or otherwise rewrite document content.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings before generic whitespace handling.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Explicitly normalize non-breaking spaces.
    text = text.replace("\u00a0", " ")

    # Collapse spaces, tabs and newlines into a single space.
    text = _WHITESPACE_RE.sub(" ", text)

    return text.strip()