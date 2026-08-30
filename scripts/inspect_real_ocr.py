"""Compact inspection of real-data OCR output."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT_ROOT = Path("outputs/assignment")


def main() -> None:
    for path in sorted(OUTPUT_ROOT.glob("*.json")):
        if path.name == "ocr_quality_report.json":
            continue

        data = json.loads(path.read_text(encoding="utf-8"))

        print()
        print("=" * 80)
        print(path.name)
        print("=" * 80)

        for page in data.get("pages", []):
            text = page.get("text", "")
            preview = " ".join(text.split())[:250]

            print(
                f"PAGE {page.get('page_number')}: "
                f"{len(text)} chars | {preview}"
            )


if __name__ == "__main__":
    main()