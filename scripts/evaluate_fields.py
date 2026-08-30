from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_eval.critical_fields import evaluate_fields


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate extracted OCR critical fields against ground truth."
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Ground-truth fields JSON file.",
    )
    parser.add_argument(
        "--hypothesis",
        required=True,
        type=Path,
        help="Extracted fields JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the evaluation JSON result.",
    )

    args = parser.parse_args()

    reference_data = json.loads(
        args.reference.read_text(encoding="utf-8")
    )
    hypothesis_data = json.loads(
        args.hypothesis.read_text(encoding="utf-8")
    )

    # Ground-truth files contain document metadata plus a "fields" object.
    reference_fields = reference_data.get("fields", reference_data)

    # Hypothesis may either be the fields object itself or a wrapper
    # containing "extracted_fields".
    hypothesis_fields = hypothesis_data.get(
        "extracted_fields",
        hypothesis_data,
    )

    result = evaluate_fields(reference_fields, hypothesis_fields)

    print("=" * 80)
    print("CRITICAL FIELD EVALUATION")
    print("=" * 80)
    print(f"Reference: {args.reference}")
    print(f"Hypothesis: {args.hypothesis}")
    print()
    print(f"Total fields:       {result['total_fields']}")
    print(f"Correct fields:     {result['correct_fields']}")
    print(f"Exact matches:      {result['exact_matches']}")
    print(f"Normalized matches: {result['normalized_matches']}")
    print(f"Mismatches:         {result['mismatches']}")
    print(f"Missing:            {result['missing']}")
    print(f"Accuracy:           {result['accuracy']:.4f}")

    if result["failures"]:
        print()
        print("FAILURES")
        print("-" * 80)
        for failure in result["failures"]:
            print(
                f"{failure['field']}: "
                f"{failure['status']} | "
                f"reference={failure['reference']!r} | "
                f"hypothesis={failure['hypothesis']!r}"
            )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print()
        print(f"Saved evaluation to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())