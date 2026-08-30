from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from ocr_eval.critical_fields import evaluate_fields
from ocr_eval.extraction import extract_critical_fields


DATA_ROOT = Path("data/development/synthetic_documents")
OCR_ROOT = Path("outputs/raw")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    documents = ["doc_001", "doc_002", "doc_003"]

    for document_id in documents:
        print()
        print("=" * 80)
        print(document_id)
        print("=" * 80)

        ground_truth_path = (
            DATA_ROOT
            / document_id
            / "ground_truth"
            / "fields.json"
        )

        ocr_candidates = sorted(
            OCR_ROOT.glob(f"{document_id}_*.json")
        )

        if not ocr_candidates:
            print("ERROR: No OCR output found")
            continue

        # Prefer the GPU final output where available.
        preferred = [
            path
            for path in ocr_candidates
            if "gpu_final" in path.name
        ]

        ocr_path = preferred[0] if preferred else ocr_candidates[0]

        ground_truth = load_json(ground_truth_path)
        ocr_output = load_json(ocr_path)

        ocr_text = ocr_output["full_text"]

        extracted = extract_critical_fields(ocr_text)

        evaluation = evaluate_fields(
            ground_truth["fields"],
            extracted,
        )

        print(f"OCR source: {ocr_path}")
        print(f"Difficulty:  {ground_truth['difficulty']}")
        print()

        print("SUMMARY")
        print("-" * 80)
        print(f"Total fields:       {evaluation['total_fields']}")
        print(f"Correct fields:     {evaluation['correct_fields']}")
        print(f"Exact matches:      {evaluation['exact_matches']}")
        print(f"Normalized matches: {evaluation['normalized_matches']}")
        print(f"Mismatches:         {evaluation['mismatches']}")
        print(f"Missing:            {evaluation['missing']}")
        print(f"Accuracy:           {evaluation['accuracy']:.4f}")

        print()
        print("EXTRACTED FIELDS")
        print("-" * 80)
        print(
            json.dumps(
                extracted,
                indent=2,
                ensure_ascii=False,
            )
        )

        failures = evaluation.get("failures", [])

        print()
        print("FAILURES")
        print("-" * 80)

        if not failures:
            print("None")
        else:
            for failure in failures:
                print(
                    f"{failure['field']}: "
                    f"{failure['status']}"
                )
                print(f"  reference:  {failure['reference']!r}")
                print(f"  hypothesis: {failure['hypothesis']!r}")


if __name__ == "__main__":
    main()