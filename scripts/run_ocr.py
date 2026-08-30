"""CLI: PDF/image -> rendered pages -> PaddleOCR -> JSON."""

import argparse
import json
import time
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from ocr.paddleocr_adapter import PaddleOCRAdapter
from ocr_eval.pipeline import run_document_ocr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu:0"], default="auto")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    init_start = time.perf_counter()
    engine = PaddleOCRAdapter(lang=args.lang, device=args.device)
    init_seconds = time.perf_counter() - init_start

    infer_start = time.perf_counter()
    document = run_document_ocr(
        args.input,
        engine,
        document_id=Path(args.input).stem,
        dpi=args.dpi,
    )
    inference_seconds = time.perf_counter() - infer_start

    payload = document.to_dict()
    payload["timing"] = {
        "initialization_seconds": round(init_seconds, 4),
        "document_inference_seconds": round(inference_seconds, 4),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(json.dumps(payload["config"]["runtime"], indent=2))
    print(json.dumps(payload["timing"], indent=2))
    print(f"Pages OCR'd: {len(document.pages)}")
    print(f"Saved OCR output to: {output}")


if __name__ == "__main__":
    main()
