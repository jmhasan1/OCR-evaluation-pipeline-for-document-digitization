# Step 2 — OCR Foundation and Execution

Step 2 establishes the OCR execution boundary and the stable representation consumed by later evaluation stages.

## Execution flow

```text
PDF / images
    ↓
PDF/image collection
    ↓
PaddleOCR adapter
    ↓
OCRPage
    ↓
OCRDocument
    ↓
JSON
```

The OCR layer is intentionally separated from evaluation. This allows the evaluation stack to remain independent of PaddleOCR-specific result structures.

## Core components

- `src/ocr/base.py` — OCR adapter boundary.
- `src/ocr/paddleocr_adapter.py` — PaddleOCR implementation.
- `src/ocr_eval/pdf_utils.py` — deterministic PDF/image collection.
- `src/ocr_eval/schema.py` — stable OCR document/page/region representation.
- `src/ocr_eval/pipeline.py` — document-level OCR orchestration.
- `scripts/run_ocr.py` — command-line OCR execution.

## Stable OCR representation

The schema preserves:

- document metadata
- page numbers
- page text
- OCR regions
- region confidence
- bounding boxes where available
- runtime metadata

This information is retained because later evaluation requires more than a final text string.
