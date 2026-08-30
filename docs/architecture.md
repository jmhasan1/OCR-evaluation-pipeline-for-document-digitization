# Architecture — Step 2

## Principle

Separate **OCR execution** from **evaluation**.

The OCR adapter converts engine-specific results into a stable internal representation:

```text
PDF / images
    |
    v
PDF renderer
    |
    v
OCR adapter (PaddleOCR)
    |
    v
OCRDocument
  - page text
  - regions
  - confidence
  - bounding boxes
    |
    v
Evaluation layer (next milestone)
  - normalization
  - CER/WER
  - critical fields
  - faithfulness
  - robustness
```

## Why retain regions and confidence?

The assignment asks for production-oriented confidence-based human review. Keeping region-level confidence and bounding boxes now means the later evaluator can investigate critical fields and route uncertain cases without changing the OCR runner's output contract.

## Why render PDFs ourselves?

The evaluation must compare OCR against page-level source material and later generate controlled degradations. Converting PDFs to deterministic PNG pages gives the pipeline a stable image input boundary.
