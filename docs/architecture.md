# Architecture

## Current system

The project separates OCR execution, evaluation, robustness, and reporting.

```text
                    PDF / images
                         │
                         ▼
                  PDF/image utilities
                         │
                         ▼
                 OCR adapter boundary
                         │
                         ▼
                   PaddleOCR adapter
                         │
                         ▼
                    OCRDocument
                  ┌──────┴──────┐
                  │             │
              page text      regions
                  │          confidence
                  │          bounding boxes
                  └──────┬──────┘
                         ▼
                    Evaluation
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       CER/WER       critical fields  faithfulness
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                controlled degradation
                         │
                         ▼
                    robustness
                         │
                         ▼
                   error analysis
                         │
                         ▼
                     reports
```

## Boundaries

### `src/ocr/`

Engine integration.

### `src/ocr_eval/`

Reusable evaluation primitives:

- schema
- normalization
- edit distance
- metrics
- extraction
- critical fields
- faithfulness
- degradation
- PDF utilities
- document pipeline

### `scripts/`

Workflow orchestration and CLI entry points.

### `tests/`

Executable contracts and regression coverage.

## Design principle

The evaluation framework should remain usable if the OCR engine changes. Conversely, OCR execution should not need to know how CER/WER, field accuracy, robustness, or error analysis are calculated.

## Data boundary

Synthetic development data supplies authoritative ground truth.

Assignment data is treated as real input and remains outside version control.

## Reporting boundary

Reports consume structured outputs from existing evaluation layers. Reporting does not redefine the underlying accuracy metrics.
