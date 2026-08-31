# PaddleOCR Notes

The project currently targets PaddleOCR 3.7.0 and isolates it behind `PaddleOCRAdapter`.

The recorded development environment uses Paddle 3.3.0 and PaddleOCR 3.7.0. The Windows GPU environment uses Paddle's CUDA-enabled package configuration documented in `pyproject.toml`.

## Current integration boundary

PaddleOCR is responsible for OCR inference. The project converts its output into the internal `OCRDocument` schema before evaluation.

This prevents downstream evaluation code from depending directly on PaddleOCR result structures.

## Language and document considerations

The project does not assume that an OCR engine's confidence score is equivalent to transcription accuracy. Real documents are inspected using observable OCR signals, while accuracy claims require appropriate ground truth.

## Replacement strategy

If PaddleOCR is replaced, the intended change is concentrated in the OCR adapter layer. The evaluation modules under `src/ocr_eval/` should remain reusable.
