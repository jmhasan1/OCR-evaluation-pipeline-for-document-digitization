# Step 2.5 — Changes from the uploaded Step 6 ZIP

The uploaded repository already separated OCR behind `src/ocr/` and used
PaddleOCR. Step 2.5 adds hardware portability and measurement without changing
the evaluation logic.

## 1. PaddleOCR adapter
`src/ocr/paddleocr_adapter.py`
- Added `auto`, `cpu`, and `gpu:0` device resolution.
- `auto` uses GPU only when Paddle reports CUDA support.
- Explicit GPU requests fail loudly if CUDA is unavailable.
- Added runtime metadata: requested/resolved device, GPU name, Paddle and
  PaddleOCR versions.

## 2. OCR CLI
`scripts/run_ocr.py`
- Added `--device`.
- Default is now `auto` instead of the previous CPU-oriented default.
- Measures initialization and inference time.
- Stores runtime information beside raw OCR output.

## 3. New benchmark
`scripts/benchmark_ocr.py`
- Runs the same image through CPU/GPU configurations.
- Uses warmup iterations to reduce first-run initialization effects.
- Reports mean/median/min/max latency.
- Records GPU memory allocated by Paddle when available.

## 4. New environment diagnostic
`scripts/environment_check.py`
- Reports OS, Python, Paddle CUDA support, GPU count, and resolved device.

## 5. New tests
`tests/test_device.py`
- Tests CPU selection.
- Tests GPU auto-selection.
- Tests CPU fallback.
- Tests explicit GPU failure.
- Tests invalid device handling.

## 6. Documentation
This change log explains the modifications and why they were made.

## What remains unchanged
The OCR evaluation algorithms and robustness/evaluation concepts are not
changed by Step 2.5. The purpose is hardware portability and measurement.