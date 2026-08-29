# OCR Document Evaluation Pipeline

A development-first evaluation framework for OCR and document-AI systems handling scanned official documents such as sale deeds and land records.

> **Important:** `data/development/` contains fictional synthetic documents created only to develop and test the pipeline while the employer-provided samples are unavailable. They are not real land records and must not be presented as employer-provided documents.

## Current milestone

**Step 2 — Project foundation + OCR pipeline**

Implemented:
- PDF page rendering to PNG
- OCR adapter interface
- PaddleOCR 3.x adapter
- Raw OCR JSON output schema
- CLI for running OCR over a PDF or image directory
- Basic smoke tests for PDF rendering and OCR output serialization

Next milestones:
1. CER/WER implementation
2. Critical-field exact match
3. Faithfulness/rewrite detection
4. Controlled robustness degradation
5. Excel/HTML reporting
6. Continuous/regression testing strategy

## Project layout

```text
ocr_document_evaluation/
├── configs/
├── data/
│   ├── development/       # synthetic development data
│   └── assignment/        # employer-provided documents will go here
├── docs/
├── outputs/
│   ├── raw/               # raw OCR outputs
│   └── reports/
├── scripts/
├── src/
│   ├── ocr/
│   └── ocr_eval/
├── tests/
├── requirements.txt
└── README.md
```

## Environment setup

Create a virtual environment first.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
```

Install core dependencies:

```bash
pip install -r requirements.txt
```

PaddlePaddle itself can require a platform-specific installation. Install the CPU or GPU build appropriate for the machine before running the PaddleOCR adapter.

## Run the OCR pipeline

Render a PDF into page images and run OCR:

```bash
python scripts/run_ocr.py \
  --input data/development/synthetic_documents/doc_001/doc_001.pdf \
  --output outputs/raw/doc_001.json
```

Or run OCR on an image directory:

```bash
python scripts/run_ocr.py \
  --input data/development/synthetic_documents/doc_001/source_images \
  --output outputs/raw/doc_001.json
```

The output is JSON containing document metadata, OCR engine/configuration, per-page text, detected regions, confidence values where available, and source image paths.

## Design principle

The OCR engine is deliberately isolated behind an adapter. The evaluation layer will consume a stable internal representation rather than depending on PaddleOCR's raw API. This lets us replace PaddleOCR with Tesseract, EasyOCR, AWS Textract, etc. without rewriting the evaluator.
## Step 2.5 — CPU/GPU Portability and Benchmarking

Run on any supported laptop with:

```powershell
python scripts/run_ocr.py --input <image> --output <json> --device auto
```

Force CPU:

```powershell
python scripts/run_ocr.py --input <image> --output <json> --device cpu
```

Require NVIDIA GPU:

```powershell
python scripts/run_ocr.py --input <image> --output <json> --device gpu:0
```

Check the environment:

```powershell
python scripts/environment_check.py
```

Benchmark:

```powershell
python scripts/benchmark_ocr.py --input <image> --devices cpu gpu:0 --repeats 3 --warmup 1
```

The benchmark measures inference speed and memory separately from OCR
accuracy; performance alone does not establish OCR quality or safety.
## Step 2.5.1 — Project Execution Architecture

The repository uses a `src/` layout. Repository scripts bootstrap `src`
automatically, so commands can be run directly from the project root.

### Environment check

```powershell
python scripts\environment_check.py
```

### OCR

```powershell
python scripts\run_ocr.py `
  --input data\development\synthetic_documents\doc_001\doc_001.pdf `
  --output outputs\raw\doc_001.json `
  --device auto
```

Force CPU with `--device cpu`; require NVIDIA GPU with `--device gpu:0`.

### Benchmark

```powershell
python scripts\benchmark_ocr.py `
  --input data\development\synthetic_documents\doc_001\source_images\page_01.png `
  --devices cpu gpu:0 `
  --repeats 3 `
  --warmup 1
```

The benchmark measures OCR inference latency and GPU memory. It does not
measure OCR accuracy.

### Pipeline

```text
PDF / image directory / image
        ↓
collect_images()
        ↓
PaddleOCRAdapter.recognize()
        ↓
OCRPage
        ↓
OCRDocument
        ↓
JSON