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
- CPU/GPU/`auto` device selection
- Environment diagnostics
- End-to-end OCR benchmarking
- CPU/GPU portability validation on a development laptop

Next milestone:
1. Step 3 — CER/WER, normalization, and critical-field evaluation
2. Step 4 — Faithfulness/rewrite detection
3. Step 5 — Controlled robustness degradation
4. Excel/HTML reporting
5. Continuous/regression testing strategy

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

The OCR runtime supports three execution modes:

- `auto` — use GPU when Paddle reports CUDA support, otherwise CPU
- `cpu` — force CPU execution
- `gpu:0` — require the first NVIDIA GPU

### Environment diagnostics

```powershell
python scripts\environment_check.py
```

### OCR execution

Automatic device selection:

```powershell
python scripts\run_ocr.py `
  --input data\development\synthetic_documents\doc_001\doc_001.pdf `
  --output outputs\raw\doc_001.json `
  --device auto
```

Force CPU:

```powershell
python scripts\run_ocr.py `
  --input data\development\synthetic_documents\doc_001\doc_001.pdf `
  --output outputs\raw\doc_001_cpu.json `
  --device cpu
```

Require NVIDIA GPU:

```powershell
python scripts\run_ocr.py `
  --input data\development\synthetic_documents\doc_001\doc_001.pdf `
  --output outputs\raw\doc_001_gpu.json `
  --device gpu:0
```

### Benchmark

```powershell
python scripts\benchmark_ocr.py `
  --input data\development\synthetic_documents\doc_001\doc_001.pdf `
  --devices cpu gpu:0 auto `
  --repeats 3 `
  --warmup 1 `
  --output outputs\benchmarks\doc_001_benchmark.json
```

The benchmark measures end-to-end OCR inference latency and GPU memory
separately from OCR accuracy. Performance alone does not establish OCR
quality or safety.

### Development benchmark result

> **Synthetic development benchmark — not employer-provided data**

The following measurements were obtained from the project's synthetic
2-page `doc_001` PDF at 200 DPI on a Windows development laptop with an
NVIDIA GeForce GTX 1650 Ti (4 GB VRAM).

| Requested device | Resolved device | Status | Pages | Mean inference | Median | Seconds/page | Pages/sec | Peak GPU memory |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `gpu:0` | `gpu:0` | `ok` | 2 | 3.635 s | 3.627 s | 1.817 | 0.5503 | 1351.81 MB |
| `auto` | `gpu:0` | `ok` | 2 | 3.754 s | 3.766 s | 1.877 | 0.5327 | 1351.81 MB |
| `cpu` | `cpu` | `ok` | 2 | 126.264 s | 127.226 s | 63.132 | 0.0158 | 0 MB |

For this particular synthetic document and development environment, the
GPU completed OCR inference approximately 34.7× faster than CPU execution.
This is a single-document development measurement and should not be
interpreted as a general GPU/CPU performance guarantee.

A separate GPU smoke benchmark also completed successfully using one
warm-up run and one measured run.

### Project execution architecture

The repository uses a `src/` layout. Repository scripts bootstrap `src`
automatically, so commands can be run directly from the project root.

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
```

The OCR engine is isolated behind an adapter, allowing the evaluation layer
to consume a stable internal representation rather than depending directly
on PaddleOCR's raw API.
