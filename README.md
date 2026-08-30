# OCR Document Evaluation Pipeline

A development-first evaluation framework for OCR and document-AI systems handling scanned official documents such as sale deeds and land records.

The project is designed around **evaluation rigor rather than OCR accuracy alone**, with emphasis on meaningful metrics, critical-field correctness, adversarial testing, reproducibility, automation, and awareness of document-processing risks.

> **Important:** `data/development/` contains fictional synthetic documents created only to develop and test the pipeline while the employer-provided samples are unavailable. They are not real land records and must not be presented as employer-provided documents.

---

## Current milestone

**Step 3.2 — Critical-field extraction and evaluation**

The project currently has an end-to-end synthetic validation path:

```text
PDF / image
    ↓
OCR
    ↓
normalized OCR representation
    ↓
critical-field extraction
    ↓
field-level evaluation
    ↓
structured evaluation results
```

The implementation deliberately keeps OCR, extraction, and evaluation separated so that failures can be attributed to the appropriate stage.

---

## Implemented

### Step 2 — Project foundation + OCR pipeline

- PDF-to-image rendering
- PaddleOCR adapter
- Stable normalized OCR document schema
- CPU/GPU/`auto` device selection
- CLI for OCR inference
- Raw OCR JSON output
- Synthetic development documents for controlled validation

### Step 2.5 — CPU/GPU portability and benchmarking

- Explicit CPU execution
- Explicit NVIDIA GPU execution
- Automatic device selection
- Environment diagnostics
- OCR initialization and inference timing
- Per-page latency and throughput measurements
- GPU memory measurement where available
- GPU smoke benchmarking
- Regression tests for device-selection behavior

Performance measurements are kept separate from OCR-quality measurements. A faster OCR configuration is not automatically considered a better or safer configuration.

### Step 2.5.1 — Project execution architecture

The repository uses a `src/` layout with repository scripts that bootstrap the source tree automatically.

Core execution flow:

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

The OCR engine is isolated behind an adapter so that alternative OCR engines can be introduced without rewriting the evaluation layer.

---

## Step 3 — Evaluation framework

### Step 3.1 — Text normalization and OCR error metrics

Implemented:

- Unicode-aware text normalization using NFKC
- Whitespace normalization
- Character Error Rate (CER)
- Word Error Rate (WER)
- Levenshtein edit-distance calculation
- Substitution / insertion / deletion breakdown
- Explicit handling of empty-reference cases
- Devanagari text support
- Identifier-sensitive evaluation

Normalization is intentionally conservative. It removes formatting noise such as inconsistent whitespace without silently correcting meaningful identifier differences.

For example:

```text
Reference:  Khasra 54/1
OCR:        Khasra 541
```

The missing `/` remains an evaluation error rather than being silently repaired.

### Step 3.2 — Critical-field extraction and evaluation

The pipeline now evaluates fields that are particularly important for document digitization workflows:

- owner / purchaser names
- father / husband names
- survey / plot / khasra numbers
- area and unit
- village
- tehsil
- district
- registration number and date
- mutation number and date

#### Critical-field extraction

The initial extractor is deliberately rule-based and label-aware rather than attempting to infer values from arbitrary document positions.

It supports:

- scalar fields
- list fields
- structured fields such as area + unit
- multiple document wording variants
- explicit `None` for missing fields
- preservation of OCR-extracted identifiers
- nearby annotation handling
- multiple-seller scenarios
- whitespace cleanup
- date-format preservation

The extractor does **not** silently repair corrupted OCR identifiers.

#### Critical-field evaluation

Field evaluation distinguishes:

- `exact_match`
- `normalized_match`
- `mismatch`
- `missing`

This prevents a high aggregate score from hiding failures in important document identifiers.

For example, an OCR result changing:

```text
REG-2025-00125
```

to:

```text
REG-2025-0012S
```

is treated as a field failure.

### Adversarial testing

The extraction layer includes regression tests covering cases such as:

- corrupted identifiers
- missing fields
- nearby Khasra annotations
- multiple sellers
- whitespace variation
- date-format variation
- preservation of identifier punctuation
- avoidance of silently "correcting" OCR output

These tests are intended to validate evaluator behavior and failure visibility, not merely maximize extraction success on clean examples.

---

## Current validation baseline

The synthetic development dataset contains three controlled documents:

| Document | Difficulty | Critical-field accuracy |
|---|---|---:|
| `doc_001` | baseline | 100% |
| `doc_002` | moderate | 100% |
| `doc_003` | stress | 100% |

The current synthetic OCR → extraction → critical-field evaluation path successfully evaluates all 11 critical fields for each document.

> **Important:** These results are a validation of the controlled synthetic development dataset. They are **not** a claim of production OCR accuracy or real-world land-record performance.

The synthetic documents intentionally include conditions such as:

- handwritten annotations
- stamp/seal overlap
- mixed font sizes
- numeric identifiers
- spacing variation
- nearby text that could confuse field extraction

---

## Test suite

The project currently maintains a regression suite covering:

- OCR pipeline utilities
- PDF rendering
- device selection
- benchmarking
- OCR schema serialization
- text normalization
- CER/WER
- edit-distance breakdown
- critical-field extraction
- critical-field evaluation
- adversarial extraction behavior

Current baseline:

**52 tests passing**

A warning from the Paddle runtime regarding the optional `ccache` dependency may appear during the test suite; it does not currently cause test failures.

---

## Development datasets

Synthetic development data is stored under:

```text
data/development/
```

with documents under:

```text
data/development/synthetic_documents/
```

The synthetic dataset is version-controlled because it is intentionally created for reproducible development and regression testing.

Employer-provided documents are kept separately under:

```text
data/assignment/
```

These documents are **not committed to the repository**.

They must be treated as external evaluation material and should not be represented as synthetic or employer-provided samples interchangeably.

---

## Project layout

```text
ocr_document_evaluation/
├── configs/
│   └── default.json
├── data/
│   ├── development/          # synthetic development data
│   └── assignment/           # employer-provided documents; not committed
├── docs/
├── outputs/
│   ├── benchmarks/           # selected benchmark artifacts
│   ├── raw/                  # generated OCR outputs; ignored
│   └── reports/              # generated reports; ignored
├── scripts/
│   ├── benchmark_ocr.py
│   ├── environment_check.py
│   ├── evaluate_fields.py
│   ├── render_documents.py
│   ├── run_ocr.py
│   └── validate_extraction.py
├── src/
│   ├── ocr/
│   │   ├── base.py
│   │   └── paddleocr_adapter.py
│   └── ocr_eval/
│       ├── critical_fields.py
│       ├── edit_distance.py
│       ├── extraction.py
│       ├── metrics.py
│       ├── normalization.py
│       ├── pdf_utils.py
│       ├── pipeline.py
│       └── schema.py
├── tests/
├── main.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
└── README.md
```

---

## Environment setup

Create a virtual environment first.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The project also maintains `pyproject.toml` and `uv.lock` for reproducible environment management.

PaddlePaddle may require a platform-specific CPU or GPU installation.

---

## Run the OCR pipeline

Render a PDF into page images and run OCR:

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

Require the first NVIDIA GPU:

```powershell
python scripts\run_ocr.py `
  --input data\development\synthetic_documents\doc_001\doc_001.pdf `
  --output outputs\raw\doc_001_gpu.json `
  --device gpu:0
```

The output contains document metadata, OCR engine/configuration, per-page text, detected regions, confidence values where available, and source image paths.

---

## Benchmark OCR execution

```powershell
python scripts\benchmark_ocr.py `
  --input data\development\synthetic_documents\doc_001\doc_001.pdf `
  --devices cpu gpu:0 auto `
  --repeats 3 `
  --warmup 1 `
  --output outputs\benchmarks\doc_001_benchmark.json
```

The benchmark measures inference performance independently from OCR quality.

For the current synthetic `doc_001` development benchmark, GPU execution was substantially faster than CPU execution on the development machine. This is a development measurement only and should not be treated as a general hardware-performance guarantee.

---

## Evaluate critical fields

The field evaluator compares extracted fields against a ground-truth `fields.json` file.

```powershell
python scripts\evaluate_fields.py `
  --reference data\development\synthetic_documents\doc_001\ground_truth\fields.json `
  --hypothesis <extracted-fields.json>
```

Optional JSON output:

```powershell
python scripts\evaluate_fields.py `
  --reference data\development\synthetic_documents\doc_001\ground_truth\fields.json `
  --hypothesis <extracted-fields.json> `
  --output outputs\reports\doc_001_field_evaluation.json
```

The evaluator reports:

- total fields
- correct fields
- exact matches
- normalized matches
- mismatches
- missing fields
- field-level failures
- aggregate field accuracy

---

## Validate OCR → extraction → evaluation

For development validation, the repository includes:

```powershell
python scripts\validate_extraction.py
```

This runs the critical-field extraction and evaluation path against the available synthetic development documents.

The validation output is intended to make field-level failures visible rather than reducing evaluation to a single opaque score.

---

## Design principles

### 1. Evaluation before optimization

The project prioritizes reliable evaluation methodology over maximizing a single OCR accuracy number.

### 2. Separate OCR from evaluation

OCR engines are isolated behind adapters. Evaluation operates on a stable internal representation.

### 3. Preserve meaningful errors

Identifiers, dates, punctuation, and numeric values can carry legal or operational significance. The evaluator should not silently repair OCR output.

### 4. Missing is different from wrong

A missing critical field is explicitly represented and reported rather than being treated as an ordinary text mismatch.

### 5. Aggregate metrics are not sufficient

CER/WER and field accuracy provide useful summaries, but field-level failures and error categories are necessary to understand document-processing risk.

### 6. Synthetic results are controlled validation

Synthetic documents are useful for deterministic regression testing, but performance on them must not be presented as evidence of production performance.

### 7. Reproducibility matters

Configuration, test cases, benchmark artifacts, and development datasets are kept structured so that evaluation behavior can be reproduced and compared over time.

---

## Next milestones

### Step 3.3 — Faithfulness / rewrite detection

Evaluate whether downstream document-AI output faithfully represents the source OCR/document content and detect unsupported rewriting or altered critical information.

### Step 3.4 — Controlled robustness degradation

Introduce controlled perturbations such as:

- character substitutions
- dropped characters
- whitespace corruption
- numeric corruption
- identifier punctuation corruption
- OCR confidence degradation
- layout/annotation interference

Measure whether the evaluation system detects the resulting failures.

### Step 3.5 — Dataset-level evaluation

Move from individual-document checks to aggregate evaluation across datasets, including:

- field-level accuracy
- CER/WER distributions
- failure rates by field
- failure rates by difficulty
- error-category breakdowns
- missing-field rates

### Step 3.6 — Reporting and automation

Add structured Excel/HTML reporting and an automated evaluation workflow suitable for repeated regression testing.

### Step 3.7 — Assignment dataset validation

Run the complete evaluation workflow against the employer-provided samples while keeping those documents outside the repository.

The final evaluation should explicitly distinguish:

- OCR errors
- extraction errors
- field-evaluation errors
- missing information
- potentially unsafe or misleading outputs

---

## Project status

| Area | Status |
|---|---|
| OCR pipeline | Complete |
| CPU/GPU portability | Complete |
| OCR benchmarking | Complete |
| Text normalization | Complete |
| CER/WER metrics | Complete |
| Critical-field extraction | Complete |
| Critical-field evaluation | Complete |
| Adversarial extraction tests | Complete |
| Synthetic end-to-end validation | Complete |
| Faithfulness / rewrite detection | Next |
| Robustness degradation | Planned |
| Dataset-level reporting | Planned |
| Excel/HTML reporting | Planned |
| Assignment-data evaluation | Pending |

The project is intentionally being developed incrementally so that each evaluation layer can be tested independently before being incorporated into the full document-evaluation workflow.