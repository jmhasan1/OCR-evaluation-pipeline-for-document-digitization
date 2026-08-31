# OCR Document Evaluation Pipeline

A development-first evaluation framework for OCR and document-AI systems handling scanned official documents such as sale deeds and land records.

The project is designed around **evaluation rigor rather than OCR accuracy alone**, with emphasis on meaningful metrics, critical-field correctness, faithfulness, controlled robustness testing, reproducibility, automation, error analysis, and awareness of document-processing risks.

> **Important:** `data/development/` contains fictional synthetic documents created only to develop and test the pipeline. They are not real land records.

> **Important:** Assignment(Real Users)-provided assignment documents are kept outside version control and are never treated as synthetic ground truth.

---
## Project progression

### Step 1 — Evaluation objectives and repository foundation

Step 1 established the methodological and engineering basis for the project.

Implemented and defined:

- evaluation objectives beyond a single OCR-accuracy score
- separation of OCR quality, critical-field correctness, faithfulness, and runtime performance
- synthetic-development-data policy and real-data ground-truth boundaries
- repository structure using `src/`, `scripts/`, `tests/`, `data/`, `docs/`, and `outputs/`
- reproducibility and dependency-management conventions
- evaluator-facing functional and non-functional requirements

The central methodological boundary is:

```text
Synthetic development data
    → authoritative references for deterministic evaluation

Assignment-provided real data
    → observable OCR-quality evidence only when authoritative references are unavailable

```
---

## Step 2 — Project foundation + OCR pipeline

Implemented:

- PDF-to-image rendering
- PaddleOCR adapter
- Stable normalized OCR document schema
- CPU/GPU/`auto` device selection
- CLI for OCR inference
- Raw OCR JSON output
- Synthetic development documents for controlled validation

The OCR engine is isolated behind an adapter so alternative OCR engines can be introduced without rewriting the evaluation layer.

### Step 2.5 — CPU/GPU portability and benchmarking

Implemented:

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

The OCR layer produces a stable internal representation containing document metadata, page text, detected regions, confidence information where available, and execution metadata.
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
Reference: Khasra 54/1
OCR:       Khasra 541
```

The missing `/` remains an evaluation error rather than being silently repaired.

This distinction is important for document identifiers where punctuation and numeric structure may carry meaning.

### Step 3.2 — Critical-field extraction and evaluation

The pipeline evaluates fields that are particularly important for document digitization workflows:

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

The extractor does not silently repair corrupted OCR identifiers.

#### Critical-field evaluation

Field evaluation distinguishes:

- `exact_match`
- `normalized_match`
- `mismatch`
- `missing`

This prevents a high aggregate score from hiding failures in important document identifiers.

For example:

```text
Reference:
REG-2025-00125

OCR:
REG-2025-0012S
```

is treated as a field failure.

### Adversarial extraction testing

The extraction layer includes regression tests covering:

- corrupted identifiers
- missing fields
- nearby Khasra annotations
- multiple sellers
- whitespace variation
- date-format variation
- preservation of identifier punctuation
- avoidance of silently "correcting" OCR output

These tests validate evaluator behavior and failure visibility, not merely extraction success on clean examples.

### Step 3.3 — Faithfulness / rewrite detection

The project includes a separate faithfulness evaluation layer for comparing source OCR text with downstream document-AI output.

Implemented:

- structured faithfulness result contract
- source/output text comparison
- normalized comparison
- token-level comparison
- difference reporting
- human-readable reporting
- machine-readable reporting
- adversarial faithfulness tests

The faithfulness layer is deliberately separate from OCR accuracy.

**OCR accuracy:**  
"Did OCR recognize the source correctly?"

**Faithfulness:**  
"Does the downstream output remain supported by the source?"

This distinction allows the evaluation framework to identify downstream rewriting or altered information even when the original OCR output itself is not being re-evaluated.

### Step 3.4 — Real-data OCR evaluation

The evaluation workflow has been exercised against the Assignment(Real Users)-provided real documents while keeping those documents outside version control.

Two real assignment PDFs were evaluated:

| Document | Pages |
|---|---:|
| `DocScanner Apr 21_ 2026 6-48 PM.pdf` | 22 |
| `MP22IGR17182025A100906565 (1).pdf` | 31 |

The current OCR outputs provide the following observable signals:

| Document | Pages | Characters | Words | Mean OCR confidence |
|---|---:|---:|---:|---:|
| DocScanner | 22 | 3,087 | 616 | 0.5329 |
| MP22IGR | 31 | 10,990 | 1,991 | 0.6404 |

These measurements are OCR-quality observations, not claims of transcription accuracy.

Authoritative transcriptions or critical-field ground truth are unavailable for the assignment documents. Therefore the project does not claim CER/WER or field-level accuracy for these documents.

The real-data evaluation reports:

- page count
- character count
- word count
- characters/page
- words/page
- confidence distributions
- confidence percentiles
- low-confidence pages
- low-confidence regions
- near-empty pages
- OCR initialization time
- OCR inference time
- throughput

The real-data workflow therefore provides useful evidence about OCR behavior without manufacturing ground truth.

#### Assignment-data handling

Assignment documents are treated as external evaluation material. They are:

- not committed to Git
- not used as synthetic ground truth
- not included in the synthetic robustness experiment
- not represented as authoritative transcriptions unless such ground truth is actually available

### Step 3.5 — Controlled robustness testing

The project includes a deterministic controlled-degradation framework for measuring how OCR-like corruption propagates through the evaluation stack.

#### Step 3.5.1 — Controlled degradation framework

Implemented degradation primitives include:

- character substitution
- character deletion
- character insertion
- whitespace corruption
- punctuation corruption
- identifier corruption

The framework supports deterministic severity-controlled transformations and records changed positions.

The design deliberately separates:

```text
reference text
    ↓
controlled degradation
    ↓
degraded text
    ↓
evaluation
```

This makes the robustness experiment reproducible and allows individual degradation behaviors to be tested independently.

#### Step 3.5.2 — Measurable robustness experiment

The current robustness experiment contains:

```text
3 synthetic documents
×
6 degradation types
×
5 severity levels
=
90 cases
```

Severity levels:

```text
0.0
0.25
0.5
0.75
1.0
```

The experiment therefore contains:

- 90 total cases
- 18 zero-severity control cases
- 72 degraded cases

Each case measures multiple evaluation layers:

```text
controlled degradation
        ↓
degraded text
        ↓
CER / WER
        ↓
critical-field evaluation
        ↓
faithfulness
        ↓
changed-position analysis
```

#### Control validation

The 18 zero-severity control cases establish that:

- CER = 0
- WER = 0
- critical-field accuracy = 1.0
- faithfulness CER = 0
- no text positions are changed

This provides a deterministic baseline for the experiment.

#### Current aggregate robustness results

| Degradation | Mean CER | Mean WER | Mean Field Accuracy |
|---|---:|---:|---:|
| Character substitution | 0.4282 | 0.5158 | 0.4182 |
| Character deletion | 0.5022 | 0.5158 | 0.4182 |
| Character insertion | 0.4264 | 0.5128 | 0.4182 |
| Whitespace corruption | 0.0675 | 0.5896 | 0.4545 |
| Punctuation corruption | 0.0228 | 0.1342 | 0.5939 |
| Identifier corruption | 0.0013 | 0.0086 | 0.8909 |

These results demonstrate why OCR evaluation should not rely on a single aggregate metric.

For example:

- whitespace corruption produces relatively low CER but the highest WER
- punctuation corruption has low CER/WER but still reduces field accuracy
- identifier corruption produces very low aggregate text error while still causing measurable critical-field degradation

This illustrates that small text-level changes can have disproportionate impact on structured document information.

#### Robustness experiment scope

The robustness experiment is intentionally limited to the controlled synthetic development documents.

This is important because synthetic degradation requires known reference text.

Assignment(Real Users)-provided documents are therefore not artificially assigned ground truth merely to make the experiment possible.


### Step 3.6 — Error Analysis & Reporting

The project now consolidates the outputs of the existing evaluation layers into an evaluator-facing error-analysis report.

The implementation intentionally does not introduce another accuracy metric or duplicate the core evaluation architecture.

Instead, it consumes the existing:

- robustness results
- CER/WER evidence
- field-level evaluation
- faithfulness evaluation
- real-data OCR-quality observations

and turns them into categorized, interpretable evidence.

### Error categorization

Supported error categories include:

| Category | Purpose |
|---|---|
| `character_level` | Character substitutions, insertions, and deletions |
| `whitespace` | Token-boundary and spacing corruption |
| `punctuation` | Punctuation-related corruption |
| `identifier` | Structured identifier corruption |
| `field_level` | Critical-field extraction/evaluation failures |
| `faithfulness` | Downstream differences from source OCR |
| `real_data_observation` | Observable real-data OCR quality signals |

The categorization is deliberately consequence-oriented rather than treating every textual difference as equally important.

### Representative failure examples

The consolidated report selects representative cases from the existing robustness evidence.

Each representative example records available evidence such as:

- document
- degradation type
- severity
- CER where available
- WER where available
- field accuracy
- faithfulness CER
- changed positions where available
- critical-field consequences
- evaluator-facing interpretation

The report does not invent metrics that are unavailable in the underlying evaluation artifact.

This keeps the error analysis traceable to existing evidence.

### Critical-field consequences

The error-analysis layer explicitly connects OCR degradation with structured information loss.

This is important because:

```text
small text error
      ↓
may produce
      ↓
large field-level consequence
```

For example, a small punctuation or identifier change may have limited effect on aggregate CER/WER while changing the meaning or validity of a registration number, survey number, or other structured identifier.

The evaluator therefore sees not only how much text changed, but also what information may have been affected.

### Real-data observations

Real assignment documents are included only as observable OCR-quality evidence.

The consolidated report can surface:

- page counts
- character and word counts
- OCR confidence
- low-confidence pages
- low-confidence regions
- near-empty pages
- processing observations

The report explicitly preserves the limitation that the assignment documents do not have authoritative ground-truth transcriptions or critical-field labels.

Therefore:

```text
Real assignment data
        ↓
observable OCR quality
        ≠
verified transcription accuracy
```

Assignment documents are never presented as synthetic ground truth.

### Consolidated evaluator-facing report

The error-analysis script generates:

```text
outputs/reports/error_analysis.json
outputs/reports/error_analysis.txt
```

The JSON report provides machine-readable evidence.

The text report provides a concise human-readable interpretation suitable for evaluator review.

The report includes:

- evaluation scope
- error categories
- robustness-case counts
- representative robustness failures
- critical-field consequences
- real-data observations
- evaluator findings
- limitations

The report is designed to make the existing evidence easier to inspect without changing the underlying evaluation methodology.

### Step 3.7 — Final evaluation & evaluator-facing evidence

The final evaluation layer is complete. A single runner consolidates the existing evaluation layers without duplicating the core architecture:

```powershell
python scripts\run_evaluation.py
```

The runner produces:

```text
outputs/evaluation/evaluation_report.json
outputs/evaluation/evaluation_report.txt
```

The final consolidated evidence includes:

- synthetic CER/WER and per-document results
- critical-field consequences
- faithfulness evidence
- the 90-case robustness experiment
- real-data OCR-quality observations
- categorized error analysis
- evaluator findings and production-relevant interpretation
- limitations and ground-truth boundaries

Final validation baseline:

- 3 synthetic development documents
- mean CER: 0.0249
- mean WER: 0.0387
- mean critical-field accuracy: 1.0000
- mean faithfulness CER: 0.0249
- 90 robustness cases
- 2 real-data documents evaluated using observable OCR-quality signals only
- 189 automated tests passing

Generated reports remain local artifacts and are excluded from version control. The evaluator-facing methodology and final evidence are documented under `docs/evaluation/`.

---

## Current validation baseline

The project contains three controlled synthetic documents:

| Document | Difficulty | Critical-field accuracy |
|---|---|---:|
| `doc_001` | baseline | 100% |
| `doc_002` | moderate | 100% |
| `doc_003` | stress | 100% |

The current synthetic OCR → extraction → critical-field evaluation path successfully evaluates all 11 critical fields for each document.

> **Important:** These results validate the controlled synthetic development dataset. They are not claims of production OCR accuracy or real-world land-record performance.

The synthetic documents intentionally include conditions such as:

- handwritten annotations
- stamp/seal overlap
- mixed font sizes
- numeric identifiers
- spacing variation
- nearby text that could confuse field extraction

---

## Test suite

The project maintains a regression suite covering:

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
- faithfulness comparison
- faithfulness reporting
- controlled OCR degradation
- robustness experiment matrix
- robustness reporting
- synthetic-only robustness scope
- error categorization
- representative failure analysis
- critical-field consequence analysis
- real-data observation reporting
- consolidated error-analysis reporting

Current full-suite baseline:

**189 passed, 1 warning**

The robustness experiment has 16 dedicated regression tests covering:

- degradation configuration
- synthetic document discovery
- 90-case matrix structure
- degradation coverage
- severity coverage
- control behavior
- text changes
- positive error generation
- changed-position tracking
- result contract
- summary consistency
- report readability
- JSON serialization
- assignment-data exclusion

The Step 3.6 reporting layer is additionally covered by tests for:

- error categorization
- representative failure selection
- critical-field consequence reporting
- real-data observation handling
- consolidated report structure
- machine-readable and human-readable report generation

A warning from the Paddle runtime regarding the optional `ccache` dependency may appear during the test suite. It does not currently cause test failures.

Example full-suite command:

```bash
python -m pytest -q
```
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

Assignment(Real Users)-provided assignment documents are kept separately under:

```text
data/assignment/
```

These documents are not committed to the repository.

Generated assignment OCR outputs are also kept outside version control.

Generated evaluation outputs are treated as local artifacts unless explicitly selected as repository documentation artifacts.

---

## Project layout

```text
ocr_document_evaluation/
├── configs/
│   └── default.json
├── data/
│   ├── development/          # synthetic development data
│   └── assignment/           # Assignment(Real Users)-provided documents; not committed
├── docs/
├── outputs/
│   ├── benchmarks/           # selected benchmark artifacts
│   ├── assignment/           # generated real-data evaluation; ignored
│   ├── raw/                  # generated OCR outputs; ignored
│   ├── reports/              # generated error-analysis reports; ignored
│   └── robustness/           # generated robustness reports; ignored
├── scripts/
│   ├── analyze_errors.py
│   ├── analyze_real_ocr.py
│   ├── benchmark_ocr.py
│   ├── environment_check.py
│   ├── evaluate_fields.py
│   ├── evaluate_real_data.py
│   ├── inspect_real_ocr.py
│   ├── render_documents.py
│   ├── run_ocr.py
│   ├── run_robustness_experiment.py
│   ├── validate_extraction.py
│   └── validate_ground_truth.py
├── src/
│   ├── ocr/
│   │   ├── base.py
│   │   └── paddleocr_adapter.py
│   └── ocr_eval/
│       ├── critical_fields.py
│       ├── degradation.py
│       ├── edit_distance.py
│       ├── extraction.py
│       ├── faithfulness.py
│       ├── faithfulness_reporting.py
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

The project also maintains:

- `pyproject.toml`
- `uv.lock`

for reproducible environment management.

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

For development validation:

```powershell
python scripts\validate_extraction.py
```

This runs the critical-field extraction and evaluation path against the available synthetic development documents.

The validation output is intended to make field-level failures visible rather than reducing evaluation to a single opaque score.

---

## Validate synthetic ground truth

The repository includes a ground-truth validation script for checking the structure and consistency of the synthetic development dataset:

```powershell
python scripts\validate_ground_truth.py
```

This validation is intentionally focused on the synthetic dataset where authoritative reference material is available.

---

## Run error analysis and reporting

Run the consolidated evaluator-facing error-analysis report:

```powershell
python scripts\analyze_errors.py
```

The script generates:

```text
outputs/reports/error_analysis.json
outputs/reports/error_analysis.txt
```

The generated reports summarize existing evaluation evidence without introducing another accuracy metric.

---

## Run real-data OCR evaluation

After OCR outputs have been generated locally for assignment documents:

```powershell
python scripts\evaluate_real_data.py
```

The script generates machine-readable and human-readable OCR-quality reports under:

```text
outputs/assignment/
```

These reports contain observable OCR signals but do not claim CER/WER or field accuracy where authoritative assignment ground truth is unavailable.

---

## Run robustness experiment

Run the complete controlled synthetic robustness experiment:

```powershell
python scripts\run_robustness_experiment.py
```

The current experiment evaluates:

**90 cases**

and writes:

```text
outputs/robustness/robustness_results.json
outputs/robustness/robustness_results.txt
```

The generated outputs are local evaluation artifacts and are not required to be committed to the repository.

Run the dedicated regression tests:

```powershell
python -m pytest tests\test_robustness_experiment.py -v
```

Run the complete test suite:

```powershell
python -m pytest -q
```

---

## Design principles

### 1. Evaluation before optimization

The project prioritizes reliable evaluation methodology over maximizing a single OCR accuracy number.

### 2. Separate OCR from evaluation

OCR engines are isolated behind adapters. Evaluation operates on a stable internal representation.

### 3. Preserve meaningful errors

Identifiers, dates, punctuation, and numeric values can carry legal or operational significance.

The evaluator should not silently repair OCR output.

### 4. Missing is different from wrong

A missing critical field is explicitly represented and reported rather than being treated as an ordinary text mismatch.

### 5. Aggregate metrics are not sufficient

CER/WER and field accuracy provide useful summaries, but field-level failures and error categories are necessary to understand document-processing risk.

### 6. Faithfulness is a separate evaluation dimension

A downstream output can be fluent while still altering or omitting information from the source.

Faithfulness is therefore evaluated separately from OCR recognition quality.

### 7. Controlled degradation should be reproducible

Robustness experiments use deterministic transformations, explicit severity levels, and known synthetic references.

### 8. Synthetic results are controlled validation

Synthetic documents are useful for deterministic regression testing, but performance on them must not be presented as evidence of production performance.

### 9. Real-data evaluation must not manufacture ground truth

When authoritative assignment transcriptions or field labels are unavailable, the framework reports observable OCR-quality signals rather than inventing CER/WER or field accuracy.

### 10. Missing information is itself an evaluation constraint

The absence of ground truth is explicitly represented in the evaluation workflow rather than hidden.

### 11. Reproducibility matters

Configuration, test cases, benchmark artifacts, and development datasets are kept structured so evaluation behavior can be reproduced and compared over time.

### 12. Test the evaluator, not only the happy path

Adversarial tests intentionally exercise:

- corrupted identifiers
- missing fields
- misleading nearby annotations
- multiple sellers
- punctuation changes
- whitespace changes
- downstream text alterations
- controlled OCR degradation

The goal is to ensure that the evaluation framework exposes meaningful failures rather than accidentally correcting them.

---

## Evaluation strategy

The project is organized around the major evaluation dimensions relevant to OCR/document-AI systems:

| Evaluation area | Current evidence |
|---|---|
| Ground Truth Quality | Synthetic ground-truth validation + controlled development dataset |
| Metrics & Automation | CER, WER, edit distance, field-level evaluation, faithfulness |
| Robustness Testing | 90-case controlled degradation experiment |
| Error Analysis & Reporting | Error categorization, representative failures, critical-field consequences, consolidated reports|
| Testing Strategy & Thinking | Layered regression suite, adversarial tests, deterministic experiments, separation of real and synthetic data |

The framework intentionally avoids collapsing these dimensions into a single score.

---

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
| Faithfulness / rewrite detection | Complete |
| Faithfulness reporting | Complete |
| Real-data OCR evaluation | Complete for available assignment material |
| Synthetic ground-truth validation | Complete |
| Controlled degradation framework | Complete |
| 90-case robustness experiment | Complete |
| Robustness regression tests | Complete |
| Error analysis / consolidated reporting | Complete |
| Final evaluation runner | Complete |

---

## Repository hygiene

The following are intentionally excluded from version control:

```text
.venv/
.pytest_cache/
__pycache__/
outputs/raw/
outputs/assignment/
outputs/reports/
outputs/robustness/
data/assignment/
```

In particular, Assignment(Real Users)-provided assignment documents must remain outside the repository.

Synthetic development documents under `data/development/` may be version-controlled because they are intentionally created for reproducible testing.

---

## Final principle

The objective of this project is not simply to demonstrate that an OCR engine can produce text.

It is to demonstrate that an evaluation system can answer:

```text
How accurate is the OCR?
        ↓
Which text errors occurred?
        ↓
Which important fields were affected?
        ↓
Did downstream output remain faithful?
        ↓
How does performance degrade under controlled corruption?
        ↓
What failure modes create the greatest document-processing risk?
```

A reliable document-AI evaluation system should make those failures measurable, reproducible, explainable, and difficult to hide behind aggregate accuracy scores.