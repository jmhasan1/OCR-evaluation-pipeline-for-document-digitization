# Step 1 — Repository and Development Setup

## Repository layout

The project uses a `src/` layout:

```text
src/
├── ocr/
└── ocr_eval/
```

Repository-level executable workflows live under:

```text
scripts/
```

Tests live under:

```text
tests/
```

Development data is under:

```text
data/development/
```

Generated evaluation artifacts are under `outputs/`.

## Development data policy

Synthetic documents contain fictional development content and authoritative ground truth:

```text
data/development/synthetic_documents/
├── doc_001/
├── doc_002/
└── doc_003/
```

Assignment-provided documents are kept outside version control:

```text
data/assignment/
```

The repository `.gitignore` prevents those files from being committed.

## Reproducibility

The project pins the Python major/minor range and key OCR dependencies in `pyproject.toml`.

Tests use pytest and the repository source tree is exposed through pytest's configured Python path.

## Design principle

Scripts orchestrate workflows; reusable evaluation logic belongs in `src/ocr_eval/`; OCR-engine integration belongs in `src/ocr/`.
