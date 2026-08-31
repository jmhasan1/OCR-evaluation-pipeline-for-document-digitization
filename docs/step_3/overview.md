# Step 3 — Evaluation Framework

Step 3 builds the evaluation stack on top of the stable OCR representation.

```text
OCR output
    ↓
normalization
    ↓
CER / WER / edit operations
    ↓
critical-field extraction
    ↓
field-level evaluation
    ↓
faithfulness comparison
    ↓
controlled degradation
    ↓
robustness analysis
    ↓
error categorization
    ↓
consolidated reporting
```

The stages answer different questions and are intentionally not collapsed into one score.

## Implemented subsystems

- **3.1** — text normalization and OCR metrics
- **3.2** — critical-field extraction and evaluation
- **3.3** — faithfulness and human-readable/token-level reporting
- **3.4** — real-data OCR quality evaluation
- **3.5** — controlled robustness testing
- **3.6** — error analysis and reporting
- **3.7** — Final evaluation & evaluator-facing evidence

The complete regression suite currently passes all **165 tests** in the development environment.
