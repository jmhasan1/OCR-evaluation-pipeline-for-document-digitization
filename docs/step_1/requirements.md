# Step 1 — Evaluation Requirements

The implementation was shaped around five evaluator-facing areas.

| Area | Weight | Required evidence |
|---|---:|---|
| Ground Truth Quality | 15% | Authoritative synthetic references and validation |
| Metrics & Automation | 25% | CER/WER, field-level evaluation, faithfulness |
| Robustness Testing | 15% | Controlled degradation and impact analysis |
| Error Analysis & Reporting | 15% | Categorized failures, examples, consequences, reports |
| Testing Strategy & Thinking | 30% | Unit/integration/adversarial tests, reproducibility, production awareness |

## Functional requirements

1. Render PDFs into deterministic page images.
2. Isolate the OCR engine behind an adapter.
3. Preserve page text, regions, confidence, and metadata.
4. Normalize text conservatively.
5. Calculate CER/WER and edit-operation counts.
6. Extract and evaluate critical fields.
7. Compare OCR/source text with downstream output for faithfulness.
8. Apply controlled degradations at known severities.
9. Measure degradation impact across multiple evaluation layers.
10. Produce machine-readable and human-readable reports.
11. Keep real assignment data separate from synthetic ground truth.
12. Make tests and evaluation behavior reproducible.

## Non-requirements

The project does not assume that:

- OCR confidence equals accuracy.
- CER/WER alone represent business correctness.
- synthetic degradation represents the real OCR error distribution.
- real assignment documents can be scored with CER/WER without authoritative ground truth.
