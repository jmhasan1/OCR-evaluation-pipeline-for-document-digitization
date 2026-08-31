# Evaluation Strategy

This document maps the implementation to the assignment evaluation criteria.

| Criterion | Weight | Evidence in repository |
|---|---:|---|
| Ground Truth Quality | 15% | Three synthetic documents with `full_text.txt` and `fields.json`; `validate_ground_truth.py` |
| Metrics & Automation | 25% | normalization, CER/WER, edit counts, critical fields, faithfulness, reporting |
| Robustness Testing | 15% | controlled degradation framework and 90-case experiment |
| Error Analysis & Reporting | 15% | `analyze_errors.py`, JSON/text reports, categorized findings |
| Testing Strategy & Thinking | 30% | unit/integration/adversarial tests, deterministic controls, portability, limitations, production-risk analysis |

## Evidence boundary

Synthetic documents provide authoritative ground truth.

Assignment documents provide real-world OCR observations but are not treated as ground truth.

This distinction should be preserved in all final reports and demonstrations.

## Evidence chain

```text
ground truth
    ↓
OCR output
    ↓
text metrics
    ↓
critical fields
    ↓
faithfulness
    ↓
controlled degradation
    ↓
error analysis
    ↓
evaluator-facing evidence
```

The framework is strongest when the individual layers are interpreted together rather than reduced to one headline score.
