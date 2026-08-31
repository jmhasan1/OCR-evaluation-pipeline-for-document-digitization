# Final Evaluator Evidence

This document is a checklist for the final repository review.

## Ground truth

- Three synthetic development documents exist.
- Each has authoritative text and critical-field references.
- Ground truth is validated by `scripts/validate_ground_truth.py`.
- Assignment documents are excluded from synthetic ground truth.

## Metrics

Implemented and tested:

- conservative normalization
- CER
- WER
- substitution/insertion/deletion counts
- critical-field evaluation
- faithfulness comparison
- token-level difference reporting

## Robustness

Recorded experiment:

```text
3 documents × 6 degradation types × 5 severities = 90 cases
```

The experiment includes severity-zero controls and reports impact across multiple evaluation layers.

## Real data

Two assignment PDFs were processed.

The repository reports observable OCR signals while avoiding unsupported accuracy claims because authoritative ground truth is unavailable.

## Error analysis

The consolidated report connects:

```text
failure type
    ↓
metric impact
    ↓
field consequence
    ↓
production relevance
```

## Testing

Current development status:

```text
165 tests passed
```

The suite includes unit, integration, adversarial, deterministic-control, and exclusion-policy tests.

## Final interpretation

The framework is intended to demonstrate disciplined OCR evaluation rather than claim that any single metric completely characterizes document correctness.
