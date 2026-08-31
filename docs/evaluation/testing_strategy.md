# Testing Strategy

The project uses tests as executable contracts for both correctness and evaluation methodology.

## Current status

The complete test suite currently passes:

```text
165 passed
1 environment warning
```

The warning is from Paddle's optional ccache diagnostic and does not represent a failing test.

## Test layers

### Unit-level behavior

Tests cover normalization, edit distance, metrics, extraction, field evaluation, faithfulness, degradation, schema behavior, and utility functions.

### Integration behavior

The benchmark and pipeline tests verify that OCR execution and document-level processing compose correctly.

### Adversarial behavior

Tests intentionally exercise cases such as:

- identifier punctuation corruption
- missing fields
- multiple sellers
- whitespace variation
- date-format variation
- downstream text differences
- nonzero degradation
- assignment-data exclusion

### Control cases

The robustness experiment includes severity `0.0` controls. These establish a known baseline:

```text
no corruption
    ↓
zero CER/WER
    ↓
perfect field evaluation
    ↓
zero faithfulness CER
```

## Determinism

Controlled degradation uses deterministic behavior so that repeated experiment runs produce comparable evidence.

This is important for regression testing and for distinguishing implementation changes from random experimental variation.

## Scalability and maintainability

The architecture separates:

- OCR execution
- reusable evaluation logic
- workflow scripts
- test contracts
- reporting

This allows evaluation components to be extended without coupling them to one OCR engine or one command-line workflow.
