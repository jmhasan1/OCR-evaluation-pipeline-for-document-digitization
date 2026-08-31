# Production Risks and Mitigations

OCR evaluation for official documents has several important failure modes.

## 1. Confidence is not accuracy

A high OCR confidence score does not prove that the recognized text is correct.

**Mitigation:** retain confidence as an observable signal while requiring ground truth for accuracy claims.

## 2. CER/WER are not business correctness

A small text-level error can corrupt an important identifier.

**Mitigation:** evaluate critical fields separately.

## 3. Identifier errors are high consequence

Punctuation and digit changes can alter registration, survey, plot, or other structured identifiers.

**Mitigation:** conservative normalization and explicit field-level mismatch reporting.

## 4. Missing and incorrect fields are different

A field can be absent or present with the wrong value.

**Mitigation:** field evaluation distinguishes missing, mismatch, exact, and normalized matches.

## 5. Downstream rewriting can introduce unsupported information

A downstream component may alter source content even when OCR output is fixed.

**Mitigation:** separate faithfulness comparison from OCR accuracy.

## 6. Synthetic robustness is not production error distribution

Controlled corruption is useful for sensitivity testing but cannot establish real-world error rates.

**Mitigation:** report robustness as controlled evidence and label its scope explicitly.

## 7. Real documents may lack authoritative ground truth

Without a trusted transcription, numerical accuracy claims are not justified.

**Mitigation:** report observable real-data quality signals and explicitly state the limitation.

## 8. Runtime and quality are different dimensions

GPU acceleration can change throughput without changing OCR quality.

**Mitigation:** benchmark execution separately from evaluation metrics.

## 9. Hardware portability

GPU availability and runtime behavior differ across environments.

**Mitigation:** explicit `cpu`, `gpu:0`, and `auto` device handling plus device-selection tests.

## 10. Reproducibility

Evaluation results can become difficult to trust if experiments are nondeterministic or inputs are ambiguous.

**Mitigation:** versioned synthetic references, deterministic degradation, structured JSON outputs, and regression tests.
