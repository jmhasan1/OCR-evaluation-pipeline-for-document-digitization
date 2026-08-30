from ocr_eval.normalization import normalize_text


def test_normalization_collapses_whitespace():
    text = "  Hello\tworld\r\nOCR   test  "

    assert normalize_text(text) == "Hello world OCR test"


def test_normalization_handles_nbsp():
    text = "Registration\u00a0Number: 123"

    assert normalize_text(text) == "Registration Number: 123"


def test_normalization_preserves_devanagari():
    text = "मातादीन जाटव"

    assert normalize_text(text) == "मातादीन जाटव"


def test_normalization_preserves_digits_and_identifier_punctuation():
    text = "Khasra: 54/1"

    assert normalize_text(text) == "Khasra: 54/1"


def test_normalization_applies_nfkc():
    text = "ＡＢＣ １２３"

    assert normalize_text(text) == "ABC 123"


def test_normalization_requires_string():
    try:
        normalize_text(None)  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError")