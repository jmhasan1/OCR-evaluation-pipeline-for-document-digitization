from ocr_eval.edit_distance import levenshtein_counts
from ocr_eval.metrics import cer, text_metrics, wer


def test_levenshtein_counts_exact_match():
    result = levenshtein_counts("abc", "abc")

    assert result.substitutions == 0
    assert result.insertions == 0
    assert result.deletions == 0
    assert result.distance == 0


def test_levenshtein_counts_substitution():
    result = levenshtein_counts("cat", "cut")

    assert result.substitutions == 1
    assert result.insertions == 0
    assert result.deletions == 0
    assert result.distance == 1


def test_levenshtein_counts_insertion():
    result = levenshtein_counts("cat", "cart")

    assert result.distance == 1
    assert result.insertions + result.deletions + result.substitutions == 1


def test_levenshtein_counts_deletion():
    result = levenshtein_counts("cart", "cat")

    assert result.distance == 1
    assert result.insertions + result.deletions + result.substitutions == 1


def test_cer_exact_match():
    assert cer("Hello world", "Hello world") == 0.0


def test_wer_exact_match():
    assert wer("Hello world", "Hello world") == 0.0


def test_cer_detects_character_error():
    value = cer("cat", "cut")

    assert value is not None
    assert value == 1 / 3


def test_wer_detects_word_error():
    value = wer("hello world", "hello there")

    assert value is not None
    assert value == 1 / 2


def test_metrics_support_devanagari():
    result = text_metrics(
        "मातादीन जाटव",
        "मातादीन जाटव",
    )

    assert result["cer"] == 0.0
    assert result["wer"] == 0.0


def test_metrics_preserve_identifier_errors():
    result = text_metrics(
        "Khasra 54/1",
        "Khasra 541",
    )

    assert result["cer"] is not None
    assert result["cer"] > 0.0


def test_empty_reference_is_defined_only_for_empty_hypothesis():
    assert cer("", "") == 0.0
    assert wer("", "") == 0.0

    assert cer("", "text") is None
    assert wer("", "text") is None


def test_text_metrics_exposes_error_breakdown():
    result = text_metrics(
        "hello world",
        "hello there",
    )

    assert result["cer"] is not None
    assert result["wer"] == 0.5

    assert "substitutions" in result["characters"]
    assert "insertions" in result["characters"]
    assert "deletions" in result["characters"]

    assert "substitutions" in result["words"]
    assert "insertions" in result["words"]
    assert "deletions" in result["words"]