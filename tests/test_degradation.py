import pytest

from ocr_eval.degradation import (
    apply_degradation,
    list_degradations,
)


def test_list_degradations_exposes_expected_strategies():
    names = {item["name"] for item in list_degradations()}

    assert names == {
        "character_substitution",
        "character_deletion",
        "character_insertion",
        "whitespace_corruption",
        "punctuation_corruption",
        "identifier_corruption",
    }


@pytest.mark.parametrize(
    "degradation",
    [
        "character_substitution",
        "character_deletion",
        "character_insertion",
        "whitespace_corruption",
        "punctuation_corruption",
        "identifier_corruption",
    ],
)
def test_degradation_is_reproducible(degradation):
    text = "Registration Number: REG-2025-00125"

    first = apply_degradation(text, degradation, severity=0.5)
    second = apply_degradation(text, degradation, severity=0.5)

    assert first == second


@pytest.mark.parametrize(
    "severity",
    [0.0, 1.0, 0, 1],
)
def test_severity_boundaries_are_valid(severity):
    result = apply_degradation(
        "Registration Number: REG-2025-00125",
        "character_substitution",
        severity=severity,
    )

    assert result.severity == float(severity)


@pytest.mark.parametrize(
    "severity",
    [-0.01, 1.01, 2.0],
)
def test_invalid_severity_is_rejected(severity):
    with pytest.raises(ValueError):
        apply_degradation(
            "some text",
            "character_substitution",
            severity=severity,
        )


@pytest.mark.parametrize(
    "severity",
    ["0.5", None, True, False],
)
def test_invalid_severity_type_is_rejected(severity):
    with pytest.raises(TypeError):
        apply_degradation(
            "some text",
            "character_substitution",
            severity=severity,
        )


def test_non_string_text_is_rejected():
    with pytest.raises(TypeError):
        apply_degradation(
            123,
            "character_substitution",
        )


def test_unknown_degradation_is_rejected():
    with pytest.raises(ValueError, match="Unknown degradation"):
        apply_degradation(
            "some text",
            "does_not_exist",
        )


@pytest.mark.parametrize(
    "degradation",
    [
        "character_substitution",
        "character_deletion",
        "character_insertion",
        "punctuation_corruption",
        "identifier_corruption",
    ],
)
def test_nonzero_degradation_changes_identifier_text(degradation):
    text = "Registration Number: REG-2025-00125"

    result = apply_degradation(
        text,
        degradation,
        severity=1.0,
    )

    assert result.degraded_text != text
    assert result.changed_positions


def test_zero_severity_preserves_input():
    text = "Registration Number: REG-2025-00125"

    for degradation in list(
        {
            item["name"]
            for item in list_degradations()
        }
    ):
        result = apply_degradation(
            text,
            degradation,
            severity=0.0,
        )

        assert result.degraded_text == text
        assert result.changed_positions == ()


def test_input_is_not_mutated():
    text = "REG-2025-00125"
    original = text

    apply_degradation(
        text,
        "character_substitution",
        severity=1.0,
    )

    assert text == original


def test_result_serializes_to_dict():
    result = apply_degradation(
        "REG-2025-00125",
        "identifier_corruption",
        severity=0.5,
    )

    data = result.to_dict()

    assert data["original_text"] == "REG-2025-00125"
    assert data["degradation"] == "identifier_corruption"
    assert data["severity"] == 0.5
    assert isinstance(data["changed_positions"], tuple)


def test_identifier_corruption_preserves_unrelated_text():
    text = "Registration Number: REG-2025-00125"

    result = apply_degradation(
        text,
        "identifier_corruption",
        severity=1.0,
    )

    assert result.degraded_text.startswith("Registration Number: ")
    assert result.degraded_text != text


def test_identifier_punctuation_can_be_corrupted():
    text = "Khasra 125/3"

    result = apply_degradation(
        text,
        "identifier_corruption",
        severity=1.0,
    )

    assert result.degraded_text != text


def test_devanagari_text_is_supported():
    text = "माता का नाम: अब्दुल मजीद"

    result = apply_degradation(
        text,
        "character_substitution",
        severity=0.5,
    )

    assert result.original_text == text
    assert result.degraded_text != text


def test_whitespace_corruption_changes_spacing_at_high_severity():
    text = "hello   world\nnext line"

    result = apply_degradation(
        text,
        "whitespace_corruption",
        severity=1.0,
    )

    assert result.degraded_text != text


def test_punctuation_corruption_preserves_letters():
    text = "REG-2025-00125"

    result = apply_degradation(
        text,
        "punctuation_corruption",
        severity=1.0,
    )

    assert "REG" in result.degraded_text
    assert "2025" in result.degraded_text