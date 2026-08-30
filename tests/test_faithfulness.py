import pytest

from ocr_eval.faithfulness import compare_faithfulness


def test_exact_match():
    result = compare_faithfulness(
        "Registration Number: REG-2025-00125",
        "Registration Number: REG-2025-00125",
    )

    assert result.status == "exact_match"
    assert result.exact_match is True
    assert result.normalized_match is True
    assert result.cer == 0.0
    assert result.wer == 0.0
    assert result.character_distance == 0
    assert result.word_distance == 0


def test_normalization_only_difference():
    result = compare_faithfulness(
        "Village:  Rampur\n",
        "Village: Rampur",
    )

    assert result.status == "normalized_match"
    assert result.exact_match is False
    assert result.normalized_match is True
    assert result.cer == 0.0
    assert result.wer == 0.0


def test_identifier_change_is_mismatch():
    result = compare_faithfulness(
        "Survey Number: 125/3",
        "Survey Number: 1253",
    )

    assert result.status == "mismatch"
    assert result.normalized_match is False
    assert result.cer is not None
    assert result.cer > 0.0
    assert result.character_distance > 0


def test_registration_number_substitution_is_mismatch():
    result = compare_faithfulness(
        "Registration Number: REG-2025-00125",
        "Registration Number: REG-2025-0012S",
    )

    assert result.status == "mismatch"
    assert result.character_distance > 0


def test_added_text_is_mismatch():
    result = compare_faithfulness(
        "Village: Rampur",
        "Village: Rampur District: Nadia",
    )

    assert result.status == "mismatch"
    assert result.word_distance > 0


def test_missing_text_is_mismatch():
    result = compare_faithfulness(
        "Village: Rampur District: Nadia",
        "Village: Rampur",
    )

    assert result.status == "mismatch"
    assert result.word_distance > 0


def test_devanagari_text_is_supported():
    result = compare_faithfulness(
        "माता दीन जाटव",
        "माता दीन जाटव",
    )

    assert result.status == "exact_match"
    assert result.cer == 0.0
    assert result.wer == 0.0


def test_empty_reference_and_empty_hypothesis():
    result = compare_faithfulness("", "")

    assert result.status == "exact_match"
    assert result.cer == 0.0
    assert result.wer == 0.0


def test_empty_reference_with_text_is_mismatch():
    result = compare_faithfulness("", "text")

    assert result.status == "mismatch"
    assert result.cer is None
    assert result.wer is None


@pytest.mark.parametrize(
    "reference,hypothesis",
    [
        ("Khasra 125/3", "Khasra 1253"),
        ("MUT-2025-00481", "MUT-2025-0048I"),
        ("15/03/2025", "15/03/202S"),
    ],
)
def test_critical_identifier_changes_are_not_normalized_away(
    reference,
    hypothesis,
):
    result = compare_faithfulness(reference, hypothesis)

    assert result.status == "mismatch"
    assert result.normalized_match is False


def test_result_serializes_to_dict():
    result = compare_faithfulness(
        "hello",
        "hello",
    )

    data = result.to_dict()

    assert data["status"] == "exact_match"
    assert data["exact_match"] is True
    assert data["normalized_match"] is True
    assert data["risk_flags"] == ()

def test_exact_match_has_no_differences():
    result = compare_faithfulness(
        "Registration Number: REG-2025-00125",
        "Registration Number: REG-2025-00125",
    )

    assert result.differences == ()


def test_normalization_only_difference_has_no_differences():
    result = compare_faithfulness(
        "Village:  Rampur\n",
        "Village: Rampur",
    )

    assert result.status == "normalized_match"
    assert result.differences == ()


def test_character_substitution_is_reported():
    result = compare_faithfulness(
        "cat",
        "cut",
    )

    assert len(result.differences) == 1

    difference = result.differences[0]

    assert difference.operation == "replace"
    assert difference.reference == "a"
    assert difference.hypothesis == "u"
    assert difference.reference_start == 1
    assert difference.reference_end == 2
    assert difference.hypothesis_start == 1
    assert difference.hypothesis_end == 2


def test_character_insertion_is_reported():
    result = compare_faithfulness(
        "cat",
        "cart",
    )

    assert len(result.differences) == 1

    difference = result.differences[0]

    assert difference.operation == "insert"
    assert difference.reference == ""
    assert difference.hypothesis == "r"


def test_character_deletion_is_reported():
    result = compare_faithfulness(
        "cart",
        "cat",
    )

    assert len(result.differences) == 1

    difference = result.differences[0]

    assert difference.operation == "delete"
    assert difference.reference == "r"
    assert difference.hypothesis == ""


def test_identifier_punctuation_deletion_is_visible():
    result = compare_faithfulness(
        "Survey Number: 125/3",
        "Survey Number: 1253",
    )

    assert result.status == "mismatch"

    differences = result.differences

    assert len(differences) == 1
    assert differences[0].operation == "delete"
    assert differences[0].reference == "/"
    assert differences[0].hypothesis == ""


def test_identifier_substitution_is_visible():
    result = compare_faithfulness(
        "REG-2025-00125",
        "REG-2025-0012S",
    )

    assert len(result.differences) == 1

    difference = result.differences[0]

    assert difference.operation == "replace"
    assert difference.reference == "5"
    assert difference.hypothesis == "S"


def test_multiple_changes_are_reported():
    result = compare_faithfulness(
        "Village: Rampur District: Nadia",
        "Village: Rampur District: Murshidabad",
    )

    assert result.status == "mismatch"
    assert len(result.differences) >= 1

    assert any(
        difference.operation == "replace"
        for difference in result.differences
    )


def test_added_text_is_reported():
    result = compare_faithfulness(
        "Village: Rampur",
        "Village: Rampur District: Nadia",
    )

    assert any(
        difference.operation == "insert"
        for difference in result.differences
    )


def test_missing_text_is_reported():
    result = compare_faithfulness(
        "Village: Rampur District: Nadia",
        "Village: Rampur",
    )

    assert any(
        difference.operation == "delete"
        for difference in result.differences
    )


def test_devanagari_difference_is_reported():
    result = compare_faithfulness(
        "माता दीन",
        "माता दिन",
    )

    assert result.status == "mismatch"
    assert len(result.differences) >= 1


def test_difference_serializes_to_dict():
    result = compare_faithfulness(
        "cat",
        "cut",
    )

    data = result.to_dict()

    assert len(data["differences"]) == 1
    assert data["differences"][0]["operation"] == "replace"
    assert data["differences"][0]["reference"] == "a"
    assert data["differences"][0]["hypothesis"] == "u"

