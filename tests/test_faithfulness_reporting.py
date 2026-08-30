from ocr_eval.faithfulness_reporting import build_faithfulness_report


def test_exact_match_has_no_token_differences():
    result = build_faithfulness_report(
        "Registration Number: REG-2025-00125",
        "Registration Number: REG-2025-00125",
    )

    assert result.status == "exact_match"
    assert result.difference_count == 0
    assert result.differences == ()


def test_normalization_only_difference_has_no_token_differences():
    result = build_faithfulness_report(
        "Village:  Rampur\n",
        "Village: Rampur",
    )

    assert result.status == "normalized_match"
    assert result.normalized_match is True
    assert result.difference_count == 0


def test_token_replacement_is_reported():
    result = build_faithfulness_report(
        "Village: Rampur",
        "Village: Nadia",
    )

    assert result.status == "mismatch"
    assert result.difference_count == 1

    difference = result.differences[0]

    assert difference.operation == "replace"
    assert difference.reference == ("Rampur",)
    assert difference.hypothesis == ("Nadia",)
    assert difference.reference_start == 1
    assert difference.reference_end == 2
    assert difference.hypothesis_start == 1
    assert difference.hypothesis_end == 2


def test_identifier_punctuation_change_is_visible_at_token_level():
    result = build_faithfulness_report(
        "Survey Number: 125/3",
        "Survey Number: 1253",
    )

    assert result.status == "mismatch"
    assert result.difference_count == 1

    difference = result.differences[0]

    assert difference.operation == "replace"
    assert difference.reference == ("125/3",)
    assert difference.hypothesis == ("1253",)


def test_identifier_substitution_is_reported():
    result = build_faithfulness_report(
        "Registration Number: REG-2025-00125",
        "Registration Number: REG-2025-0012S",
    )

    assert result.status == "mismatch"
    assert result.difference_count == 1

    difference = result.differences[0]

    assert difference.operation == "replace"
    assert difference.reference == ("REG-2025-00125",)
    assert difference.hypothesis == ("REG-2025-0012S",)


def test_added_text_is_reported():
    result = build_faithfulness_report(
        "Village: Rampur",
        "Village: Rampur District: Nadia",
    )

    assert result.status == "mismatch"
    assert result.difference_count == 1

    difference = result.differences[0]

    assert difference.operation == "insert"
    assert difference.reference == ()
    assert difference.hypothesis == ("District:", "Nadia")


def test_missing_text_is_reported():
    result = build_faithfulness_report(
        "Village: Rampur District: Nadia",
        "Village: Rampur",
    )

    assert result.status == "mismatch"
    assert result.difference_count == 1

    difference = result.differences[0]

    assert difference.operation == "delete"
    assert difference.reference == ("District:", "Nadia")
    assert difference.hypothesis == ()


def test_multiple_token_changes_are_reported():
    result = build_faithfulness_report(
        "Village: Rampur District: Nadia",
        "Village: Sonapur District: Birbhum",
    )

    assert result.status == "mismatch"
    assert result.difference_count == 2
    assert all(
        difference.operation == "replace"
        for difference in result.differences
    )


def test_devanagari_tokens_are_supported():
    result = build_faithfulness_report(
        "माता दीन जाटव",
        "माता दिन जाटव",
    )

    assert result.status == "mismatch"
    assert result.difference_count == 1

    difference = result.differences[0]

    assert difference.operation == "replace"
    assert difference.reference == ("दीन",)
    assert difference.hypothesis == ("दिन",)


def test_result_preserves_underlying_metrics():
    result = build_faithfulness_report(
        "hello world",
        "hello there",
    )

    assert result.cer is not None
    assert result.wer == 0.5
    assert result.character_distance > 0
    assert result.word_distance == 1


def test_result_serializes_to_dict():
    result = build_faithfulness_report(
        "hello world",
        "hello there",
    )

    data = result.to_dict()

    assert data["status"] == "mismatch"
    assert len(data["differences"]) == 1
    assert data["differences"][0]["operation"] == "replace"
    assert data["differences"][0]["reference"] == ("world",)
    assert data["differences"][0]["hypothesis"] == ("there",)


def test_empty_reference_is_supported():
    result = build_faithfulness_report("", "text")

    assert result.status == "mismatch"
    assert result.cer is None
    assert result.wer is None
    assert result.difference_count == 1

    difference = result.differences[0]

    assert difference.operation == "insert"
    assert difference.reference == ()
    assert difference.hypothesis == ("text",)


def test_non_string_reference_is_rejected():
    try:
        build_faithfulness_report(None, "text")
    except TypeError as exc:
        assert str(exc) == "reference must be a string"
    else:
        raise AssertionError("Expected TypeError")


def test_non_string_hypothesis_is_rejected():
    try:
        build_faithfulness_report("text", None)
    except TypeError as exc:
        assert str(exc) == "hypothesis must be a string"
    else:
        raise AssertionError("Expected TypeError")

