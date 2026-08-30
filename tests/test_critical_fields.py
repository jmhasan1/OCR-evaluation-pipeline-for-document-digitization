from ocr_eval.critical_fields import evaluate_field, evaluate_fields


def test_exact_scalar_match():
    result = evaluate_field(
        "village",
        "Rampur",
        "Rampur",
    )

    assert result.status == "exact_match"


def test_normalized_scalar_match():
    result = evaluate_field(
        "village",
        "Rampur",
        "  Rampur  ",
    )

    assert result.status == "normalized_match"


def test_identifier_punctuation_error_is_not_hidden():
    result = evaluate_field(
        "survey_plot_khasra_number",
        "125/3",
        "1253",
    )

    assert result.status == "mismatch"


def test_missing_field():
    result = evaluate_field(
        "district",
        "Nadia",
        None,
    )

    assert result.status == "missing"


def test_exact_list_match():
    result = evaluate_field(
        "owner_names",
        ["Mohammad Hasan"],
        ["Mohammad Hasan"],
    )

    assert result.status == "exact_match"


def test_normalized_list_match():
    result = evaluate_field(
        "owner_names",
        ["Mohammad Hasan"],
        ["  Mohammad Hasan  "],
    )

    assert result.status == "normalized_match"


def test_list_mismatch():
    result = evaluate_field(
        "owner_names",
        ["Mohammad Hasan"],
        ["Mohammad Hasa"],
    )

    assert result.status == "mismatch"


def test_document_field_summary():
    result = evaluate_fields(
        {
            "village": "Rampur",
            "district": "Nadia",
            "registration_number": "REG-2025-00125",
        },
        {
            "village": "Rampur",
            "district": "Nadia",
            "registration_number": "REG-2025-0012S",
        },
    )

    assert result["total_fields"] == 3
    assert result["correct_fields"] == 2
    assert result["exact_matches"] == 2
    assert result["mismatches"] == 1
    assert result["missing"] == 0
    assert result["accuracy"] == 2 / 3
    assert len(result["failures"]) == 1
    assert result["failures"][0]["field"] == "registration_number"


def test_missing_field_is_reported_as_failure():
    result = evaluate_fields(
        {
            "village": "Rampur",
            "district": "Nadia",
        },
        {
            "village": "Rampur",
        },
    )

    assert result["missing"] == 1
    assert result["failures"][0]["field"] == "district"
    assert result["failures"][0]["status"] == "missing"