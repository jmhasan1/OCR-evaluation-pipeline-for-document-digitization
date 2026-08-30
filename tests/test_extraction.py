from ocr_eval.extraction import extract_critical_fields


def test_extract_doc_001_style_fields():
    text = """\
SALE DEED
This Sale Deed is executed on 15 March 2025 at Rampur.

PURCHASER
Mohammad Hasan, son of Abdul Majid, residing at Rampur.

PROPERTY DETAILS
Survey Number: 125/3
Area: 0.125 Acre
Village: Rampur
Tehsil: Chandpur
District: Nadia

REGISTRATION DETAILS
Registration Number: REG-2025-00125
Registration Date: 15/03/2025
Mutation Number: MUT-2025-00481
Mutation Date: 22/03/2025
"""

    result = extract_critical_fields(text)

    assert result == {
        "owner_names": ["Mohammad Hasan"],
        "father_husband_names": ["Abdul Majid"],
        "survey_plot_khasra_number": "125/3",
        "area": {"value": "0.125", "unit": "Acre"},
        "village": "Rampur",
        "tehsil": "Chandpur",
        "district": "Nadia",
        "registration_number": "REG-2025-00125",
        "registration_date": "15/03/2025",
        "mutation_number": "MUT-2025-00481",
        "mutation_date": "22/03/2025",
    }


def test_extract_doc_002_style_variants():
    text = """\
PURCHASER
Farhan Ahmed, son of Salim Ahmed.

PROPERTY SCHEDULE
Survey / Plot No.: 78/2A
Area: 1.375 Bigha
Village: Lakshmipur
Tehsil: Haripur
District: Murshidabad

REGISTRATION AND MUTATION
Registration No.: REG-2024-07821
Registration Date: 07-08-2024
Mutation No.: MUT-2024-01976
Mutation Date: 19-08-2024
"""

    result = extract_critical_fields(text)

    assert result["owner_names"] == ["Farhan Ahmed"]
    assert result["father_husband_names"] == ["Salim Ahmed"]
    assert result["survey_plot_khasra_number"] == "78/2A"
    assert result["area"] == {"value": "1.375", "unit": "Bigha"}
    assert result["village"] == "Lakshmipur"
    assert result["tehsil"] == "Haripur"
    assert result["district"] == "Murshidabad"
    assert result["registration_number"] == "REG-2024-07821"
    assert result["registration_date"] == "07-08-2024"
    assert result["mutation_number"] == "MUT-2024-01976"
    assert result["mutation_date"] == "19-08-2024"


def test_extract_doc_003_style_variants():
    text = """\
TRANSFEREE / PURCHASER
Md. Imran Hossain, son of Jalal Hossain

LAND PARTICULARS
Khasra / Survey No.: 304/7B
Area: 0.875 Hectare
Village: Sonapur
Tehsil: Beldanga
District: Birbhum

REGISTRATION PARTICULARS
Registration Number: WB-BIR-2023-30477
Registration Date: 21/11/2023
Mutation Number: MUT-BIR-23-8871
Mutation Date: 02/12/2023
"""

    result = extract_critical_fields(text)

    assert result["owner_names"] == ["Md. Imran Hossain"]
    assert result["father_husband_names"] == ["Jalal Hossain"]
    assert result["survey_plot_khasra_number"] == "304/7B"
    assert result["area"] == {"value": "0.875", "unit": "Hectare"}
    assert result["village"] == "Sonapur"
    assert result["tehsil"] == "Beldanga"
    assert result["district"] == "Birbhum"
    assert result["registration_number"] == "WB-BIR-2023-30477"
    assert result["registration_date"] == "21/11/2023"
    assert result["mutation_number"] == "MUT-BIR-23-8871"
    assert result["mutation_date"] == "02/12/2023"


def test_identifier_punctuation_is_preserved():
    text = """\
PROPERTY DETAILS
Survey Number: 125/3

REGISTRATION DETAILS
Registration Number: REG-2025-00125
Mutation Number: MUT-2025-00481
"""

    result = extract_critical_fields(text)

    assert result["survey_plot_khasra_number"] == "125/3"
    assert result["registration_number"] == "REG-2025-00125"
    assert result["mutation_number"] == "MUT-2025-00481"


def test_ocr_identifier_error_is_not_corrected():
    text = """\
PROPERTY DETAILS
Survey Number: 1253

REGISTRATION DETAILS
Registration Number: REG-2025-0012S
Mutation Number: MUT-2025-00481
"""

    result = extract_critical_fields(text)

    assert result["survey_plot_khasra_number"] == "1253"
    assert result["registration_number"] == "REG-2025-0012S"
    assert result["mutation_number"] == "MUT-2025-00481"


def test_missing_fields_are_explicit():
    text = """\
PURCHASER
Farhan Ahmed, son of Salim Ahmed.

PROPERTY SCHEDULE
Village: Lakshmipur
"""

    result = extract_critical_fields(text)

    assert result["owner_names"] == ["Farhan Ahmed"]
    assert result["father_husband_names"] == ["Salim Ahmed"]

    assert result["survey_plot_khasra_number"] is None
    assert result["area"] is None
    assert result["village"] == "Lakshmipur"
    assert result["tehsil"] is None
    assert result["district"] is None
    assert result["registration_number"] is None
    assert result["registration_date"] is None
    assert result["mutation_number"] is None
    assert result["mutation_date"] is None


def test_seller_is_not_extracted_as_owner():
    text = """\
SELLERS
1. Smt. Amina Begum, wife of Yusuf Ali.
2. Rahim Ali, son of Yusuf Ali.

PURCHASER
Farhan Ahmed, son of Salim Ahmed.
"""

    result = extract_critical_fields(text)

    assert result["owner_names"] == ["Farhan Ahmed"]
    assert result["father_husband_names"] == ["Salim Ahmed"]


def test_multiple_whitespace_is_cleaned():
    text = """\
PURCHASER
Mohammad Hasan,   son of   Abdul Majid.

PROPERTY DETAILS
Village:    Rampur
Tehsil:\tChandpur
District:  Nadia
"""

    result = extract_critical_fields(text)

    assert result["owner_names"] == ["Mohammad Hasan"]
    assert result["father_husband_names"] == ["Abdul Majid"]
    assert result["village"] == "Rampur"
    assert result["tehsil"] == "Chandpur"
    assert result["district"] == "Nadia"


def test_extractor_requires_string_input():
    try:
        extract_critical_fields(None)
    except TypeError as exc:
        assert str(exc) == "text must be a string"
    else:
        raise AssertionError("Expected TypeError")

def test_corrupted_identifier_is_extracted_without_silent_repair():
    text = """
    PROPERTY DETAILS
    Survey Number: 1253
    """

    result = extract_critical_fields(text)

    assert result["survey_plot_khasra_number"] == "1253"


def test_missing_mutation_number_is_reported_as_none():
    text = """
    REGISTRATION DETAILS
    Registration Number: REG-2025-00125
    Registration Date: 15/03/2025
    Mutation Date: 22/03/2025
    """

    result = extract_critical_fields(text)

    assert result["mutation_number"] is None


def test_field_extraction_prefers_field_label_over_nearby_khasra_annotation():
    text = """
    LAND PARTICULARS
    Khasra / Survey No.: 304/7B
    Area: 0.875 Hectare

    HANDWRITTEN TEST ANNOTATION:
    Check Khasra 304/7B before approval.
    """

    result = extract_critical_fields(text)

    assert result["survey_plot_khasra_number"] == "304/7B"


def test_purchaser_is_not_confused_by_multiple_sellers():
    text = """
    SELLERS
    1. Smt. Amina Begum, wife of Yusuf Ali.
    2. Rahim Ali, son of Yusuf Ali.

    PURCHASER
    Farhan Ahmed, son of Salim Ahmed.
    """

    result = extract_critical_fields(text)

    assert result["owner_names"] == ["Farhan Ahmed"]
    assert result["father_husband_names"] == ["Salim Ahmed"]


def test_identifier_punctuation_is_preserved():
    text = """
    Survey / Plot No.: 78/2A
    Registration No.: REG-2024-07821
    Mutation No.: MUT-2024-01976
    """

    result = extract_critical_fields(text)

    assert result["survey_plot_khasra_number"] == "78/2A"
    assert result["registration_number"] == "REG-2024-07821"
    assert result["mutation_number"] == "MUT-2024-01976"


def test_date_format_is_preserved():
    text = """
    Registration Date: 07-08-2024
    Mutation Date: 19-08-2024
    """

    result = extract_critical_fields(text)

    assert result["registration_date"] == "07-08-2024"
    assert result["mutation_date"] == "19-08-2024"

