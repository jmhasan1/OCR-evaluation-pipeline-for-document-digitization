from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_eval.schema import OCRDocument, OCRPage, OCRRegion


def test_schema_serializes_to_json():
    doc = OCRDocument(
        document_id="test",
        input_path="x.png",
        engine="test-engine",
        engine_version="1.0",
        config={"foo": "bar"},
        pages=[OCRPage(1, "x.png", "hello", [OCRRegion("hello", 0.99, [1, 2, 3, 4])])],
    )
    payload = json.dumps(doc.to_dict())
    assert '"document_id": "test"' in payload
    assert '"confidence": 0.99' in payload