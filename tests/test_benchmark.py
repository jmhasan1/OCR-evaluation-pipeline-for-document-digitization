import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.benchmark_ocr import _latency_summary, benchmark_document
from ocr_eval.schema import OCRDocument, OCRPage


def test_latency_summary_calculates_page_metrics():
    summary = _latency_summary([2.0, 4.0], page_count=2)

    assert summary["mean_seconds"] == 3.0
    assert summary["median_seconds"] == 3.0
    assert summary["min_seconds"] == 2.0
    assert summary["max_seconds"] == 4.0
    assert summary["mean_seconds_per_page"] == 1.5
    assert summary["mean_pages_per_second"] == round(2 / 3, 6)


def test_latency_summary_handles_zero_pages():
    summary = _latency_summary([1.0], page_count=0)
    assert summary["mean_seconds_per_page"] is None
    assert summary["mean_pages_per_second"] is None


def test_benchmark_document_uses_document_pipeline(tmp_path):
    input_path = tmp_path / "demo.pdf"
    input_path.write_bytes(b"fixture")

    class FakeEngine:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def runtime_info(self):
            return {
                "device_requested": self.kwargs["device"],
                "device_resolved": "cpu",
                "gpu_available": False,
                "gpu_name": None,
                "paddle": "test",
                "paddleocr": "test",
            }

    fake_document = OCRDocument(
        document_id="demo",
        input_path=str(input_path),
        engine="FakeEngine",
        engine_version="test",
        config={},
        pages=[
            OCRPage(1, "a.png", "one"),
            OCRPage(2, "b.png", "two"),
        ],
    )

    with patch("scripts.benchmark_ocr.PaddleOCRAdapter", FakeEngine), patch(
        "scripts.benchmark_ocr.run_document_ocr", return_value=fake_document
    ):
        result = benchmark_document(input_path, "cpu", repeats=2, warmup=1, lang="en", dpi=200)

    assert result["status"] == "ok"
    assert result["page_count"] == 2
    assert result["measured_runs"] == 2
    assert result["warmup_runs"] == 1
    assert result["runtime"]["device_resolved"] == "cpu"