from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocr_eval.pdf_utils import collect_images


def test_pdf_renders_to_expected_pages(tmp_path):
    source = ROOT / "data" / "development" / "synthetic_documents" / "doc_001" / "doc_001.pdf"
    images = collect_images(source, tmp_path / "rendered", dpi=100)
    assert len(images) == 2
    assert all(p.exists() and p.suffix == ".png" for p in images)


