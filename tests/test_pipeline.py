from ocr.base import OCREngine
from ocr_eval.pipeline import run_document_ocr
from ocr_eval.schema import OCRPage


class FakeEngine(OCREngine):
    def recognize(self, image_path: str, page_number: int) -> OCRPage:
        return OCRPage(
            page_number=page_number,
            image_path=image_path,
            text=f"page {page_number}",
        )


def test_run_document_ocr_from_images(tmp_path, monkeypatch):
    images = [tmp_path / "a.png", tmp_path / "b.png"]
    for image in images:
        image.write_bytes(b"not-an-image-fixture")

    monkeypatch.setattr(
        "ocr_eval.pipeline.collect_images",
        lambda *args, **kwargs: images,
    )

    result = run_document_ocr(tmp_path, FakeEngine(), document_id="demo")
    assert result.document_id == "demo"
    assert len(result.pages) == 2
    assert result.full_text == "page 1\npage 2"
    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2