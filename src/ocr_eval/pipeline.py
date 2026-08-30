"""Document-level OCR orchestration."""

from __future__ import annotations

from pathlib import Path

from ocr.base import OCREngine
from ocr_eval.pdf_utils import collect_images
from ocr_eval.schema import OCRDocument


def run_document_ocr(
    input_path: str | Path,
    engine: OCREngine,
    document_id: str | None = None,
    dpi: int = 200,
    rendered_dir: str | Path | None = None,
) -> OCRDocument:
    """Render PDF/images as needed, OCR each page, and normalize the result."""
    input_path = Path(input_path)
    images = collect_images(
        input_path,
        rendered_dir=rendered_dir,
        dpi=dpi,
    )

    pages = [
        engine.recognize(str(image), page_number=i)
        for i, image in enumerate(images, start=1)
    ]

    runtime = getattr(engine, "runtime_info", lambda: {})()

    return OCRDocument(
        document_id=document_id or input_path.stem,
        input_path=str(input_path),
        engine=engine.__class__.__name__,
        engine_version=runtime.get("paddleocr"),
        config={
            "dpi": dpi,
            "runtime": runtime,
        },
        pages=pages,
    )
