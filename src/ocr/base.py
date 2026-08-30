"""Abstract OCR engine interface."""

from abc import ABC, abstractmethod
from ocr_eval.schema import OCRPage


class OCREngine(ABC):
    """Common interface used by the evaluation pipeline."""

    @abstractmethod
    def recognize(self, image_path: str, page_number: int) -> OCRPage:
        """Recognize one rendered document page."""
        raise NotImplementedError