"""Stable normalized schema for OCR output."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class OCRRegion:
    text: str
    confidence: float | None = None
    bbox: list[float] = field(default_factory=list)


@dataclass
class OCRPage:
    page_number: int
    image_path: str
    text: str
    regions: list[OCRRegion] = field(default_factory=list)


@dataclass
class OCRDocument:
    document_id: str
    input_path: str
    engine: str
    engine_version: str | None
    config: dict[str, Any]
    pages: list[OCRPage] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n".join(page.text for page in self.pages if page.text)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["full_text"] = self.full_text
        return data
