"""PaddleOCR adapter with CPU/GPU portability."""

from __future__ import annotations

import ast
import json
from typing import Any

from ocr.base import OCREngine
from ocr_eval.schema import OCRPage, OCRRegion


def detect_cuda() -> tuple[bool, str | None]:
    """Return whether Paddle reports CUDA support and the first GPU name."""
    try:
        import paddle

        if not paddle.device.is_compiled_with_cuda():
            return False, None

        name = None
        try:
            name = paddle.device.cuda.get_device_name(0)
        except Exception:
            pass
        return True, name
    except Exception:
        return False, None


def resolve_device(requested: str = "auto") -> tuple[str, bool, str | None]:
    requested = requested.lower().strip()

    if requested == "cpu":
        return "cpu", False, None

    if requested == "auto":
        available, name = detect_cuda()
        return ("gpu:0" if available else "cpu"), available, name

    if requested in {"gpu", "gpu:0", "cuda", "cuda:0"}:
        available, name = detect_cuda()
        if not available:
            raise RuntimeError(
                "GPU was requested, but the installed PaddlePaddle runtime "
                "does not report CUDA support."
            )
        return "gpu:0", True, name

    raise ValueError("device must be one of: auto, cpu, gpu:0")


def package_versions() -> dict[str, str | None]:
    try:
        import paddle
        paddle_version = paddle.__version__
    except Exception:
        paddle_version = None

    try:
        import paddleocr
        paddleocr_version = getattr(paddleocr, "__version__", None)
    except Exception:
        paddleocr_version = None

    return {
        "paddle": paddle_version,
        "paddleocr": paddleocr_version,
    }


class PaddleOCRAdapter(OCREngine):
    """Concrete OCR engine using PaddleOCR."""

    def __init__(self, lang: str = "en", device: str = "auto", **kwargs: Any):
        self.lang = lang
        self.requested_device = device
        self.device, self.gpu_available, self.gpu_name = resolve_device(device)

        from paddleocr import PaddleOCR

        ocr_kwargs = dict(kwargs)

        # PaddlePaddle 3.3.0 has a CPU oneDNN/PIR compatibility issue.
        # Disable MKLDNN for CPU execution while keeping GPU configuration unchanged.
        if self.device == "cpu":
            ocr_kwargs.setdefault("enable_mkldnn", False)

        self.ocr = PaddleOCR(
            lang=lang,
            device=self.device,
            **ocr_kwargs,
        )

    def runtime_info(self) -> dict[str, Any]:
        return {
            "device_requested": self.requested_device,
            "device_resolved": self.device,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            **package_versions(),
        }

    @staticmethod
    def _extract_regions(result: Any) -> list[OCRRegion]:
        """Normalize common PaddleOCR 3.x text/score/box fields."""
        if not isinstance(result, dict):
            return []

        texts = result.get("rec_texts") or result.get("texts") or []
        scores = result.get("rec_scores") or result.get("scores") or []
        boxes = result.get("rec_boxes") or result.get("boxes") or []

        regions: list[OCRRegion] = []

        for i, text in enumerate(texts):
            score = scores[i] if i < len(scores) else None
            box = boxes[i] if i < len(boxes) else []

            regions.append(
                OCRRegion(
                    text=str(text),
                    confidence=float(score) if score is not None else None,
                    bbox=box.tolist() if hasattr(box, "tolist") else list(box),
                )
            )

        return regions

    @staticmethod
    def _extract_text(result: Any, regions: list[OCRRegion]) -> str:
        if regions:
            return "\n".join(region.text for region in regions)

        if isinstance(result, dict):
            for key in ("text", "full_text", "rec_texts"):
                value = result.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    return "\n".join(str(v) for v in value)

        return str(result) if result is not None else ""


    def recognize(self, image_path: str, page_number: int) -> OCRPage:
        """Run OCR on one image and normalize the PaddleOCR result."""
        result = self.ocr.predict(image_path)

        try:
            first = next(iter(result), {})
        except TypeError:
            first = result

        if hasattr(first, "json"):
            json_value = first.json
            first = json_value() if callable(json_value) else json_value

        if isinstance(first, str):
            try:
                first = json.loads(first)
            except json.JSONDecodeError:
                try:
                    first = ast.literal_eval(first)
                except (ValueError, SyntaxError):
                    pass

        if isinstance(first, dict) and isinstance(first.get("res"), dict):
            first = first["res"]

        regions = self._extract_regions(first)
        text = self._extract_text(first, regions)

        return OCRPage(
            page_number=page_number,
            image_path=str(image_path),
            text=text,
            regions=regions,
        )


