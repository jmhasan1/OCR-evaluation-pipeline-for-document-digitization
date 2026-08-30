"""Benchmark the end-to-end OCR document pipeline on CPU/GPU/auto."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from _bootstrap import ROOT  # noqa: F401
from ocr.paddleocr_adapter import PaddleOCRAdapter
from ocr_eval.pipeline import run_document_ocr

BENCHMARK_VERSION = "1.0"


def _gpu_memory_mb() -> float | None:
    """Return Paddle's current maximum allocated GPU memory in MiB."""
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda():
            return round(paddle.device.cuda.max_memory_allocated() / 1024**2, 2)
    except Exception:
        return None
    return None


def _reset_gpu_memory_stats() -> None:
    """Reset Paddle GPU memory statistics when supported."""
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda():
            reset = getattr(paddle.device.cuda, "reset_max_memory_allocated", None)
            if reset is not None:
                reset()
    except Exception:
        pass


def _environment_info() -> dict[str, Any]:
    """Collect stable environment metadata without requiring GPU hardware."""
    try:
        import paddle
        paddle_version = paddle.__version__
        cuda_available = bool(paddle.device.is_compiled_with_cuda())
    except Exception:
        paddle_version = None
        cuda_available = False

    try:
        import paddleocr
        paddleocr_version = getattr(paddleocr, "__version__", None)
    except Exception:
        paddleocr_version = None

    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "paddle": paddle_version,
        "paddleocr": paddleocr_version,
        "paddle_cuda_compiled": cuda_available,
    }


def _latency_summary(timings: list[float], page_count: int) -> dict[str, float | None]:
    """Summarize document inference timings and page throughput."""
    if not timings:
        return {
            "mean_seconds": None,
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "mean_seconds_per_page": None,
            "mean_pages_per_second": None,
        }

    mean_seconds = statistics.mean(timings)
    seconds_per_page = mean_seconds / page_count if page_count else None
    pages_per_second = page_count / mean_seconds if page_count and mean_seconds else None

    return {
        "mean_seconds": round(mean_seconds, 4),
        "median_seconds": round(statistics.median(timings), 4),
        "min_seconds": round(min(timings), 4),
        "max_seconds": round(max(timings), 4),
        "mean_seconds_per_page": round(seconds_per_page, 4) if seconds_per_page is not None else None,
        "mean_pages_per_second": round(pages_per_second, 6) if pages_per_second is not None else None,
    }


def benchmark_document(
    input_path: str | Path,
    device: str,
    repeats: int,
    warmup: int,
    lang: str,
    dpi: int,
) -> dict[str, Any]:
    """Benchmark one document with one requested device."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    input_path = Path(input_path)

    init_start = time.perf_counter()
    engine = PaddleOCRAdapter(lang=lang, device=device)
    initialization_seconds = time.perf_counter() - init_start

    # Warmup is intentionally excluded from measured inference statistics.
    for _ in range(warmup):
        run_document_ocr(input_path, engine, dpi=dpi)

    timings: list[float] = []
    page_count = 0
    last_document = None

    _reset_gpu_memory_stats()
    for _ in range(repeats):
        start = time.perf_counter()
        last_document = run_document_ocr(input_path, engine, dpi=dpi)
        timings.append(time.perf_counter() - start)
        page_count = len(last_document.pages)

    runtime = engine.runtime_info()
    return {
        "requested_device": device,
        "resolved_device": runtime.get("device_resolved"),
        "status": "ok",
        "runtime": runtime,
        "warmup_runs": warmup,
        "measured_runs": repeats,
        "page_count": page_count,
        "initialization_seconds": round(initialization_seconds, 4),
        "inference": _latency_summary(timings, page_count),
        "gpu_max_memory_mb": _gpu_memory_mb(),
        "last_run_pages": len(last_document.pages) if last_document else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="PDF, image file, or image directory")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["auto"],
        choices=["auto", "cpu", "gpu:0"],
        help="Requested execution devices",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--output", default="outputs/benchmarks/ocr_benchmark.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    results: list[dict[str, Any]] = []
    for device in args.devices:
        try:
            results.append(
                benchmark_document(
                    input_path=input_path,
                    device=device,
                    repeats=args.repeats,
                    warmup=args.warmup,
                    lang=args.lang,
                    dpi=args.dpi,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "requested_device": device,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "input": {
            "path": str(input_path),
            "document_id": input_path.stem,
            "dpi": args.dpi,
        },
        "environment": _environment_info(),
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved benchmark to: {output}")


if __name__ == "__main__":
    main()
