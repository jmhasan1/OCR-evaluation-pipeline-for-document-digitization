"""Inspect the local OCR runtime."""

import json
import platform
import sys

from _bootstrap import ROOT  # noqa: F401
from ocr.paddleocr_adapter import detect_cuda, resolve_device, package_versions


def main():
    cuda_available, gpu_name = detect_cuda()
    resolved, _, _ = resolve_device("auto")

    data = {
        "platform": platform.platform(),
        "python": sys.version,
        "paddle_cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "auto_device": resolved,
        "versions": package_versions(),
    }

    try:
        import paddle
        data["paddle_cuda_device_count"] = (
            paddle.device.cuda.device_count() if cuda_available else 0
        )
    except Exception as exc:
        data["paddle_cuda_device_count_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
