from unittest.mock import patch
import pytest

from ocr.paddleocr_adapter import resolve_device


def test_cpu():
    assert resolve_device("cpu")[0] == "cpu"


def test_auto_gpu():
    with patch("ocr.paddleocr_adapter.detect_cuda", return_value=(True, "Test GPU")):
        assert resolve_device("auto") == ("gpu:0", True, "Test GPU")


def test_auto_cpu():
    with patch("ocr.paddleocr_adapter.detect_cuda", return_value=(False, None)):
        assert resolve_device("auto") == ("cpu", False, None)


def test_explicit_gpu_failure():
    with patch("ocr.paddleocr_adapter.detect_cuda", return_value=(False, None)):
        with pytest.raises(RuntimeError):
            resolve_device("gpu:0")


def test_invalid():
    with pytest.raises(ValueError):
        resolve_device("tpu")
