from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ocr_eval.pdf_utils import collect_images

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--dpi", type=int, default=200)
args = parser.parse_args()

images = collect_images(args.input, args.output, dpi=args.dpi)
for p in images:
    print(p)