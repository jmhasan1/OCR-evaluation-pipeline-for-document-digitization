from __future__ import annotations

from pathlib import Path
import pymupdf


def render_pdf(pdf_path: str | Path, output_dir: str | Path, dpi: int = 200) -> list[Path]:
    """Render each PDF page to a PNG and return paths in page order."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)

    paths: list[Path] = []
    with pymupdf.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            output_path = output_dir / f"page_{index:03d}.png"
            pixmap.save(output_path)
            paths.append(output_path)
    return paths


def collect_images(input_path: str | Path, rendered_dir: str | Path | None = None, dpi: int = 200) -> list[Path]:
    """Accept either a PDF or an image directory/file."""
    input_path = Path(input_path)
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        if rendered_dir is None:
            rendered_dir = input_path.parent / f"{input_path.stem}_rendered"
        return render_pdf(input_path, rendered_dir, dpi=dpi)

    if input_path.is_file() and input_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        return [input_path]

    if input_path.is_dir():
        images = [p for p in input_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}]
        return sorted(images)

    raise FileNotFoundError(f"Unsupported or missing input: {input_path}")