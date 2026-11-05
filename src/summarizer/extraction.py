"""
Page extraction and text cleaning module for audit report PDFs.
"""

from __future__ import annotations

import io
import logging
import os
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

try:
    import pytesseract
    from PIL import Image
except ImportError:  # pragma: no cover
    pytesseract = None
    Image = None

PAGE_NUM_RE = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.I)


def normalize_line(line: str) -> str:
    """Normalize unicode characters, remove soft hyphens, and convert tabs to spaces.

    Args:
        line: Raw text line.

    Returns:
        Normalized line string.
    """
    line = unicodedata.normalize("NFKC", line).replace("\u00ad", "").replace("\t", "   ")
    return line.strip()


def join_hyphenated(lines: List[str]) -> List[str]:
    """Join words split by hyphen across consecutive lines.

    Args:
        lines: List of text lines.

    Returns:
        Lines with wrapped hyphens joined.
    """
    out: List[str] = []
    for line in lines:
        if out and out[-1] and out[-1].endswith("-") and line and line[0].islower():
            out[-1] = out[-1][:-1] + line
        else:
            out.append(line)
    return out


def ocr_page(page: Any, dpi: int = 300, tesseract_cmd: Optional[str] = None) -> str:
    """Perform OCR on a PDF page using Tesseract OCR.

    Args:
        page: PyMuPDF Page object.
        dpi: Resolution for rendering the page pixmap.
        tesseract_cmd: Optional path to tesseract executable.

    Returns:
        Extracted text string from OCR, or empty string on failure.
    """
    if pytesseract is None or Image is None:
        logger.warning("pytesseract or PIL is not installed; skipping OCR.")
        return ""

    cmd = tesseract_cmd or os.environ.get("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

    try:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        return text or ""
    except Exception as e:
        logger.warning("OCR failed on page %s: %s", getattr(page, "number", "?"), e)
        return ""


def extract_pages(
    pdf_path: str,
    min_chars_text_layer: int = 30,
    ocr_dpi: int = 300,
    tesseract_cmd: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract page text from a PDF, falling back to OCR when text layer is absent.

    Args:
        pdf_path: Path to PDF file.
        min_chars_text_layer: Threshold below which a page is treated as scanned.
        ocr_dpi: DPI to use for OCR pixmap rendering.
        tesseract_cmd: Optional path to tesseract binary.

    Returns:
        List of dictionaries with keys 'page' (1-indexed), 'text', and 'method'.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF is required: pip install pymupdf")

    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: List[Dict[str, Any]] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            method = "text_layer"
            if len(text.strip()) < min_chars_text_layer:
                logger.info("Page %d has no text layer (%d chars); running OCR fallback.", i, len(text.strip()))
                text = ocr_page(page, dpi=ocr_dpi, tesseract_cmd=tesseract_cmd)
                method = "ocr"
            pages.append({"page": i, "text": text, "method": method})
    return pages


def clean_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize lines, remove repeated headers/footers and page numbers.

    Args:
        pages: List of page dicts from extract_pages.

    Returns:
        Cleaned page dicts with a 'lines' list of processed strings.
    """
    raw_pages = [[normalize_line(l) for l in p["text"].splitlines()] for p in pages]

    # Count occurrences of candidate header/footer lines across distinct pages
    counts: Counter[str] = Counter()
    for raw in raw_pages:
        unique_on_page = {l for l in raw if len(l) >= 4}
        counts.update(unique_on_page)

    n_pages = len(pages)
    repeated = set()
    if n_pages >= 3:
        repeated = {
            l for l, c in counts.items()
            if c >= 3 and (c / n_pages) > 0.5 and len(l) < 120
        }

    cleaned: List[Dict[str, Any]] = []
    for p, raw in zip(pages, raw_pages):
        keep: List[str] = []
        for l in raw:
            if not l:
                keep.append("")
                continue
            if l in repeated or PAGE_NUM_RE.fullmatch(l):
                continue
            keep.append(l)
        cleaned.append({
            "page": p["page"],
            "method": p.get("method", ""),
            "lines": join_hyphenated(keep),
        })
    return cleaned
