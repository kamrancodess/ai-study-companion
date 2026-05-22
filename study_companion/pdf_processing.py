from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

import fitz
import pytesseract
from PyPDF2 import PdfReader
from PIL import Image

from .config import OCR_DPI, OCR_LANG
from .text_processing import clean_text, chunk_pages, TextChunk


WINDOWS_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def configure_tesseract() -> None:
    configured_path = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if configured_path:
        pytesseract.pytesseract.tesseract_cmd = configured_path
    elif WINDOWS_TESSERACT_PATH.exists():
        pytesseract.pytesseract.tesseract_cmd = str(WINDOWS_TESSERACT_PATH)


def extract_pdf_pages_with_pypdf(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        page_text = clean_text(page_text)
        if page_text:
            pages.append((index, page_text))
    return pages


def extract_pdf_pages_with_pymupdf(pdf_path: Path) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            page_text = clean_text(page.get_text("text") or "")
            if page_text:
                pages.append((index, page_text))
    return pages


def ocr_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    configure_tesseract()
    pages: list[tuple[int, str]] = []
    scale = OCR_DPI / 72
    matrix = fitz.Matrix(scale, scale)

    try:
        with fitz.open(pdf_path) as document:
            for index, page in enumerate(document, start=1):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                page_text = pytesseract.image_to_string(image, lang=OCR_LANG)
                page_text = clean_text(page_text)
                if page_text:
                    pages.append((index, page_text))
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "This PDF appears to be scanned. Tesseract OCR is required for scanned PDFs, but it was not found."
        ) from exc
    except pytesseract.TesseractError as exc:
        raise RuntimeError(f"OCR failed while reading this scanned PDF: {exc}") from exc
    return pages


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    pages = extract_pdf_pages_with_pypdf(pdf_path)
    if pages:
        return pages

    pages = extract_pdf_pages_with_pymupdf(pdf_path)
    if pages:
        return pages

    return ocr_pdf_pages(pdf_path)


def extract_pdf_text(pdf_path: Path) -> str:
    pages = extract_pdf_pages(pdf_path)
    return "\n\n".join(text for _, text in pages)


def pdf_to_chunks(
    pdf_path: Path,
    chunk_size_words: int,
    overlap_words: int,
) -> tuple[str, list[TextChunk]]:
    pages = extract_pdf_pages(pdf_path)
    full_text = "\n\n".join(text for _, text in pages)
    chunks = chunk_pages(pages, chunk_size_words, overlap_words)
    return full_text, chunks
