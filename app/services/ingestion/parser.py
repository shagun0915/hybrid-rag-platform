"""
Text extraction. Day 2 scope: .txt/.md (trivial) and .pdf (via pypdf).

Scanned PDFs, tables, and images are explicitly OUT of scope today — the
project spec calls that out as a separate v2 feature (multimodal
ingestion, OCR), and pypdf can only extract text that's actually stored
as text in the PDF, not pixels. A scanned PDF will silently return little
or no text with plain pypdf; that's a known, documented limitation here,
not a bug to chase today.
"""

import io

from pypdf import PdfReader


class UnsupportedFileType(Exception):
    pass


class EmptyDocumentError(Exception):
    pass


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract_text(filename: str, content: bytes) -> str:
    lower = filename.lower()

    if lower.endswith(".pdf"):
        text = _extract_pdf_text(content)
    elif lower.endswith(".txt") or lower.endswith(".md"):
        text = content.decode("utf-8", errors="replace")
    else:
        ext = "." + lower.rsplit(".", 1)[-1] if "." in lower else "(no extension)"
        raise UnsupportedFileType(
            f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    text = text.strip()
    if not text:
        raise EmptyDocumentError(
            "No extractable text found. If this is a scanned/image-only PDF, "
            "OCR support isn't implemented yet (planned for v2)."
        )
    return text


def _extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n\n".join(pages)
