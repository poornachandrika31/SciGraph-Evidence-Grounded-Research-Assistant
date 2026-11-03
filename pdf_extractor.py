"""
PDF Extractor Service
Extracts text and metadata from uploaded research paper PDFs.
Uses pypdf (pure Python) for Windows compatibility.
"""
import io
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_pdf_text(text: str) -> str:
    """Repair common PDF layout artifacts while preserving readable text."""
    text = re.sub(r"(?<=\w)[\-\u00ad]\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract raw text from a PDF file given its byte content.
    Returns cleaned text string.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
        return normalize_pdf_text("\n\n".join(pages_text))
    except ImportError:
        raise RuntimeError(
            "pypdf is not installed. Run: pip install pypdf>=4.0.0"
        )
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise RuntimeError(f"Could not extract text from PDF: {str(e)}")


def extract_metadata_heuristic(text: str, filename: str = "") -> dict:
    """
    Use heuristics to extract title and abstract from raw PDF text.
    Falls back to filename-based title if heuristics fail.
    """
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    # --- Title detection ---
    # Title is usually in the first few non-empty lines, often the longest short line
    title = ""
    for i, line in enumerate(lines[:10]):
        # Skip lines that look like author affiliations or metadata
        if len(line) > 20 and len(line) < 200 and not re.match(r'^\d', line):
            if not any(kw in line.lower() for kw in ['abstract', 'introduction', 'http', '@', 'doi:']):
                title = line
                break

    if not title and filename:
        title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()

    # --- Abstract detection ---
    abstract = ""
    abstract_match = re.search(
        r'(?i)abstract[:\s]*\n?(.*?)(?=\n(?:introduction|keywords|1\.|background|methods))',
        text,
        re.DOTALL
    )
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        abstract = re.sub(r'\s+', ' ', abstract)
        abstract = abstract[:1200]  # Cap at 1200 chars

    if not abstract and len(lines) > 5:
        # Fall back: use first meaningful paragraph
        para = []
        for line in lines[2:15]:
            if len(line) > 40:
                para.append(line)
                if len(' '.join(para)) > 300:
                    break
        abstract = ' '.join(para)[:600]

    return {"title": title, "abstract": abstract}


def chunk_text(text: str, max_chars: int = 6000) -> str:
    """
    Return the first `max_chars` characters of text for LLM context.
    Prioritizes abstract + introduction sections.
    """
    # Try to find and prioritize key sections
    priority_sections = []

    abstract_match = re.search(r'(?i)abstract.*?(?=\nintroduction|\n1\.)', text, re.DOTALL)
    if abstract_match:
        priority_sections.append(abstract_match.group(0)[:1500])

    intro_match = re.search(r'(?i)introduction.*?(?=\n(?:2\.|methods|background|related))', text, re.DOTALL)
    if intro_match:
        priority_sections.append(intro_match.group(0)[:2000])

    conclusion_match = re.search(r'(?i)conclusion.*?(?=\nreferences|\nappendix|$)', text, re.DOTALL)
    if conclusion_match:
        priority_sections.append(conclusion_match.group(0)[:1500])

    if priority_sections:
        combined = '\n\n'.join(priority_sections)
        if len(combined) >= 1000:
            return combined[:max_chars]

    # Fallback: just return beginning of full text
    return text[:max_chars]
