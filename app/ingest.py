"""Load documents (PDF, Markdown, plain text) and split them into overlapping chunks.

Every chunk keeps its provenance (file, page, section) so answers can cite exactly
where a fact came from.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM_RE = re.compile(r"^([-*•]|\d+[.)])\s+")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9₹$])")


@dataclass
class Document:
    text: str
    source: str
    page: int | None = None
    section: str | None = None


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None = None
    section: str | None = None

    @property
    def index_text(self) -> str:
        """Text used for retrieval: the section title adds useful context."""
        return f"{self.section}\n{self.text}" if self.section else self.text

    @property
    def location(self) -> str:
        parts = [self.source]
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.section:
            parts.append(self.section)
        return " · ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


# --------------------------------------------------------------------------- loading


def load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(text=text, source=path.name, page=number))
    return docs


def load_markdown(path: Path) -> list[Document]:
    """One Document per heading section, so chunks know which section they belong to."""
    docs: list[Document] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            docs.append(Document(text=text, source=path.name, section=section))
        buffer.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            section = match.group(2).strip()
        else:
            buffer.append(line)
    flush()
    return docs


def load_docx(path: Path) -> list[Document]:
    """Word documents: headings start sections, list paragraphs stay bullets, tables become rows."""
    from docx import Document as open_docx

    word = open_docx(str(path))
    docs: list[Document] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = "\n\n".join(buffer).strip()
        if text:
            docs.append(Document(text=text, source=path.name, section=section))
        buffer.clear()

    for paragraph in word.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = (paragraph.style.name if paragraph.style is not None else "").lower()
        if style.startswith("heading") or style == "title":
            flush()
            section = text
        elif style.startswith("list"):
            buffer.append(f"- {text}")
        else:
            buffer.append(text)
    flush()

    for number, table in enumerate(word.tables, start=1):
        rows = [" | ".join(cell.text.strip() for cell in row.cells) for row in table.rows]
        rows = [r for r in rows if r.strip(" |")]
        if rows:
            docs.append(Document(text="\n".join(f"- {r}" for r in rows), source=path.name, section=f"Table {number}"))
    return docs


def load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8").strip()
    return [Document(text=text, source=path.name)] if text else []


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".docx":
        return load_docx(path)
    if suffix in {".md", ".markdown"}:
        return load_markdown(path)
    if suffix == ".txt":
        return load_text(path)
    raise ValueError(f"Unsupported file type: {path.name}")


def load_directory(directory: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_file(path))
    return docs


# --------------------------------------------------------------------------- chunking


def split_units(text: str) -> list[str]:
    """Break text into small atomic units: list items and sentences.

    Hard line wraps inside a paragraph (common in PDFs) are joined back together,
    while list items stay separate.
    """
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
        merged: list[str] = []
        for line in lines:
            if merged and not _LIST_ITEM_RE.match(line) and not _looks_like_heading(merged[-1], line):
                merged[-1] = f"{merged[-1]} {line}"
            else:
                merged.append(line)
        for block in merged:
            units.extend(_sentences(block))
    return units


_ABBREVIATIONS = ("Rs.", "No.", "Dr.", "Mr.", "Mrs.", "Ms.", "St.", "vs.", "e.g.", "i.e.", "etc.", "Inc.", "Ltd.")


def _looks_like_heading(previous: str, line: str) -> bool:
    """A short line without end punctuation followed by a capitalised line is a title, not a wrap."""
    return len(previous) < 60 and previous[-1] not in ".,;:!?)" and line[:1].isupper()


def _sentences(block: str) -> list[str]:
    """Split on sentence ends, without breaking after abbreviations such as 'Rs.' or 'e.g.'."""
    sentences: list[str] = []
    for piece in (s.strip() for s in _SENTENCE_END_RE.split(block)):
        if not piece:
            continue
        if sentences and sentences[-1].endswith(_ABBREVIATIONS):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def _hard_split(unit: str, size: int) -> list[str]:
    """Split an over-long unit on word boundaries."""
    pieces, current = [], ""
    for word in unit.split():
        candidate = f"{current} {word}".strip()
        if len(candidate) > size and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def split_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    """Greedily pack sentences into chunks of at most ``chunk_size`` characters.

    The last sentences of each chunk (up to ``overlap`` characters) are repeated at the
    start of the next one, so a fact that straddles a boundary is still retrievable.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    pieces: list[str] = []
    for unit in split_units(text):
        pieces.extend(_hard_split(unit, chunk_size) if len(unit) > chunk_size else [unit])

    chunks: list[str] = []
    current: list[str] = []

    def size_of(items: list[str]) -> int:
        return sum(len(p) for p in items) + max(len(items) - 1, 0)

    for piece in pieces:
        if current and size_of(current + [piece]) > chunk_size:
            chunks.append("\n".join(current))
            tail: list[str] = []
            for previous in reversed(current):
                if size_of([previous] + tail) > overlap:
                    break
                tail.insert(0, previous)
            current = tail if size_of(tail + [piece]) <= chunk_size else []
        current.append(piece)
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_documents(docs: list[Document], chunk_size: int = 700, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for doc in docs:
        for text in split_text(doc.text, chunk_size, overlap):
            n = counters.get(doc.source, 0)
            counters[doc.source] = n + 1
            chunks.append(
                Chunk(id=f"{doc.source}#{n}", text=text, source=doc.source, page=doc.page, section=doc.section)
            )
    return chunks
