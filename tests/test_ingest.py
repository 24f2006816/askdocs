from pathlib import Path

import pytest

from app.ingest import chunk_documents, load_markdown, load_pdf, split_text, split_units

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    chunks = split_text(text, chunk_size=200, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 200 for c in chunks)
    # The last sentence of each chunk is repeated at the start of the next one.
    for current, following in zip(chunks, chunks[1:], strict=False):
        assert current.splitlines()[-1] in following


def test_long_sentence_is_hard_split():
    chunks = split_text("word " * 400, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)


def test_abbreviations_do_not_end_sentences():
    units = split_units("The fee is Rs. 1,000 for late payment. A second sentence follows.")
    assert units == ["The fee is Rs. 1,000 for late payment.", "A second sentence follows."]


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=100, overlap=100)


def test_markdown_keeps_sections():
    sections = {doc.section for doc in load_markdown(DOCS / "student-handbook.md")}
    assert {"Mock tests", "Doubt-clearing sessions", "Batches and timings"} <= sections


def test_pdf_keeps_page_numbers():
    pages = load_pdf(DOCS / "fee-and-refund-policy.pdf")
    assert [p.page for p in pages] == [1, 2, 3, 4]
    assert "50% refund" in pages[2].text


def test_docx_keeps_headings_lists_and_tables(tmp_path):
    from docx import Document as new_docx

    from app.ingest import load_docx

    word = new_docx()
    word.add_heading("Hostel rules", level=1)
    word.add_paragraph("Gates close at 10 PM on weekdays.")
    word.add_paragraph("Visitors must sign in at reception.", style="List Bullet")
    table = word.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Room", "Fee"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "Single", "Rs. 9,000"
    path = tmp_path / "hostel.docx"
    word.save(path)

    docs = load_docx(path)
    assert docs[0].section == "Hostel rules"
    assert "Gates close at 10 PM" in docs[0].text and "- Visitors must sign in" in docs[0].text
    assert docs[1].section == "Table 1" and "Single | Rs. 9,000" in docs[1].text


def test_chunk_ids_are_unique_and_located():
    chunks = chunk_documents(load_pdf(DOCS / "fee-and-refund-policy.pdf"))
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[0].location == "fee-and-refund-policy.pdf · p. 1"
