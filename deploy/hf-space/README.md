---
title: AskDocs
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: true
short_description: RAG chatbot that answers from documents with citations
---

# AskDocs

A chatbot that answers questions from a set of documents and cites the exact page and section it used.
Hybrid retrieval (embeddings + BM25), cross-encoder reranking, streaming answers and a refusal gate for
questions the documents don't cover.

Source code: https://github.com/24f2006816/askdocs
