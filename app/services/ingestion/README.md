# Ingestion service — built Day 2 ✅

Document upload → parsing (.txt/.md/.pdf) → chunking (word-based,
configurable size/overlap) → embedding (fastembed, bge-small-en-v1.5,
384-dim) → storage in pgvector.

Deferred to v2: OCR for scanned PDFs, table extraction, image handling.
