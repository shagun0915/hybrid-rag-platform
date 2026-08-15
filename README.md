# Enterprise Multimodal RAG Platform

Hybrid retrieval, cross-encoder reranking, agentic query reformulation,
citation verification, and confidence-aware human review — built
incrementally, with evaluation at every stage instead of assumed quality.

**Status:** Day 1 of 7 — foundation only. No AI functionality yet, on purpose.

---

## Architecture (target — built incrementally)

```
User
  |
Frontend
  |
API (FastAPI)
  |
Query Orchestrator / Retrieval Agent
  |
Query Analysis
  |
Hybrid Retrieval ── Dense Vector Search + BM25 / Lexical Search
  |
Result Fusion
  |
Cross-Encoder Reranking
  |
Context Selection
  |
LLM Generation
  |
Citation Extraction + Verification
  |
Confidence / Quality Evaluation ── high conf -> answer / low conf -> retry or human review
  |
Final Answer with Citations
```

Ingestion pipeline (separate service):

```
Upload -> Classification -> Text/OCR/Table Extraction -> Chunking + Metadata
  -> Embedding Generation -> Vector Index (pgvector) + BM25 Index -> Ready
```

## Why Postgres + pgvector (not a separate vector DB)

Document metadata (permissions, source, timestamps) and embeddings live in
the *same* transactional database. No syncing two systems, no eventual-
consistency bugs between "what the vector store thinks exists" and "what
actually exists." Standard SQL for everything except the nearest-neighbor
search, which pgvector adds as a native column type + index.

## Repo structure

```
app/
  main.py              FastAPI entrypoint
  core/
    config.py          Typed settings (env-var driven, nothing hard-coded)
    database.py        Async engine, session management, pgvector init
  api/
    health.py           Liveness + readiness endpoints
  models/               SQLAlchemy models (Day 2+)
  services/
    ingestion/           Document parsing, chunking, embedding (Day 2)
    retrieval/            Vector, hybrid, reranking, agentic retrieval (Days 3-5)
    generation/            LLM generation + citation verification (Day 3+)
    evaluation/             Golden dataset + metrics (Day 6)
tests/                  Test suite, grows alongside each service
```

## Setup (local)

Requires Docker + Docker Compose.

```bash
cp .env.example .env
docker compose up --build
```

Then check:
- http://localhost:8000/                — app info
- http://localhost:8000/docs             — interactive API docs (FastAPI auto-generates this)
- http://localhost:8000/health/live      — liveness (no DB dependency)
- http://localhost:8000/health/ready     — readiness (checks DB connectivity)

## Running tests

```bash
pip install -r requirements.txt
pytest
```

## API — Day 2 additions

```bash
# Upload a document (.txt, .md, or .pdf — max 20MB)
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/your/document.pdf"

# List all uploaded documents and their status
curl http://localhost:8000/documents

# Inspect how a document got chunked (debugging tool)
curl http://localhost:8000/documents/{document_id}/chunks
```

First upload will be slower than subsequent ones — the embedding model
(~130MB) downloads once on first use and is cached after that.

## API — Day 3 additions

```bash
# Ask a question — retrieves the most relevant chunks via pgvector
# cosine similarity, then asks the configured LLM to answer using
# only that context
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What experience does this person have with Dynamics 365?"}'
```

Response includes both the generated answer and the source chunks it was
grounded in (filename, similarity score, content preview) — so you can
verify *why* the model said what it said, not just trust it blindly.

**LLM provider is swappable** via `LLM_PROVIDER` in `.env`:
- `ollama` (default) — free, runs locally via [Ollama](https://ollama.com), no API key or billing needed. Requires the Ollama app running on your machine with a model pulled (`ollama pull llama3.1:8b`).
- `anthropic` — real Claude, requires `ANTHROPIC_API_KEY` and an Anthropic account with billing set up.

Both providers are grounded with the same system prompt and go through
the identical `/query` endpoint — switching is a one-line `.env` change,
no code changes.

## Roadmap

- [x] **Day 1** — Repo structure, FastAPI skeleton, Postgres+pgvector via Docker, health checks
- [x] **Day 2** — Document ingestion: parsing (.txt/.md/.pdf), chunking, embeddings (fastembed/bge-small), storage in pgvector
- [x] **Day 3** — Baseline RAG: pgvector cosine-similarity retrieval -> Claude generation -> grounded answer with sources
- [ ] **Day 4** — Hybrid retrieval: BM25 + vector fusion
- [ ] **Day 5** — Cross-encoder reranking + agentic query reformulation
- [ ] **Day 6** — Evaluation: golden dataset, Recall@K, MRR, faithfulness
- [ ] **Day 7** — Deployment, docs, architecture polish

**v2 roadmap** (deferred past the initial week, built as follow-up commits):
multimodal document processing (OCR, tables), claim-level citation
verification, confidence-aware human review queue, full observability/tracing.
