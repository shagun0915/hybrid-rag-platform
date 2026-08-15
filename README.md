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

## API — Day 4 changes

`/query` now uses **hybrid retrieval** instead of pure vector search:
vector (semantic) search and keyword (lexical, via Postgres full-text
search) run independently, then get merged via Reciprocal Rank Fusion.
Response `sources` now show `fusion_score` instead of `similarity_score`.

No request/response shape changes beyond that field rename — same
endpoint, same usage:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What experience does this person have with Dynamics 365?"}'
```

Why this matters in practice: pure vector search can miss exact-term
queries (a specific product name, an ID, an acronym) because those don't
carry rich semantic meaning the way a full sentence does. Keyword search
catches exactly that case. Neither approach is strictly better — that's
the whole justification for fusing both rather than picking one.

## API — Day 5 changes

`/query` is now genuinely agentic, not a fixed pipeline. Each request:

1. Embeds the question, runs hybrid search (Day 4), then **reranks** the
   candidates with a cross-encoder for a more precise top-N.
2. Checks whether the top reranked result is confident enough
   (`MIN_RERANK_SCORE`, default 0.5).
3. If not, and attempts remain (`MAX_RETRIEVAL_ATTEMPTS`, default 2), the
   LLM **reformulates the query** and the loop retries — this is the
   agentic step: the system decides to retry based on its own previous
   result, rather than following a fixed script.
4. Stops either when confident, or when attempts are exhausted — **never
   indefinitely**. This cap is not optional; every agent loop needs one.

The `retrieval_debug.attempts` field in the response shows exactly what
the system tried on each attempt (query used, candidates found, top
score) — visible in the API response itself, not just server logs.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the Rank-1 accuracy in the fingerprint recognition research?"}'
```

Note: the per-request `top_k` override from Day 3/4 was dropped for now
— retrieval width is controlled by `RETRIEVAL_TOP_K` / `RERANK_TOP_N` in
`.env`. Could be reintroduced by threading it through `agentic_retrieve`;
left out today to keep the loop's signature simple while it's new.

## Evaluation — Day 6

Run the golden-dataset evaluation against the live system:

```bash
docker compose exec api python -m app.services.evaluation.run_eval
```

This calls `/query` for every case in `app/services/evaluation/golden_dataset.py`,
scores each result, prints a summary table, and saves a full JSON report
to `app/services/evaluation/reports/`.

**Metrics measured:**
- **Recall@K** — did the correct source document appear anywhere in the retrieved results?
- **MRR (Mean Reciprocal Rank)** — did it appear *near the top*, not just somewhere in range?
- **Keyword coverage** — does the generated answer contain the expected facts?
- **Correct abstention** — for the one deliberate unanswerable case, does the system say "I don't know" instead of hallucinating?
- **Retrieval attempts** — how often did the agentic reformulation loop actually fire?

**Honest scope note:** this is an 11-question hand-built golden set, not
the 100-300 questions a production eval suite would have — a deliberate
choice for a solo one-week build, not a hidden shortcut. Every question
is traced to content this system has already returned correctly during
manual testing on Days 3-5, so the ground truth itself is verified. The
harness (metrics, runner, report format) scales to any dataset size
without code changes — expanding the question set is natural follow-up
work.

**Known limitation, stated plainly:** keyword coverage checks for
substring presence, not semantic correctness — a technically-worded-wrong
answer that happens to contain the right number would still "pass." An
LLM-as-judge (a second model scoring "does this answer correctly address
the question") is the standard stronger approach and the honest v2
upgrade path from here.

## Known Limitations

Found by actually running the system, not predicted in advance. Listed
here plainly rather than left for someone else to discover.

### Retrieval is strong on literal terms, measurably weaker on paraphrases

The clearest example, caught by the Day 6 eval suite: asking **"SonarQube"**
directly retrieves the correct chunk with a cross-encoder rerank score of
**0.98** — confident, correct, cited. But asking the *semantically
identical* question **"What security tools were used for remediation?"**
— which never uses the word "SonarQube" — collapses retrieval confidence
to **0.0006**. The correct chunk is still technically present in the
final context (tied last among five near-zero-scored candidates), but the
LLM reads a low-signal excerpt and reports it found nothing.

Two compounding causes, both identified from the actual eval report
(`app/services/evaluation/reports/`), not guessed at:
1. **Chunking split the fact from its context.** The original sentence
   was "...owned remediation of ~80% of application security findings
   (Checkmarx, SonarQube)." Word-based chunking (Day 2) cut this near a
   boundary, so the retrieved chunk contains "SonarQube" but not
   "remediation" — the exact word the paraphrased question used.
2. **The reranker has no signal to work with when both query and chunk
   use different vocabulary for the same idea.** This is a known,
   general weakness of cross-encoder rerankers trained primarily on
   direct-match relevance, not paraphrase understanding.

**What would actually fix this** (v2, not built): semantic chunking
(splitting on topic boundaries instead of fixed word counts, so a fact
and its context can't be separated mid-sentence), and/or query expansion
before retrieval (generating a few paraphrased/synonym versions of the
query and searching with all of them, not just the literal one asked).

### Small local LLM (Ollama) shows real answer variance between identical runs

Re-running the exact same 11-question eval suite three times in a row
produced different pass/fail results on two borderline questions
(`resume_sonarqube`, `resume_career_progression`) — not because retrieval
changed, but because Ollama's `llama3.1:8b` phrased answers differently
run to run given weakly-ranked context. This is a genuine, known
tradeoff of the free/local provider path (see `LLM_PROVIDER` in
`.env.example`) — a larger model like Claude would very likely show less
run-to-run variance on the same borderline evidence, though this hasn't
been directly A/B tested here.

### Other documented tradeoffs (noted inline in code, summarized here)

- **Lexical search is Postgres full-text search, not literal BM25** — a
  related but different ranking formula. See `keyword_search.py`.
- **`to_tsvector` is computed on the fly, not stored in an indexed
  column** — simpler schema, no migration needed, but slower at large
  document-count scale than a persisted GIN-indexed column would be.
- **Keyword-coverage faithfulness checking is substring presence, not
  semantic correctness** — see the Evaluation section above.
- **The golden dataset is 11 hand-verified questions, not 100-300** — a
  deliberate scope choice for a solo one-week build.

## Roadmap

- [x] **Day 1** — Repo structure, FastAPI skeleton, Postgres+pgvector via Docker, health checks
- [x] **Day 2** — Document ingestion: parsing (.txt/.md/.pdf), chunking, embeddings (fastembed/bge-small), storage in pgvector
- [x] **Day 3** — Baseline RAG: pgvector cosine-similarity retrieval -> swappable Ollama/Claude generation -> grounded answer with sources
- [x] **Day 4** — Hybrid retrieval: vector + Postgres full-text search, fused via Reciprocal Rank Fusion
- [x] **Day 5** — Cross-encoder reranking + agentic query reformulation with a hard iteration cap
- [x] **Day 6** — Evaluation: golden dataset, Recall@K, MRR, keyword coverage, correct-abstention check
- [ ] **Day 7** — Deployment, docs, architecture polish

**v2 roadmap** (deferred past the initial week, built as follow-up commits):
multimodal document processing (OCR, tables), claim-level citation
verification, confidence-aware human review queue, full observability/tracing.
