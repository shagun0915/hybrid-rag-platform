# Hybrid RAG Platform

Hybrid retrieval (dense vector + lexical), cross-encoder reranking, and
agentic query reformulation with a hard iteration cap — built
incrementally over one week, evaluated against a real golden dataset
instead of assumed to work.

**Status:** Complete, evaluated, and deployed. Live demo:
**[hybrid-rag-platform.onrender.com/ui](https://hybrid-rag-platform.onrender.com/ui)**
(free tier — may take 30-50s to wake up if idle, and runs a leaner
retrieval config than local dev; see [ENGINEERING.md](ENGINEERING.md#deployment)
for exactly why).

> This README is the overview. The full engineering detail — incident
> postmortems, the evaluation methodology (including where it went
> wrong before it went right), every known-limitation case study with
> real numbers, the full deployment story, and the build history — lives
> in **[ENGINEERING.md](ENGINEERING.md)**.

---

## What this is

A question-answering system over your own documents: upload PDFs/text
files, ask questions in plain English, get answers grounded only in what
was actually uploaded — with the system explicitly saying "I don't know"
rather than guessing when the answer isn't in the corpus.

Not a LangChain tutorial wrapper. The pieces that make this a genuine
engineering project rather than a demo:

- **Hybrid retrieval** — dense vector search (pgvector cosine similarity)
  and lexical search (Postgres full-text) run independently and get
  fused via Reciprocal Rank Fusion, because neither approach alone
  reliably wins.
- **Cross-encoder reranking** — a second, more precise scoring pass over
  the top candidates before they reach the LLM.
- **Query expansion** — every attempt searches with the literal question
  plus a couple of LLM-generated paraphrased variants, fused together,
  so a chunk that loses under one phrasing gets another chance under
  different wording.
- **Agentic query reformulation** — if the top result isn't confident
  enough, the LLM rewrites the query and retries, capped at a hard
  maximum so a genuinely unanswerable question can't loop forever.
- **Real evaluation** — an 11-question golden dataset, run against the
  live system, measuring Recall@K, MRR, keyword coverage, LLM-as-judge
  correctness, and correct abstention — not eyeballed spot checks.
- **Swappable LLM provider** — Ollama (free, local), Groq (free, cloud),
  or Claude (paid, cloud) behind one interface, a one-line `.env` change
  to switch.
- **A demo UI** at `/ui` — watch the actual retrieval trace (hybrid
  search candidates, query variants, rerank scores, reformulation) as it
  happens, not just read about it.

## Architecture

```
User
  |
API (FastAPI)
  |
Agentic Retrieval Loop ─────────────────────────────┐
  |                                                  |
  embed query                                        |
  |                                                   |
  Hybrid Retrieval ── Dense Vector (pgvector) + Lexical (Postgres full-text)
  |                                                   |
  Reciprocal Rank Fusion                              |
  |                                                   |
  Cross-Encoder Reranking                             |
  |                                                   |
  confident? ── no, attempts remain ── reformulate ───┘
  |
  yes, or attempts exhausted
  |
LLM Generation (grounded in reranked chunks only)
  |
Answer + Sources + Retrieval Debug Trace
```

Ingestion pipeline (separate path):

```
Upload -> Parse (.txt/.md/.pdf) -> Chunk (semantic, sentence-similarity based)
  -> Embed (fastembed, bge-small, 384-dim) -> Store (Postgres + pgvector)
```

**Why Postgres + pgvector, not a separate vector database?** Document
metadata and embeddings live in the *same* transactional database — no
syncing two systems, no eventual-consistency gap between what the vector
store thinks exists and what actually does. It's also what let hybrid
search be nearly free to add — the lexical half is just Postgres
full-text search on the same table, no second database to stand up.

## Security

Every endpoint is unauthenticated on purpose — this is a public demo,
not a multi-tenant service, so the controls target the real exposure
(**cost and abuse**) rather than a generic checklist:

- Per-IP rate limits on the two endpoints that cost money or CPU when hammered (`/query`, `/documents/upload`)
- A global concurrency cap on top of the rate limit, closing a gap that caused a real production incident — see [ENGINEERING.md](ENGINEERING.md#build-journal)
- An optional admin token gating the one destructive operation (document deletion)
- Security headers on every response (HSTS, CSP, X-Frame-Options, nosniff)
- Secrets only ever in env vars, never committed
- Row Level Security enabled on the Supabase tables, closing a public REST exposure Supabase's own linter flagged
- **Known gap, stated rather than hidden:** no prompt-injection defense against malicious uploaded documents

## Known limitations

Found by actually running the system and evaluation suite, not predicted
in advance. Full case studies with real before/after numbers in
[ENGINEERING.md](ENGINEERING.md#known-limitations).

- Retrieval is strong on literal terms, measurably weaker on paraphrases — root-caused across two real runs and two separate bugs, partially fixed with query expansion
- A small local LLM (Ollama) shows real answer variance between identical runs — confirmed via a direct A/B against a larger cloud model
- A generation-side precision/attribution weakness, independent of retrieval quality — isolated with a pipeline-free test against the model directly
- Query reformulation needed grounding in the actual corpus to stop it guessing wrong domains for ambiguous terms — found, fixed, verified live
- Lexical search is Postgres full-text, not literal BM25; `tsvector` is computed on the fly, not indexed
- No OCR/table extraction, no claim-level citation verification, no human-review queue

## Quick start

Requires Docker + Docker Compose, and (for the default free setup)
[Ollama](https://ollama.com) running locally.

```bash
git clone <this-repo>
cd hybrid-rag-platform
cp .env.example .env
ollama pull llama3.1:8b        # one-time, ~4.7GB — skip if using LLM_PROVIDER=anthropic instead
docker compose up --build
```

Upload a document and ask it something:

```bash
curl -X POST http://localhost:8000/documents/upload -F "file=@/path/to/your/document.pdf"

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about X?"}'
```

Or skip curl entirely — a demo UI is included:

**http://localhost:8000/ui**

Upload documents, ask questions, and watch the actual retrieval trace —
hybrid search candidates, rerank scores, and any query reformulation —
rendered live, not just described. This is the fastest way to see the
system actually working, including the agentic retry loop in action.

Interactive API docs (auto-generated by FastAPI): http://localhost:8000/docs

Every tunable lives in `.env` — see `.env.example` for the full list, or
the [full configuration reference](ENGINEERING.md#configuration-reference)
for what each one does.

## Repo structure

```
app/
  main.py                 FastAPI entrypoint, also mounts /ui (demo frontend)
  static/
    index.html              Demo UI — visualizes the retrieval trace live
  core/
    config.py              Typed settings — nothing hard-coded, every
                            tunable (chunk size, retrieval K, rerank
                            threshold, iteration cap) is an env var
    database.py             Async engine, session management, pgvector init
  api/
    health.py                Liveness + readiness endpoints
    documents.py               Upload, list, inspect chunks
    query.py                    The agentic RAG loop endpoint
  models/
    document.py                 Document + Chunk (pgvector column) tables
  services/
    ingestion/                   parser.py, chunker.py + semantic_chunker.py
                                  (swappable via chunking_strategy.py), embedder.py, pipeline.py
    retrieval/                    vector_search.py, keyword_search.py,
                                   hybrid_search.py, reranker.py,
                                   query_reformulation.py, query_expansion.py,
                                   expanded_search.py, agentic_retrieval.py
    generation/                    llm_client.py (Ollama/Anthropic dispatch), answer.py
    evaluation/                     golden_dataset.py, metrics.py, llm_judge.py, run_eval.py, reports/
tests/                          Pure-logic unit tests, offline-runnable, one file per service
```

## Running tests

```bash
pip install -r requirements.txt
pytest
```

All tests are pure-logic (chunking, RRF fusion, sigmoid scoring, stop-
decision logic, evaluation metrics) — no live DB, model, or LLM call
required, so they run in under a second.

## Evaluation

An 11-question golden dataset, run end to end against the live system,
not eyeballed spot checks:

```bash
docker compose exec api python -m app.services.evaluation.run_eval
```

**Metrics:** Recall@K, MRR, keyword coverage, LLM-as-judge (a second,
independent LLM call assessing semantic correctness), and
correct-abstention rate. The judge caught a real scoring gap on its
first live run — a technically keyword-complete answer that was actually
substantively thin. Full methodology, including a wrong theory caught
and corrected mid-investigation, in
[ENGINEERING.md](ENGINEERING.md#evaluation).

## Deployment

Live on [Render](https://render.com) (API) + [Supabase](https://supabase.com)
(database + pgvector), both free tier. The public deployment runs a
leaner retrieval config than local dev — a deliberate, documented trade
against the free tier's CPU limits, not a hidden compromise. Full deploy
steps and the real free-tier issues hit and fixed along the way:
[ENGINEERING.md](ENGINEERING.md#deployment).

## v2 roadmap (deferred, not built)

Multimodal document processing (OCR, tables), claim-level citation
verification, confidence-aware human review queue, full
observability/tracing integration, per-document-type retrieval
weighting.

## License

MIT — see [LICENSE](LICENSE).
