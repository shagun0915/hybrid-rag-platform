"""
Query expansion.

Concept note: rather than searching with only the literal question,
generate a few alternate phrasings and search with all of them, then
fuse every variant's results together (same RRF used to fuse
vector+keyword search in hybrid_search.py — see that file's docstring
for why RRF specifically, not a weighted average). A chunk that a
numerically larger document class out-competed under one phrasing gets
another chance under a differently-worded variant.

This is proactive — used on every attempt, including the first — unlike
Day 5's query_reformulation.py, which is reactive (only fires after a
confirmed weak attempt). The two are complementary: expansion widens the
net immediately; reformulation is a fallback if the widened net still
isn't confident enough.

Honest scope note: this doesn't fix the "SonarQube" reranker weakness
documented in Known Limitations (that's a reranker-side vocabulary-
mismatch problem, not something retrieval breadth alone can fix) — it
targets the corpus-imbalance failure mode specifically, where the
correct chunk never even enters the candidate pool under one phrasing.
"""

from app.core.config import settings
from app.services.generation.llm_client import call_llm

EXPANSION_SYSTEM_PROMPT = """You generate alternate phrasings of a search query to improve document retrieval recall.
Given a question, produce different phrasings that ask the same thing using different vocabulary and sentence structure.
Respond with ONLY the alternate phrasings, one per line, no numbering, no explanation, no quotation marks."""


def _parse_variants(raw_text: str, original_question: str, num_variants: int) -> list[str]:
    """Pure parsing logic, deliberately separated from the LLM call
    below so it's unit-testable without a real model (see
    tests/test_query_expansion.py)."""
    lines = [line.strip().strip('"') for line in raw_text.strip().split("\n") if line.strip()]
    variants = lines[:num_variants]

    all_queries = [original_question] + variants

    # De-duplicate case-insensitively, preserving order — the LLM
    # sometimes echoes the original question back as one of its
    # "alternate" phrasings.
    seen = set()
    deduped = []
    for q in all_queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(q)
    return deduped


async def expand_query(question: str, num_variants: int) -> list[str]:
    """Returns the original question plus up to num_variants alternate
    phrasings. Expansion is a best-effort enhancement, not a hard
    dependency — if it's disabled or the LLM call fails, this falls back
    to just the original question rather than failing retrieval
    entirely."""
    if num_variants <= 0:
        return [question]

    user_message = f"Question: {question}\n\nGenerate {num_variants} alternate phrasings."

    try:
        raw = await call_llm(EXPANSION_SYSTEM_PROMPT, user_message)
    except Exception:
        return [question]

    return _parse_variants(raw, question, num_variants)
