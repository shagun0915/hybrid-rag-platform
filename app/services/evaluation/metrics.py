"""
Evaluation metrics. Every function here is pure — takes plain data in,
returns a plain value out, no DB/HTTP/LLM calls. That's deliberate: it's
what makes these fully unit-testable offline (see tests/test_evaluation.py),
and it's good practice generally — the *scoring logic* should never be
tangled up with the *I/O* that produced the data being scored.

Concept notes (Phase 4 from the learning roadmap, made concrete):

Recall@K — did a relevant document appear ANYWHERE in the top-K
retrieved results? Binary per query: yes or no. Says nothing about
*where* in the top-K it landed, just whether it's there at all.

MRR (Mean Reciprocal Rank) — complements Recall@K by caring about
position. If the relevant document is the very first result, that query
scores 1.0. Second position scores 0.5. Tenth position scores 0.1. Not
found at all scores 0. Averaged across all queries, MRR rewards a system
that ranks the right answer *near the top*, not just somewhere in range,
which Recall@K alone can't distinguish.

Keyword coverage — a cheap, free proxy for "is this answer actually
correct," checking whether expected substrings appear in the generated
text. Real limitation, stated plainly: this is presence-of-substring, not
semantic correctness — a system could include the right keyword in a
sentence that actually gets the fact wrong, and this check wouldn't catch
that. An LLM-as-judge (asking a second model "does this answer correctly
address the question, given this context") is the standard stronger
approach, and is the honest v2 upgrade path from here.
"""


def recall_at_k(retrieved_filenames: list[str], expected_filename: str) -> bool:
    return expected_filename in retrieved_filenames


def reciprocal_rank(retrieved_filenames: list[str], expected_filename: str) -> float:
    for position, filename in enumerate(retrieved_filenames, start=1):
        if filename == expected_filename:
            return round(1.0 / position, 4)
    return 0.0


def keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(found / len(expected_keywords), 4)


ABSTENTION_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "cannot answer",
    "can't answer",
    "no relevant",
    "no information",
    "not contain",
    "doesn't contain",
    "does not contain",
    "insufficient",
    "unable to answer",
]


def is_correct_abstention(answer: str) -> bool:
    """For the deliberate unanswerable case: did the system correctly say
    it doesn't know, rather than hallucinate an answer? Simple phrase
    matching — same honest limitation as keyword_coverage above."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in ABSTENTION_PHRASES)
