"""
Query reformulation: when retrieval comes back weak, ask the LLM to
rewrite the question into something more likely to retrieve well, rather
than just giving up or returning a low-confidence answer.

This is the specific piece that makes the retrieval loop "agentic" rather
than a fixed pipeline (Days 1-4 were all fixed pipelines: input always
flows through the same fixed steps in the same order). Here, the system
makes a decision — "the evidence is weak, let me try a different
phrasing" — based on the outcome of a previous step, not a predetermined
script. See agentic_retrieval.py for the loop that decides *when* to call
this and *how many times* it's allowed to (the iteration cap is the
critical safety piece — see that module's docstring).

Grounded in the corpus (v2 follow-up, fixing a real found bug): the
original version of this function had zero visibility into what's
actually in the document collection — it just asked the LLM to guess a
"clearer, more specific" rephrasing from scratch. Found via the fourth
case study in the README's Known Limitations: asked "what is full form
of VAMP" (a Visa-related acronym in an uploaded document), and
reformulation guessed VAMP meant "Vascular Adhesion Molecule" — real
biology terms with zero connection to the actual corpus — because it had
nothing to ground its guess against. Now, the previous attempt's actual
retrieved content (even if it scored too low to be confident) is passed
in as context, so reformulation refines around real corpus vocabulary
instead of guessing a domain unrelated to what's actually there.
"""

from app.services.generation.llm_client import call_llm

REFORMULATION_SYSTEM_PROMPT = """You rewrite search queries to improve document retrieval.
Given a question that retrieved weak or irrelevant results, rewrite it as a clearer, more specific search query.
If context snippets from the document collection are provided, base your rewrite on the vocabulary and subject matter actually present in those snippets — do not guess an unrelated meaning for an ambiguous term or acronym based on general knowledge.
Respond with ONLY the rewritten query text — no explanation, no quotation marks, nothing else."""


def _format_context_snippets(candidates: list[dict], max_snippets: int = 3, snippet_length: int = 200) -> str:
    """Pure formatting logic, separated from the LLM call for testability.
    Takes the previous attempt's top candidates (even low-scoring ones —
    they're still real corpus content) and produces a short grounding
    block for the reformulation prompt."""
    if not candidates:
        return ""

    lines = []
    for c in candidates[:max_snippets]:
        preview = c.get("content", "")[:snippet_length].strip()
        filename = c.get("filename", "unknown source")
        if preview:
            lines.append(f"- From {filename}: \"{preview}...\"")

    if not lines:
        return ""

    return "Context snippets actually found in the document collection so far (low relevance, but real):\n" + "\n".join(lines)


async def reformulate_query(original_question: str, previous_candidates: list[dict] | None = None) -> str:
    context_block = _format_context_snippets(previous_candidates or [])

    user_message = (
        f"Original question: {original_question}\n\n"
        "This question did not retrieve strong matches from the document "
        "collection. Rewrite it as a more specific or differently-phrased "
        "search query that might retrieve better results."
    )
    if context_block:
        user_message += f"\n\n{context_block}"

    reformulated = await call_llm(REFORMULATION_SYSTEM_PROMPT, user_message)
    return reformulated.strip().strip('"')
