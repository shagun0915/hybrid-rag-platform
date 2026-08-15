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
"""

from app.services.generation.llm_client import call_llm

REFORMULATION_SYSTEM_PROMPT = """You rewrite search queries to improve document retrieval.
Given a question that retrieved weak or irrelevant results, rewrite it as a clearer, more specific search query.
Respond with ONLY the rewritten query text — no explanation, no quotation marks, nothing else."""


async def reformulate_query(original_question: str) -> str:
    user_message = (
        f"Original question: {original_question}\n\n"
        "This question did not retrieve strong matches from the document "
        "collection. Rewrite it as a more specific or differently-phrased "
        "search query that might retrieve better results."
    )
    reformulated = await call_llm(REFORMULATION_SYSTEM_PROMPT, user_message)
    return reformulated.strip().strip('"')
