"""
LLM-as-judge evaluation.

Upgrades the faithfulness check flagged as a known limitation since Day
6: `keyword_coverage` in metrics.py is fast and free, but it's just
substring presence — a technically-wrong answer that happens to contain
the right number would still "pass," and a correct answer phrased
differently than expected would still "fail." This module adds a
second, complementary signal: ask a separate LLM call whether the
answer actually, semantically, correctly addresses the question — not
whether the answer text happens to overlap with a keyword list.

Deliberately additive, not a replacement. run_eval.py reports both
`keyword_coverage` and `llm_judge_correct` for every answerable case, so
disagreements between them are visible rather than hidden — a case
where the judge says correct but keyword_coverage is low (a correct
answer just phrased differently) is a genuinely different, useful
finding than one where they agree.

Only applied to answerable cases. The unanswerable case's correctness
is "did it abstain," which `is_correct_abstention` already checks well
— asking an LLM to judge "is this an abstention" would be solving an
already-solved problem with a slower, more expensive tool.
"""

import json

from app.services.generation.llm_client import call_llm

JUDGE_SYSTEM_PROMPT = """You are evaluating whether an AI-generated answer correctly and directly addresses a question, given a description of what a correct answer should contain.
Respond with ONLY a JSON object in this exact format, no other text:
{"correct": true or false, "reasoning": "one brief sentence explaining your judgment"}
Judge based on factual correctness and whether the question is actually answered — not writing style, length, or phrasing."""


def _parse_judge_response(raw_text: str) -> dict:
    """Pure parsing logic, deliberately separated from the LLM call so
    it's unit-testable without a real model. LLMs don't always return
    perfectly clean JSON (extra whitespace, a stray code fence, trailing
    commentary) — this handles the common cases defensively rather than
    letting a malformed response crash the whole eval run."""
    text = raw_text.strip()

    # Strip a markdown code fence if the model wrapped its JSON in one
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {
            "correct": None,
            "reasoning": f"Could not parse judge response as JSON: {raw_text[:150]}",
        }

    if not isinstance(parsed, dict) or "correct" not in parsed:
        return {
            "correct": None,
            "reasoning": f"Judge response missing expected 'correct' field: {raw_text[:150]}",
        }

    return {
        "correct": parsed.get("correct"),
        "reasoning": parsed.get("reasoning", ""),
    }


async def judge_answer(question: str, answer: str, expected_answer_summary: str) -> dict:
    """Returns {"correct": True/False/None, "reasoning": str}.
    correct=None means the judge call itself failed or returned something
    unparseable — treated as a distinct outcome from a confirmed wrong
    answer, not silently coerced into False."""
    user_message = (
        f"Question: {question}\n\n"
        f"What a correct answer should say: {expected_answer_summary}\n\n"
        f"The AI's actual answer: {answer}\n\n"
        "Does the AI's answer correctly and directly address the question?"
    )

    try:
        raw = await call_llm(JUDGE_SYSTEM_PROMPT, user_message)
    except Exception as exc:
        return {"correct": None, "reasoning": f"Judge LLM call failed: {exc}"}

    return _parse_judge_response(raw)
