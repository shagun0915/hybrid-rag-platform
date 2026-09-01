"""
Tests for LLM-as-judge response parsing. `_parse_judge_response` is pure
logic — no LLM call — so it's fully testable offline with fabricated
model outputs, including malformed ones. `judge_answer` (which calls the
LLM) is exercised manually via run_eval.py, same pattern as every other
model-dependent piece.
"""

from app.services.evaluation.llm_judge import _parse_judge_response


def test_parse_judge_response_clean_json():
    raw = '{"correct": true, "reasoning": "The answer correctly states the value."}'
    result = _parse_judge_response(raw)

    assert result["correct"] is True
    assert result["reasoning"] == "The answer correctly states the value."


def test_parse_judge_response_correct_false():
    raw = '{"correct": false, "reasoning": "The answer is missing the key fact."}'
    result = _parse_judge_response(raw)

    assert result["correct"] is False


def test_parse_judge_response_strips_markdown_code_fence():
    # LLMs sometimes wrap JSON in a code fence even when told not to
    raw = '```json\n{"correct": true, "reasoning": "Looks right."}\n```'
    result = _parse_judge_response(raw)

    assert result["correct"] is True
    assert result["reasoning"] == "Looks right."


def test_parse_judge_response_strips_plain_code_fence():
    raw = '```\n{"correct": false, "reasoning": "Wrong."}\n```'
    result = _parse_judge_response(raw)

    assert result["correct"] is False


def test_parse_judge_response_malformed_json_returns_none_not_crash():
    # Defensive: garbage input should degrade gracefully, not raise
    raw = "I think the answer is correct because it mentions the right number."
    result = _parse_judge_response(raw)

    assert result["correct"] is None
    assert "Could not parse" in result["reasoning"]


def test_parse_judge_response_missing_correct_field():
    raw = '{"reasoning": "Some analysis but no verdict field."}'
    result = _parse_judge_response(raw)

    assert result["correct"] is None
    assert "missing expected" in result["reasoning"]


def test_parse_judge_response_missing_reasoning_defaults_to_empty():
    raw = '{"correct": true}'
    result = _parse_judge_response(raw)

    assert result["correct"] is True
    assert result["reasoning"] == ""


def test_parse_judge_response_non_dict_json():
    # e.g. the model returns a JSON array or a bare string instead of an object
    raw = '["correct", "reasoning"]'
    result = _parse_judge_response(raw)

    assert result["correct"] is None
