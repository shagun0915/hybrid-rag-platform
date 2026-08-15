"""
Day 6 tests: every function in metrics.py is pure (no DB, no HTTP, no
LLM call), so all of them are fully testable offline. The actual
evaluation run (run_eval.py hitting the live /query endpoint) is
exercised manually — same pattern as every previous day's live-system
pieces.
"""

from app.services.evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    keyword_coverage,
    is_correct_abstention,
)


def test_recall_at_k_true_when_present():
    assert recall_at_k(["a.pdf", "b.pdf", "c.pdf"], "b.pdf") is True


def test_recall_at_k_false_when_absent():
    assert recall_at_k(["a.pdf", "c.pdf"], "b.pdf") is False


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["b.pdf", "a.pdf"], "b.pdf") == 1.0


def test_reciprocal_rank_third_position():
    assert reciprocal_rank(["a.pdf", "c.pdf", "b.pdf"], "b.pdf") == round(1 / 3, 4)


def test_reciprocal_rank_not_found_is_zero():
    assert reciprocal_rank(["a.pdf", "c.pdf"], "b.pdf") == 0.0


def test_keyword_coverage_all_present():
    assert keyword_coverage("The answer is 98.49% accuracy", ["98.49"]) == 1.0


def test_keyword_coverage_partial():
    answer = "The system uses Dynamics 365 for enterprise workflows"
    # Only one of two expected keywords present
    coverage = keyword_coverage(answer, ["Dynamics 365", "SonarQube"])
    assert coverage == 0.5


def test_keyword_coverage_case_insensitive():
    assert keyword_coverage("SONARQUBE was used", ["SonarQube"]) == 1.0


def test_keyword_coverage_empty_keywords_defaults_true():
    assert keyword_coverage("anything", []) == 1.0


def test_is_correct_abstention_detects_standard_phrasing():
    answer = "I don't have enough information in the provided context to answer your question."
    assert is_correct_abstention(answer) is True


def test_is_correct_abstention_false_for_confident_answer():
    answer = "The Rank-1 accuracy was 98.49%, according to Excerpt 1."
    assert is_correct_abstention(answer) is False


# --- Regression tests: real answers that exposed real gaps in this file ---
# Both of these are verbatim outputs from an actual eval run — the model
# behaved correctly in both cases; this file's scoring logic was the
# thing that was wrong. Kept as regression tests, not just fixed
# silently, so a future change can't reintroduce either gap unnoticed.

def test_regression_decimal_phrasing_of_percentage_is_accepted():
    # Model said "0.0040" instead of "0.40%" — same value, different
    # phrasing. First eval run scored this a false failure.
    real_answer = "0.0040 (Excerpt 2)"
    coverage = keyword_coverage(real_answer, ["0.40", "0.004"])
    assert coverage >= 0.5


def test_regression_no_information_phrasing_is_detected_as_abstention():
    # Model said "no information about" — a correct refusal that the
    # original ABSTENTION_PHRASES list didn't recognize.
    real_answer = "There is no information about the capital of France in the provided context excerpts."
    assert is_correct_abstention(real_answer) is True
