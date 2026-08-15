# Evaluation service — built Day 6 ✅

Golden dataset (11 hand-verified cases) + Recall@K / MRR / keyword
coverage / correct-abstention scoring, run against the live system via
`run_eval.py`. Reports saved to `reports/` as timestamped JSON — real
measured numbers, not invented ones.

Honest limitation: keyword-coverage faithfulness checking is substring
presence, not semantic correctness. LLM-as-judge is the natural v2
upgrade.
