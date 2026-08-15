"""
Evaluation runner. Calls the live, running /query endpoint for every case
in the golden dataset, scores each result, and prints + saves an
aggregate report.

Run it with the app already up (docker compose up), from inside the
running container:

    docker compose exec api python -m app.services.evaluation.run_eval

Why inside the container rather than a host-side script: httpx is
already installed there (no new host dependency to manage), and it talks
to the API over localhost since it's the same running process — no
networking wrinkles to work around.

Per the project spec's evaluation principle: this script produces
numbers by actually calling the real, running system. Nothing in the
aggregate report is invented or hand-typed — if you re-run this after
Day 7's changes, the numbers will genuinely reflect whatever the system
does at that point, for better or worse.

Report detail level: the saved report includes full `sources` (chunk
index, rerank score, content preview) for every case, not just
filenames. This wasn't planned upfront — it was added after the first
real eval run produced two failures that couldn't be fully root-caused
because the original report only saved filenames. Real usage surfaced a
real observability gap; this is the fix, not a hypothetical improvement.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.services.evaluation.golden_dataset import GOLDEN_DATASET
from app.services.evaluation.metrics import (
    recall_at_k,
    reciprocal_rank,
    keyword_coverage,
    is_correct_abstention,
)

API_BASE_URL = "http://localhost:8000"
REPORTS_DIR = Path(__file__).parent / "reports"


async def run_case(client: httpx.AsyncClient, case: dict) -> dict:
    start = time.monotonic()
    response = await client.post("/query", json={"question": case["question"]})
    latency_seconds = round(time.monotonic() - start, 2)

    response.raise_for_status()
    data = response.json()

    answer = data.get("answer", "")
    sources = data.get("sources", [])
    retrieved_filenames = [s["filename"] for s in sources]
    attempts = data.get("retrieval_debug", {}).get("attempts", [])

    result = {
        "id": case["id"],
        "question": case["question"],
        "answer": answer,
        "retrieved_filenames": retrieved_filenames,
        # Full source detail (chunk_index, rerank_score, content_preview),
        # not just filenames — added after Day 6's first real eval run
        # produced two failures that couldn't be root-caused because the
        # report only saved filenames. This is what "log everything about
        # each attempt so it can be inspected" (from the project spec)
        # actually means in practice: found by hitting the gap for real,
        # not designed in from a checklist.
        "sources": sources,
        "retrieval_attempts": attempts,
        "num_retrieval_attempts": len(attempts),
        "latency_seconds": latency_seconds,
    }

    if case.get("unanswerable"):
        result["type"] = "unanswerable"
        result["correct_abstention"] = is_correct_abstention(answer)
        result["passed"] = result["correct_abstention"]
    else:
        expected_doc = case["expected_source_document"]
        expected_keywords = case["expected_answer_keywords"]
        recall = recall_at_k(retrieved_filenames, expected_doc)
        rr = reciprocal_rank(retrieved_filenames, expected_doc)
        kw_cov = keyword_coverage(answer, expected_keywords)

        result["type"] = "answerable"
        result["expected_source_document"] = expected_doc
        result["retrieval_recall"] = recall
        result["reciprocal_rank"] = rr
        result["keyword_coverage"] = kw_cov
        # A case "passes" if the right document was retrieved AND the
        # answer contains at least half the expected keywords — a
        # deliberately simple bar, not a rigorous correctness proof.
        result["passed"] = recall and kw_cov >= 0.5

    return result


def print_report(results: list[dict]) -> None:
    answerable = [r for r in results if r["type"] == "answerable"]
    unanswerable = [r for r in results if r["type"] == "unanswerable"]

    print("\n" + "=" * 72)
    print("EVALUATION REPORT")
    print("=" * 72)

    print(f"\n{'ID':<28}{'Passed':<10}{'Recall':<10}{'RR':<8}{'KW Cov':<10}{'Attempts':<10}{'Latency'}")
    print("-" * 72)
    for r in answerable:
        print(
            f"{r['id']:<28}{str(r['passed']):<10}{str(r['retrieval_recall']):<10}"
            f"{r['reciprocal_rank']:<8}{r['keyword_coverage']:<10}"
            f"{r['num_retrieval_attempts']:<10}{r['latency_seconds']}s"
        )
    for r in unanswerable:
        print(
            f"{r['id']:<28}{str(r['passed']):<10}{'—':<10}{'—':<8}{'—':<10}"
            f"{r['num_retrieval_attempts']:<10}{r['latency_seconds']}s"
        )

    if answerable:
        mean_recall = round(sum(r["retrieval_recall"] for r in answerable) / len(answerable), 4)
        mean_rr = round(sum(r["reciprocal_rank"] for r in answerable) / len(answerable), 4)
        mean_kw = round(sum(r["keyword_coverage"] for r in answerable) / len(answerable), 4)
    else:
        mean_recall = mean_rr = mean_kw = None

    total_passed = sum(1 for r in results if r["passed"])
    mean_latency = round(sum(r["latency_seconds"] for r in results) / len(results), 2)
    mean_attempts = round(sum(r["num_retrieval_attempts"] for r in results) / len(results), 2)

    print("-" * 72)
    print(f"\nAggregate ({len(answerable)} answerable + {len(unanswerable)} unanswerable cases):")
    print(f"  Recall@K (mean):         {mean_recall}")
    print(f"  MRR:                     {mean_rr}")
    print(f"  Keyword coverage (mean): {mean_kw}")
    print(f"  Overall pass rate:       {total_passed}/{len(results)} ({round(100*total_passed/len(results),1)}%)")
    print(f"  Mean latency:            {mean_latency}s")
    print(f"  Mean retrieval attempts: {mean_attempts}  (>1.0 means reformulation fired on average)")
    print("=" * 72 + "\n")


async def main():
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=180.0) as client:
        results = []
        for case in GOLDEN_DATASET:
            print(f"Running: {case['id']}...")
            result = await run_case(client, case)
            results.append(result)

    print_report(results)

    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"eval_run_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump({"timestamp": timestamp, "results": results}, f, indent=2)
    print(f"Full report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
