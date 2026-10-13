"""Measure retrieval with hit rate and mean reciprocal rank on labelled questions."""
from __future__ import annotations

import json
from pathlib import Path

from .config import ROOT, Settings
from .ingest import load_index


def run_eval(settings: Settings | None = None, questions_path: Path | None = None, verbose: bool = True) -> dict:
    settings = settings or Settings.from_env()
    questions_path = questions_path or ROOT / "data" / "eval_questions.jsonl"
    rows = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    index = load_index(settings)

    answerable = [r for r in rows if r["expected_source"]]
    unanswerable = [r for r in rows if not r["expected_source"]]

    hits = 0
    reciprocal_ranks = []
    misses = []
    for row in answerable:
        results = index.search(row["question"], k=settings.top_k)
        rank = next((i for i, (c, _) in enumerate(results, start=1) if c.metadata["source"] == row["expected_source"]), None)
        if rank:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
            misses.append({"question": row["question"], "expected": row["expected_source"],
                           "got": [c.metadata["source"] for c, _ in results]})

    abstained = 0
    for row in unanswerable:
        results = index.search(row["question"], k=1)
        if not results or results[0][1] < settings.min_similarity:
            abstained += 1

    report = {
        "embeddings": settings.embeddings_provider,
        "top_k": settings.top_k,
        "answerable_questions": len(answerable),
        f"hit_rate@{settings.top_k}": round(hits / len(answerable), 3),
        "mrr": round(sum(reciprocal_ranks) / len(answerable), 3),
        "out_of_scope_questions": len(unanswerable),
        "out_of_scope_refused": f"{abstained}/{len(unanswerable)}",
        "misses": misses,
    }
    if verbose:
        print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_eval()
