#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["pytrec-eval-terrier==0.5.10"]
# ///
"""Independently verify published binary-evidence metrics with TREC evaluation."""

import argparse
import importlib.metadata
from pathlib import Path

import pytrec_eval

from rerank_eval import (
    DATA,
    RESULTS,
    ROOT,
    digest,
    read_json,
    validate_fixture,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS.parent / "reranker-analysis" / "metric-validation.json",
    )
    args = parser.parse_args()
    models = [
        c["reranker_id"] for c in read_json(ROOT / "reranker_models.json")["rerankers"]
    ]
    mappings = {
        "ndcg_at_10": "ndcg_cut_10",
        "mrr": "recip_rank",
        "recall_at_5": "recall_5",
        "recall_at_10": "recall_10",
    }
    checks, maximum_error, runs = 0, 0.0, []
    for dataset, name, labels_path in [
        ("locomo", "locomo10-top100", None),
        ("scifact", "scifact-test-top100", None),
        (
            "scifact-evidence",
            "scifact-test-top100",
            DATA / "scifact-evidence-labels.json",
        ),
    ]:
        fixture = read_json(DATA / (name + ".json.gz"))
        validate_fixture(fixture)
        queries = {q["id"]: q for q in fixture["queries"]}
        qrels = {qid: q["relevant"] for qid, q in queries.items()}
        label_view = read_json(labels_path) if labels_path else None
        if label_view:
            if (
                digest({k: v for k, v in label_view.items() if k != "label_sha256"})
                != label_view["label_sha256"]
            ):
                raise ValueError("Evidence-label hash mismatch")
            qrels = label_view["labels"]
            queries = {qid: queries[qid] for qid in qrels}
        if any(grade != 1 for labels in qrels.values() for grade in labels.values()):
            raise ValueError("TREC cross-check requires these binary-evidence fixtures")
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(mappings.values()))
        for model in models:
            filename = model + "--" + dataset
            result = read_json(args.results / (filename + ".json"))
            raw = read_json(args.results / "records" / (filename + ".json.gz"))
            rows = {r["query_id"]: r for r in raw}
            if len(rows) != len(raw) or rows.keys() != queries.keys():
                raise ValueError(f"Incomplete or duplicate query coverage: {filename}")
            if (
                result["fixture_sha256"] != fixture["fixture_sha256"]
                or result["metadata"]["fixture_sha256"] != fixture["fixture_sha256"]
                or result["run_sha256"] != digest(result["metadata"])
                or any(r["run_sha256"] != result["run_sha256"] for r in raw)
                or result["evaluation_label_sha256"]
                != (label_view["label_sha256"] if label_view else None)
            ):
                raise ValueError(f"Mismatched provenance: {filename}")
            run = {}
            for qid, row in rows.items():
                candidates = queries[qid]["candidates"]
                if sorted(row["order"]) != list(range(len(candidates))):
                    raise ValueError(f"Invalid ranking: {filename}, {qid}")
                # Strict rank scores preserve the benchmark's input-order tie rule.
                run[qid] = {
                    candidates[index]: float(len(candidates) - rank)
                    for rank, index in enumerate(row["order"])
                }
            evaluated = evaluator.evaluate(run)
            for ours, trec in mappings.items():
                errors = [
                    abs(row["metrics"][ours] - evaluated[qid][trec])
                    for qid, row in rows.items()
                ]
                aggregate_error = abs(
                    result[ours]
                    - sum(values[trec] for values in evaluated.values()) / len(rows)
                )
                error = max(*errors, aggregate_error)
                if error > 1e-12:
                    raise ValueError(f"TREC metric mismatch: {filename}, {ours}")
                maximum_error = max(maximum_error, error)
                checks += len(rows) + 1
            runs.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "queries": len(rows),
                    "fixture_sha256": fixture["fixture_sha256"],
                    "run_sha256": result["run_sha256"],
                    "evaluation_label_sha256": result["evaluation_label_sha256"],
                }
            )
    report = {
        "evaluator": "pytrec-eval-terrier",
        "version": importlib.metadata.version("pytrec-eval-terrier"),
        "metrics": mappings,
        "comparisons": checks,
        "maximum_absolute_error": maximum_error,
        "runs": runs,
    }
    write_json(args.output, report)
    print(f"TREC validation passed: {checks} comparisons across {len(runs)} result exports")


if __name__ == "__main__":
    main()
