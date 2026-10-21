#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.5.3"]
# ///
"""Validate all direct runs, export results, and compute paired quality differences."""

import argparse
import collections
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rerank_eval import ROOT, read_json, summarize, write_json


def paired_interval(rows_a, rows_b, metric):
    if rows_a.keys() != rows_b.keys():
        raise ValueError("Paired comparison requires identical queries")
    groups = collections.defaultdict(list)
    for qid in sorted(rows_a):
        a, b = rows_a[qid], rows_b[qid]
        if a["group"] != b["group"]:
            raise ValueError("Paired comparison group mismatch")
        groups[a["group"]].append(a["metrics"][metric] - b["metrics"][metric])
    totals = np.array([sum(values) for values in groups.values()])
    counts = np.array([len(values) for values in groups.values()])
    rng = np.random.default_rng(20261003)
    samples = rng.integers(0, len(groups), size=(5000, len(groups)))
    distribution = totals[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return {
        "delta": float(totals.sum() / counts.sum()),
        "ci95": [float(v) for v in np.quantile(distribution, [0.025, 0.975])],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "results" / "leaderboard" / "reranker",
    )
    args = parser.parse_args()
    models = [
        c["reranker_id"] for c in read_json(ROOT / "reranker_models.json")["rerankers"]
    ]
    comparisons = []
    for dataset, name, labels in [
        ("locomo", "locomo10-top100", None),
        ("scifact", "scifact-test-top100", None),
        (
            "scifact-evidence",
            "scifact-test-top100",
            ROOT / "datasets" / "reranker" / "scifact-evidence-labels.json",
        ),
    ]:
        fixture = ROOT / "datasets" / "reranker" / (name + ".json.gz")
        data = read_json(fixture)
        records = {}
        for model in models:
            summarize(
                SimpleNamespace(
                    fixture=fixture,
                    labels=labels,
                    model=model,
                    records=args.records,
                    output=args.output,
                )
            )
            records[model] = {
                row["query_id"]: row
                for row in read_json(
                    args.output / "records" / (model + "--" + dataset + ".json.gz")
                )
            }
        pairs = []
        for i, a in enumerate(models):
            for b in models[i + 1 :]:
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        **{
                            metric: paired_interval(records[a], records[b], metric)
                            for metric in ["ndcg_at_10", "mrr", "recall_at_5"]
                        },
                    }
                )
        comparisons.append(
            {
                "dataset": dataset,
                "fixture_sha256": data["fixture_sha256"],
                "evaluation_label_sha256": read_json(labels)["label_sha256"]
                if labels
                else None,
                "pairs": pairs,
            }
        )
    write_json(
        args.output.parent / "reranker-analysis" / "paired-comparisons.json",
        {
            "method": "5000 paired bootstrap samples; whole conversations for LoCoMo, individual queries for SciFact; 95% percentile intervals; unadjusted for multiple comparisons",
            "comparisons": comparisons,
        },
    )


if __name__ == "__main__":
    main()
