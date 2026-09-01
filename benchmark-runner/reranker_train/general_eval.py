#!/usr/bin/env python3
"""
General reranking eval — does the student hold up outside our own domain?

Our benchmark measures conversational memory on LoCoMo. That is the thing we
care about, but a model tuned hard to one distribution can quietly become
useless everywhere else, and we would not see it. This scores NDCG@10 on
standard IR datasets and compares the student against the incumbent and the
teacher on the same queries.

DATASET CHOICE IS THE POINT. It only uses sets that are NOT in our training
mix: SciFact, NFCorpus and TREC-COVID. HotpotQA and MS MARCO are excluded
because we trained on them, so a good score there would prove nothing.

Reranking a full BEIR corpus is far too slow for a single run, so this uses the
standard shortcut: take the qrels-positive documents plus a fixed random sample
of negatives per query, rerank that pool, and score NDCG@10 within it. Absolute
numbers are therefore not comparable to published BEIR leaderboards — only the
comparison BETWEEN models on this identical pool is meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import httpx

# Deliberately excludes anything in our stage-1 mix. MS MARCO and HotpotQA are
# out because we trained on them, so a good score there would prove nothing.
#
# Wikipedia-corpus sets (NQ, DBPedia, FEVER, Climate-FEVER) are also out. We did
# not train on their queries, but HotpotQA and MIRACL are built over Wikipedia,
# so the student has seen a lot of that corpus and a score there overstates
# generalisation. Everything below is a domain the student has never met.
DATASETS = [
    ("BeIR/scifact", "BeIR/scifact-qrels", "scifact"),                     # scientific claim verification
    ("BeIR/nfcorpus", "BeIR/nfcorpus-qrels", "nfcorpus"),                  # medical / nutrition
    ("BeIR/trec-covid", "BeIR/trec-covid-qrels", "trec-covid"),            # biomedical
    ("BeIR/fiqa", "BeIR/fiqa-qrels", "fiqa"),                              # financial question answering
    ("BeIR/arguana", "BeIR/arguana-qrels", "arguana"),                     # counter-argument retrieval
    ("BeIR/scidocs", "BeIR/scidocs-qrels", "scidocs"),                     # citation prediction
    ("BeIR/webis-touche2020", "BeIR/webis-touche2020-qrels", "touche2020"),# argument retrieval
    ("BeIR/quora", "BeIR/quora-qrels", "quora"),                           # duplicate question detection
]


def ndcg_at_k(ranked_rels: list[int], k: int = 10) -> float:
    dcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ranked_rels[:k]))
    ideal = sorted(ranked_rels, reverse=True)
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def rerank(client: httpx.Client, url: str, query: str, texts: list[str]) -> list[int]:
    """Return document indices ordered best-first."""
    r = client.post(url.rstrip("/") + "/rerank",
                    json={"query": query, "texts": texts, "return_text": False})
    r.raise_for_status()
    return [item["index"] for item in r.json()]


def load_dataset_pools(name: str, qrels_name: str, n_queries: int, n_negatives: int, seed: int):
    """Build (query, [texts], [rels]) pools from a BeIR dataset."""
    from datasets import load_dataset

    rng = random.Random(seed)
    corpus = load_dataset(name, "corpus", split="corpus")
    queries = load_dataset(name, "queries", split="queries")
    qrels = load_dataset(qrels_name, split="test")

    docs = {str(r["_id"]): (r.get("title", "") + " " + r.get("text", "")).strip() for r in corpus}
    qtext = {str(r["_id"]): r["text"] for r in queries}

    by_query: dict[str, list[tuple[str, int]]] = {}
    for r in qrels:
        qid, did, score = str(r["query-id"]), str(r["corpus-id"]), int(r["score"])
        if score > 0 and did in docs:
            by_query.setdefault(qid, []).append((did, score))

    all_ids = list(docs)
    pools = []
    for qid in list(by_query)[:n_queries]:
        if qid not in qtext:
            continue
        pos = by_query[qid]
        neg_ids = [d for d in rng.sample(all_ids, min(n_negatives + len(pos), len(all_ids)))
                   if d not in {p for p, _ in pos}][:n_negatives]
        items = [(docs[d], s) for d, s in pos] + [(docs[d], 0) for d in neg_ids]
        rng.shuffle(items)
        pools.append((qtext[qid], [t for t, _ in items], [s for _, s in items]))
    return pools


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="TEI endpoint serving the model under test")
    ap.add_argument("--label", required=True, help="Name for this model in the output")
    ap.add_argument("--out", default="general_eval.json")
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--negatives", type=int, default=99)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    results = {}
    with httpx.Client(timeout=180.0) as client:
        info = client.get(args.url.rstrip("/") + "/info").json()
        print(f"model: {info.get('model_id')}  ctx {info.get('max_input_length')}", flush=True)
        for name, qrels_name, short in DATASETS:
            try:
                pools = load_dataset_pools(name, qrels_name, args.queries, args.negatives, args.seed)
            except Exception as exc:
                print(f"  {short}: unavailable ({type(exc).__name__}: {str(exc)[:90]})", flush=True)
                continue
            scores = []
            for query, texts, rels in pools:
                try:
                    order = rerank(client, args.url, query, texts)
                except Exception as exc:
                    print(f"    rerank failed: {type(exc).__name__}", flush=True)
                    continue
                scores.append(ndcg_at_k([rels[i] for i in order], 10))
            if scores:
                results[short] = round(sum(scores) / len(scores), 4)
                print(f"  {short:12s} NDCG@10 {results[short]:.4f}  over {len(scores)} queries", flush=True)

    if results:
        results["average"] = round(sum(results.values()) / len(results), 4)
        print(f"  {'AVERAGE':12s} NDCG@10 {results['average']:.4f}", flush=True)

    out = Path(args.out)
    payload = {}
    if out.exists():
        payload = json.loads(out.read_text())
    payload[args.label] = results
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
