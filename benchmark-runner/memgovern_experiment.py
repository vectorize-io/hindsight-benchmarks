"""Full-corpus MemGovern retrieval, frozen hybrid reranking and independent scoring."""

import argparse
import asyncio
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np

from embedding_experiment import fused_candidates
from memgovern_gpu_prepare import checkpoint_metadata
from memgovern_prepare import (
    DEFAULT_DATA,
    DEFAULT_WORK,
    MODEL_NAMES,
    read_items,
    selected_queries,
    source_manifest,
    validate_vectors,
)
from rerank_eval import (
    ROOT,
    Remote,
    digest,
    load_records,
    mapped_scores,
    rank_scores,
    read_json,
    versions,
    write_json,
)

OUTPUT = ROOT.parent / "results/experiments/memgovern"
PATHS = ("dense", "jev-listwise", "ettin-150m")


def sha_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_input(args, repo):
    source = read_json(args.work / "source.json")
    base = args.data / "Procedural/MemGovern" / repo
    for name, sha in source["repositories"][repo]["files"].items():
        if sha_file(base / name) != sha:
            raise ValueError("Changed source input")
    return base


def validate_pool(pool):
    actual = dict(pool)
    claimed = actual.pop("pool_sha256")
    if digest(actual) != claimed:
        raise ValueError("Candidate pool checksum mismatch")
    if len({q["id"] for q in pool["queries"]}) != len(pool["queries"]):
        raise ValueError("Duplicate frozen queries")
    for q in pool["queries"]:
        for kind in ("dense", "hybrid"):
            if len(q[kind]) != 100 or len(set(q[kind])) != 100:
                raise ValueError("Invalid top100 candidate pool")


def freeze_repository(args, embedding, repo):
    from rank_bm25 import BM25Okapi

    dest = args.work / embedding / "pools" / f"{repo}.json.gz"
    encoding = args.work / embedding / "encodings"
    if dest.exists():
        validate_pool(read_json(dest))
        return True
    paths = {kind: encoding / f"{repo}.{kind}.npz" for kind in ("document", "query")}
    if not all(p.exists() for p in paths.values()):
        return False
    metadata = read_json(encoding / "metadata.json")
    base = repository_input(args, repo)
    corpus = read_items(base / "corpus.jsonl")
    selection = metadata["selection"]
    queries = selected_queries(
        read_items(base / "queries.jsonl"),
        repo,
        selection["count_per_repository"],
        selection["seed"],
    )
    matrices = {}
    execution = {}
    for kind, rows in (("document", corpus), ("query", queries)):
        with np.load(paths[kind], allow_pickle=False) as checkpoint:
            execution[kind] = checkpoint_metadata(checkpoint, metadata)
            if checkpoint["ids"].tolist() != [r["id"] for r in rows]:
                raise ValueError("Encoding identity mismatch")
            matrices[kind] = checkpoint["vectors"].copy()
            validate_vectors(matrices[kind], len(rows))
    ids = [r["id"] for r in corpus]
    bm25 = BM25Okapi([re.findall(r"\w+", r["text"].lower()) for r in corpus])
    pool = {
        "repository": repo,
        "embedding": embedding,
        "corpus_count": len(ids),
        "encoding_metadata": metadata,
        "encoding_execution": execution,
        "encodings_sha256": {kind: sha_file(path) for kind, path in paths.items()},
        "fusion": "BM25Okapi lowercase regex word tokens; full-corpus stable RRF k=60, 1-based ranks",
        "queries": [],
    }
    for query, vector in zip(queries, matrices["query"], strict=True):
        dense = matrices["document"] @ vector
        lexical = bm25.get_scores(re.findall(r"\w+", query["text"].lower()))
        pool["queries"].append(
            {
                "id": repo + "/" + query["id"],
                "source_id": query["id"],
                "text": query["text"],
                "dense": [ids[i] for i in np.argsort(-dense, kind="stable")[:100]],
                "hybrid": fused_candidates(ids, lexical, dense, 100),
            }
        )
    pool["pool_sha256"] = digest(pool)
    validate_pool(pool)
    write_json(dest, pool)
    print(f"FROZEN {embedding} {repo} {len(queries)}", flush=True)
    return True


def freeze(args):
    source = read_json(args.work / "source.json")
    pending = {
        (embedding, repo)
        for embedding in MODEL_NAMES
        for repo in source["repositories"]
    }
    while pending:
        for embedding, repo in sorted(pending):
            if freeze_repository(args, embedding, repo):
                pending.remove((embedding, repo))
        if pending:
            if not args.watch:
                raise ValueError(f"Unprepared repositories: {len(pending)}")
            time.sleep(5)
    print("COMPLETE all candidate pools", flush=True)


def run_metadata(pool, config, workers):
    return {
        "pool_sha256": pool["pool_sha256"],
        "config": config,
        "versions": versions(),
        "protocol": "memgovern-frozen-hybrid-v1",
        "workers": workers,
        "adapter_sha256": sha_file(ROOT / "rerank_eval.py"),
        "listwise_adapter_sha256": sha_file(ROOT / "jev_listwise.py"),
        "batch_size": 32 if config["provider"] == "local" else None,
        "dtype": "bfloat16" if config["provider"] == "local" else None,
        "requests_per_second_per_worker": 2 if config["provider"] != "local" else None,
        "timing": "Concurrent jobs; latency excludes queue wait; warmup excluded",
    }


def checked_order(row, count=100):
    if row["order"] != rank_scores(row["scores"], count) or sorted(
        row["order"]
    ) != list(range(count)):
        raise ValueError("Invalid reranker permutation")


def record_destination(args, embedding, repo):
    return args.work / embedding / "records" / args.reranker / f"{repo}.jsonl"


def append_row(stream, records, row):
    checked_order(row)
    if row["query_id"] in records:
        raise ValueError("Duplicate scored query")
    stream.write(json.dumps(row, allow_nan=False) + "\n")
    stream.flush()
    records[row["query_id"]] = row


def initialize_records(args, embedding, repo, pool, config):
    metadata = run_metadata(
        pool, config, args.workers if args.reranker == "jev-listwise" else 1
    )
    fingerprint = digest(metadata)
    dest = record_destination(args, embedding, repo)
    dest.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(dest, fingerprint)
    expected = {q["id"] for q in pool["queries"]}
    if not records.keys() <= expected:
        raise ValueError("Unknown checkpoint queries")
    for row in records.values():
        checked_order(row)
    write_json(dest.with_suffix(".metadata.json"), metadata)
    return dest, records, fingerprint


async def hosted_repository(args, embedding, repo, pool, config, remotes):
    dest, records, fingerprint = initialize_records(args, embedding, repo, pool, config)
    base = repository_input(args, repo)
    corpus = {r["id"]: r["text"] for r in read_items(base / "corpus.jsonl")}
    queue = asyncio.Queue()
    for q in pool["queries"]:
        if q["id"] not in records:
            queue.put_nowait(q)
    with dest.open("a") as stream:

        async def worker(remote):
            while not queue.empty():
                q = queue.get_nowait()
                started = time.perf_counter()
                before = remote.retries
                scores, usage, scoring = await remote.score(
                    q["text"], [corpus[d] for d in q["hybrid"]]
                )
                row = {
                    "run_sha256": fingerprint,
                    "query_id": q["id"],
                    "group": repo,
                    "order": rank_scores(scores, 100),
                    "scores": scores,
                    "usage": usage,
                    "latency_s": time.perf_counter() - started,
                    "scoring_latency_s": scoring,
                    "retries": remote.retries - before,
                }
                append_row(stream, records, row)

        try:
            async with asyncio.TaskGroup() as group:
                for remote in remotes:
                    group.create_task(worker(remote))
        except BaseException:
            write_json(
                dest.with_suffix(".failure.json"),
                {
                    "completed": len(records),
                    "run_sha256": fingerprint,
                    "message": "Interrupted; incomplete metrics forbidden; failed-attempt billing unknown",
                },
            )
            raise
    print(f"SCORED {embedding} {repo} Jev {len(records)}", flush=True)


async def hosted(args, config):
    remote_args = SimpleNamespace(
        secrets_dir=Path.home() / ".secrets", concurrency=8, requests_per_second=2
    )
    remotes = [Remote(config, remote_args) for _ in range(args.workers)]
    try:
        await asyncio.gather(
            *(
                r.score(
                    "Which planet is red?", ["Mars is red.", "Jupiter is a gas giant."]
                )
                for r in remotes
            )
        )
        await process_pools(
            args,
            lambda embedding, repo, pool: hosted_repository(
                args, embedding, repo, pool, config, remotes
            ),
        )
    finally:
        await asyncio.gather(*(r.close() for r in remotes))


async def process_pools(args, callback):
    source = read_json(args.work / "source.json")
    embeddings = getattr(args, "embeddings", None) or MODEL_NAMES
    if len(set(embeddings)) != len(embeddings):
        raise ValueError("Duplicate model ownership")
    pending = {(e, r) for e in embeddings for r in source["repositories"]}
    while pending:
        for embedding, repo in sorted(pending):
            path = args.work / embedding / "pools" / f"{repo}.json.gz"
            if path.exists():
                pool = read_json(path)
                validate_pool(pool)
                await callback(embedding, repo, pool)
                pending.remove((embedding, repo))
        if pending:
            if not args.watch:
                raise ValueError(f"Missing frozen pools: {len(pending)}")
            await asyncio.sleep(5)


def cached_pair_scores(query, baseline_query, record):
    """A raw cross-encoder logit belongs to a query/card pair, not to its rank."""
    if (
        query["id"] != baseline_query["id"]
        or query["text"] != baseline_query["text"]
        or record["query_id"] != query["id"]
    ):
        raise ValueError("Changed query for pointwise score reuse")
    checked_order(record)
    return dict(zip(baseline_query["hybrid"], record["scores"], strict=True))


def load_pair_cache(args, embedding, repo, pool, config):
    if embedding == "bge-small":
        return {}, None
    baseline = read_json(args.work / "bge-small" / "pools" / f"{repo}.json.gz")
    validate_pool(baseline)
    for key in ("source_sha256", "selection"):
        if pool["encoding_metadata"][key] != baseline["encoding_metadata"][key]:
            raise ValueError("Changed source or query selection for score reuse")
    dest = args.work / "bge-small" / "records" / "ettin-150m" / f"{repo}.jsonl"
    metadata = read_json(dest.with_suffix(".metadata.json"))
    if metadata != run_metadata(baseline, config, 1):
        raise ValueError("Changed pointwise scorer for score reuse")
    fingerprint = digest(metadata)
    records = load_records(dest, fingerprint)
    baseline_queries = {q["id"]: q for q in baseline["queries"]}
    if records.keys() != baseline_queries.keys() or records.keys() != {
        q["id"] for q in pool["queries"]
    }:
        raise ValueError("Incomplete baseline for score reuse")
    return {
        q["id"]: cached_pair_scores(q, baseline_queries[q["id"]], records[q["id"]])
        for q in pool["queries"]
    }, fingerprint


def merge_pair_scores(candidates, cache, computed):
    cached_indices = [i for i, doc in enumerate(candidates) if doc in cache]
    fresh_indices = [i for i, doc in enumerate(candidates) if doc not in cache]
    if len(computed) != len(fresh_indices):
        raise ValueError("Incomplete pointwise score batch")
    rows = [
        {"index": i, "relevance_score": cache[candidates[i]]} for i in cached_indices
    ]
    rows.extend(
        {"index": i, "relevance_score": score}
        for i, score in zip(fresh_indices, computed, strict=True)
    )
    return mapped_scores(rows, len(candidates)), cached_indices


async def local(args, config):
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(8)
    model = CrossEncoder(
        config["model"],
        revision=config["revision"],
        device="cuda:0",
        max_length=config["max_length"],
        model_kwargs={"dtype": torch.bfloat16},
        local_files_only=True,
    )
    for _ in range(3):
        model.predict(
            [("Which planet is red?", "Mars is red.")] * 16,
            batch_size=32,
            activation_fn=torch.nn.Identity(),
        )
    torch.cuda.synchronize()

    async def callback(embedding, repo, pool):
        dest, records, fingerprint = initialize_records(
            args, embedding, repo, pool, config
        )
        execution = {
            "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hardware": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "batch_size": 32,
        }
        execution_path = dest.with_suffix(".execution.json")
        if execution_path.exists() and read_json(execution_path) != execution:
            raise ValueError("Changed GPU owner for checkpoint")
        write_json(execution_path, execution)
        pair_cache, cache_fingerprint = load_pair_cache(
            args, embedding, repo, pool, config
        )
        base = repository_input(args, repo)
        corpus = {r["id"]: r["text"] for r in read_items(base / "corpus.jsonl")}
        with dest.open("a") as stream:
            for q in pool["queries"]:
                if q["id"] in records:
                    continue
                torch.cuda.synchronize()
                started = time.perf_counter()
                cache = pair_cache.get(q["id"], {})
                fresh_docs = [doc for doc in q["hybrid"] if doc not in cache]
                computed = (
                    model.predict(
                        [(q["text"], corpus[d]) for d in fresh_docs],
                        batch_size=32,
                        activation_fn=torch.nn.Identity(),
                    )
                    .astype(float)
                    .tolist()
                    if fresh_docs
                    else []
                )
                scores, cached_indices = merge_pair_scores(q["hybrid"], cache, computed)
                torch.cuda.synchronize()
                row = {
                    "run_sha256": fingerprint,
                    "query_id": q["id"],
                    "group": repo,
                    "order": rank_scores(scores, 100),
                    "scores": scores,
                    "usage": {},
                    "execution_sha256": digest(execution),
                    "cached_candidate_indices": cached_indices,
                    "cache_source_run_sha256": cache_fingerprint,
                    "latency_s": time.perf_counter() - started,
                }
                append_row(stream, records, row)
        print(f"SCORED {embedding} {repo} Ettin {len(records)}", flush=True)

    await process_pools(args, callback)


def rerank(args):
    config = next(
        c
        for c in read_json(ROOT / "reranker_models.json")["rerankers"]
        if c["reranker_id"] == args.reranker
    )
    if args.reranker == "ettin-150m":
        config = {**config, "revision": "025501c4e0f9bbeb4c5b198318e0089ff061cc14"}
    asyncio.run(
        hosted(args, config) if args.reranker == "jev-listwise" else local(args, config)
    )


def single_positive_metrics(order, positive):
    if len(order) != len(set(order)):
        raise ValueError("Duplicate ranking IDs")
    rank = order.index(positive) + 1 if positive in order else None
    return {
        "ndcg_at_10": 1 / math.log2(rank + 1) if rank and rank <= 10 else 0.0,
        "mrr": 1 / rank if rank else 0.0,
        **{
            f"recall_at_{k}": float(rank is not None and rank <= k)
            for k in (1, 5, 10, 100)
        },
    }


def aggregate(rows):
    if not rows or len({r["query_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate result coverage")
    repos = collections.defaultdict(list)
    for row in rows:
        repos[row["group"]].append(row)
    keys = rows[0]["metrics"].keys()
    by_repo = {
        repo: {key: float(np.mean([r["metrics"][key] for r in group])) for key in keys}
        for repo, group in repos.items()
    }
    return {
        "queries": len(rows),
        "repositories": len(repos),
        "repository_macro": {
            key: float(np.mean([r[key] for r in by_repo.values()])) for key in keys
        },
        "query_macro": {
            key: float(np.mean([r["metrics"][key] for r in rows])) for key in keys
        },
        "per_repository": by_repo,
    }


def export(args):
    import pytrec_eval
    from summarize_rerankers import paired_interval

    source = source_manifest(args.data)
    if source != read_json(args.work / "source.json"):
        raise ValueError("Source changed before scoring")
    summaries, all_rows, checks = {}, {}, 0
    for embedding in MODEL_NAMES:
        paths = {name: [] for name in PATHS}
        pool_hashes = {}
        for repo in sorted(source["repositories"]):
            pool = read_json(args.work / embedding / "pools" / f"{repo}.json.gz")
            validate_pool(pool)
            pool_hashes[repo] = pool["pool_sha256"]
            base = repository_input(args, repo)
            doc_ids = {r["id"] for r in read_items(base / "corpus.jsonl")}
            qrels = {
                q: d
                for q, d, _ in (
                    line.split("\t")
                    for line in (base / "qrels.tsv").read_text().splitlines()
                )
            }
            queries = {q["id"]: q for q in pool["queries"]}
            expected = {
                repo + "/" + q["id"]
                for q in selected_queries(
                    read_items(base / "queries.jsonl"), repo, 200, 20261003
                )
            }
            if queries.keys() != expected:
                raise ValueError("Wrong sampled query coverage")
            for q in queries.values():
                if not set(q["dense"]) <= doc_ids or not set(q["hybrid"]) <= doc_ids:
                    raise ValueError("Candidates outside repository corpus")
            for name in PATHS:
                records = None
                if name != "dense":
                    dest = args.work / embedding / "records" / name / f"{repo}.jsonl"
                    metadata = read_json(dest.with_suffix(".metadata.json"))
                    if metadata["pool_sha256"] != pool["pool_sha256"]:
                        raise ValueError("Reranked the wrong pool")
                    config = next(
                        c
                        for c in read_json(ROOT / "reranker_models.json")["rerankers"]
                        if c["reranker_id"] == name
                    )
                    if name == "ettin-150m":
                        config = {
                            **config,
                            "revision": "025501c4e0f9bbeb4c5b198318e0089ff061cc14",
                        }
                    if metadata != run_metadata(pool, config, metadata["workers"]):
                        raise ValueError("Reranker provenance mismatch")
                    write_json(
                        OUTPUT / embedding / "metadata" / name / f"{repo}.json",
                        metadata,
                    )
                    records = load_records(dest, digest(metadata))
                    execution_path = dest.with_suffix(".execution.json")
                    if execution_path.exists():
                        execution = read_json(execution_path)
                        if any(
                            r.get("execution_sha256", digest(execution))
                            != digest(execution)
                            for r in records.values()
                        ):
                            raise ValueError("Local GPU provenance mismatch")
                        write_json(
                            OUTPUT
                            / embedding
                            / "metadata"
                            / name
                            / f"{repo}.execution.json",
                            execution,
                        )
                    if records.keys() != queries.keys():
                        raise ValueError("Incomplete result coverage")
                pair_cache, cache_fingerprint = (
                    load_pair_cache(args, embedding, repo, pool, config)
                    if name == "ettin-150m"
                    else ({}, None)
                )
                ranking, labels, rows = {}, {}, []
                for q in queries.values():
                    positive = qrels[q["source_id"]]
                    record = records[q["id"]] if records else None
                    if record:
                        checked_order(record)
                        if "cached_candidate_indices" in record:
                            cached = record["cached_candidate_indices"]
                            if (
                                sorted(set(cached)) != cached
                                or any(not 0 <= i < 100 for i in cached)
                                or record["cache_source_run_sha256"]
                                != cache_fingerprint
                            ):
                                raise ValueError("Invalid cache provenance")
                            for i in cached:
                                if (
                                    record["scores"][i]
                                    != pair_cache[q["id"]][q["hybrid"][i]]
                                ):
                                    raise ValueError(
                                        "Cached score differs from baseline pair score"
                                    )
                    order = (
                        [q["hybrid"][i] for i in record["order"]]
                        if record
                        else q["dense"]
                    )
                    row = {
                        "query_id": q["id"],
                        "group": repo,
                        "ranking": order,
                        "metrics": single_positive_metrics(order, positive),
                    }
                    if record:
                        row.update(record)
                    rows.append(row)
                    labels[q["id"]] = {positive: 1}
                    ranking[q["id"]] = {
                        doc: float(100 - i) for i, doc in enumerate(order)
                    }
                measured = pytrec_eval.RelevanceEvaluator(
                    labels,
                    {
                        "ndcg_cut_10",
                        "recip_rank",
                        "recall_1",
                        "recall_5",
                        "recall_10",
                        "recall_100",
                    },
                ).evaluate(ranking)
                for row in rows:
                    for key, trec in (
                        ("ndcg_at_10", "ndcg_cut_10"),
                        ("mrr", "recip_rank"),
                        *((f"recall_at_{k}", f"recall_{k}") for k in (1, 5, 10, 100)),
                    ):
                        if (
                            abs(row["metrics"][key] - measured[row["query_id"]][trec])
                            > 1e-10
                        ):
                            raise ValueError("Independent metric disagreement")
                        checks += 1
                paths[name].extend(rows)
        for name, rows in paths.items():
            key = embedding + "/" + name
            summary = aggregate(rows)
            if summary["queries"] != 9600 or summary["repositories"] != 48:
                raise ValueError("Incomplete experiment")
            summary["candidate_pools_sha256"] = pool_hashes
            summary["source_sha256"] = digest(source)
            usage = collections.Counter()
            for row in rows:
                usage.update(row.get("usage", {}))
            summary["successful_usage"] = dict(usage)
            if name == "ettin-150m":
                summary["cached_pairs"] = sum(
                    len(r.get("cached_candidate_indices", [])) for r in rows
                )
                summary["computed_pairs"] = len(rows) * 100 - summary["cached_pairs"]
            summary["usd_successful_requests"] = (
                usage.get("input_tokens", 0) * 0.042 / 1e6
                if name == "jev-listwise"
                else 0
            )
            summaries[key] = summary
            all_rows[key] = {r["query_id"]: r for r in rows}
            write_json(OUTPUT / embedding / f"{name}.json", summary)
            write_json(OUTPUT / embedding / f"{name}.records.json.gz", rows)
        for repo in source["repositories"]:
            pool = read_json(args.work / embedding / "pools" / f"{repo}.json.gz")
            write_json(OUTPUT / embedding / "pools" / f"{repo}.json.gz", pool)
    pairs = []
    for name in PATHS:
        for i, a in enumerate(MODEL_NAMES):
            for b in MODEL_NAMES[i + 1 :]:
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "path": name,
                        **{
                            metric: paired_interval(
                                all_rows[a + "/" + name],
                                all_rows[b + "/" + name],
                                metric,
                            )
                            for metric in ("ndcg_at_10", "mrr", "recall_at_100")
                        },
                    }
                )
    write_json(OUTPUT / "source.json", source)
    write_json(
        OUTPUT / "summary.json",
        {
            "results": summaries,
            "independent_metric_checks": checks,
            "paired_comparisons": pairs,
            "uncertainty": "5000 paired repository bootstrap samples, seed20261003; 95% percentile intervals; unadjusted for multiple comparisons; fixed balanced query sample",
        },
    )
    print(f"EXPORTED 12 combinations; {checks} independent metric checks", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "rerank", "export"))
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--reranker", choices=PATHS[1:])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--embeddings", nargs="+", choices=MODEL_NAMES)
    args = parser.parse_args()
    if args.action == "rerank" and not args.reranker:
        parser.error("--reranker required")
    {"freeze": freeze, "rerank": rerank, "export": export}[args.action](args)


if __name__ == "__main__":
    main()
