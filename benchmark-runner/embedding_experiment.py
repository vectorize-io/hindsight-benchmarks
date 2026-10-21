"""LoCoMo embedding comparison; reuse audited BGE labels and reranker adapters."""

import argparse
import asyncio
import collections
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from rerank_eval import (
    DATA,
    ROOT,
    Remote,
    digest,
    load_records,
    metrics,
    rank_scores,
    read_json,
    summarize,
    validate_fixture,
    versions,
    write_json,
)

EXPERIMENT = ROOT.parent / "results/experiments/embedding-reranker-locomo"
MODELS = {
    "voyage": {
        "model": "voyageai/voyage-4-nano",
        "revision": "67fabc9bef010dabc5f6024aa1b1b6b93410426f",
        "trust_remote_code": True,
        "query_prompt": "Represent the query for retrieving supporting documents: ",
        "document_prompt": "Represent the document for retrieval: ",
    },
    "bge-small": {
        "model": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "trust_remote_code": False,
        "query_prompt": "",
        "document_prompt": "",
    },
    "leaf": {
        "model": "MongoDB/mdbr-leaf-ir",
        "revision": "4262131b32c3182bd06e67e92ae69d7bd66e0c5c",
        "trust_remote_code": False,
        "query_prompt": "Represent this sentence for searching relevant passages: ",
        "document_prompt": "",
    },
    "granite": {
        "model": "ibm-granite/granite-embedding-small-english-r2",
        "revision": "2ab6fa8ea2d674564defd37171ae19079b864b33",
        "trust_remote_code": False,
        "query_prompt": "",
        "document_prompt": "",
    },
}
MODELS["voyage-1024"] = {**MODELS["voyage"], "truncate_dim": 1024}


def encode(model, texts, config, kind, batch_size):
    # Explicit prompt avoids depending on a model's default_prompt_name.
    return model.encode(
        texts,
        prompt=config[kind + "_prompt"],
        normalize_embeddings=True,
        **(
            {"truncate_dim": config["truncate_dim"]} if "truncate_dim" in config else {}
        ),
        batch_size=batch_size,
        show_progress_bar=True,
    )


def fused_candidates(ids, lexical, dense, count):
    import numpy as np

    ranks = []
    for scores in (lexical, dense):
        order = np.argsort(-scores, kind="stable")
        rank = np.empty(len(ids), dtype=int)
        rank[order] = np.arange(1, len(ids) + 1)
        ranks.append(rank)
    scores = 1 / (60 + ranks[0]) + 1 / (60 + ranks[1])
    return [ids[i] for i in np.argsort(-scores, kind="stable")[:count]]


def prepare(args):
    import re
    import torch
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    config = MODELS[args.embedding]
    output = EXPERIMENT / args.embedding / "fixture.json.gz"
    if output.exists():
        raise ValueError("Fixture exists; reuse it rather than overwriting")
    fixture = read_json(DATA / "locomo10-top100.json.gz")
    validate_fixture(fixture)
    baseline_hash = fixture.pop("fixture_sha256")
    if config["model"] == "voyageai/voyage-4-nano":
        # This official snapshot omits config_class; Transformers 5 requires it
        # during AutoModel registration. No forward-pass or weight change.
        from transformers import Qwen3Config
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        model_class = get_class_from_dynamic_module(
            "modeling_qwen3_bidirectional.Qwen3BidirectionalModel",
            config["model"],
            revision=config["revision"],
        )
        model_class.config_class = Qwen3Config
    model = SentenceTransformer(
        config["model"],
        revision=config["revision"],
        device="cuda:0",
        trust_remote_code=config["trust_remote_code"],
        model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
    )
    hardware = torch.cuda.get_device_properties(0)
    fixture["retriever"] = {
        **config,
        "max_seq_length": model.max_seq_length,
        "dimension": config.get(
            "truncate_dim", model.get_sentence_embedding_dimension()
        ),
        "precision": "float32",
        "registration_compatibility": "Explicit Qwen3Config config_class for Transformers 5; forward unchanged"
        if config["model"] == "voyageai/voyage-4-nano"
        else None,
        "attention": "sdpa",
        "batch_size": args.batch_size,
        "fusion": "1/(60+BM25_rank) + 1/(60+dense_rank); ranks start at 1; all corpus documents participate",
        "candidate_count": 100,
        "baseline_fixture_sha256": baseline_hash,
        "hardware": hardware.name,
        "versions": versions(),
        "preparation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    groups = collections.defaultdict(list)
    for q in fixture["queries"]:
        groups[q["group"]].append(q)
    for group, queries in groups.items():
        # LoCoMo corpus IDs are prefixed with their conversation group.
        ids = sorted({d for d in fixture["corpus"] if d.startswith(group + "/")})
        if not ids:
            raise ValueError(f"No source corpus for {group}")
        texts = [fixture["corpus"][d] for d in ids]
        bm25 = BM25Okapi([re.findall(r"\w+", t.lower()) for t in texts])
        docs = encode(model, texts, config, "document", args.batch_size)
        vectors = encode(
            model, [q["query"] for q in queries], config, "query", args.batch_size
        )
        for q, vector in zip(queries, vectors):
            lexical = bm25.get_scores(re.findall(r"\w+", q["query"].lower()))
            q["candidates"] = fused_candidates(ids, lexical, docs @ vector, 100)
        print(
            f"Prepared {args.embedding} {group}: {len(ids)} documents, {len(queries)} queries",
            flush=True,
        )
    fixture["fixture_sha256"] = digest(fixture)
    validate_fixture(fixture)
    write_json(output, fixture)
    print(f"Frozen {output}: {fixture['fixture_sha256']}", flush=True)


async def concurrent_queries(fixture, remotes, records, fingerprint, append):
    """Each worker owns its transport counters; the event loop owns append ordering."""
    queue = asyncio.Queue()
    for q in fixture["queries"]:
        if q["id"] not in records:
            queue.put_nowait(q)

    async def worker(remote):
        while not queue.empty():
            q = queue.get_nowait()
            docs = [fixture["corpus"][d] for d in q["candidates"]]
            started = time.perf_counter()
            before = remote.retries
            scores, usage, scoring = await remote.score(q["query"], docs)
            order = rank_scores(scores, len(docs))
            row = {
                "run_sha256": fingerprint,
                "query_id": q["id"],
                "group": q["group"],
                "category": q["category"],
                "scores": scores,
                "order": order,
                "latency_s": time.perf_counter() - started,
                "scoring_latency_s": scoring,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "retries": remote.retries - before,
                "usage": usage,
                "metrics": metrics([q["candidates"][i] for i in order], q["relevant"]),
            }
            if q["id"] in records:
                raise ValueError("Duplicate completed query")
            append(row)
            records[q["id"]] = row
            if len(records) % 100 == 0:
                print(f"Completed {len(records)}/{len(fixture['queries'])}", flush=True)

    async with asyncio.TaskGroup() as tasks:
        for remote in remotes:
            tasks.create_task(worker(remote))


async def run_hosted(args):
    fixture = read_json(EXPERIMENT / args.embedding / "fixture.json.gz")
    validate_fixture(fixture)
    config = next(
        c
        for c in read_json(ROOT / "reranker_models.json")["rerankers"]
        if c["reranker_id"] == "jev-listwise"
    )
    metadata = {
        "config": config,
        "fixture_sha256": fixture["fixture_sha256"],
        "adapter_sha256": hashlib.sha256(
            (ROOT / "rerank_eval.py").read_bytes()
        ).hexdigest(),
        "experiment_adapter_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "listwise_adapter_sha256": hashlib.sha256(
            (ROOT / "jev_listwise.py").read_bytes()
        ).hexdigest(),
        "versions": versions(),
        "query_workers": args.workers,
        "requests_per_second_per_worker": 10,
        "concurrency_per_worker": 8,
        "latency_conditions": "Concurrent hosted queries and GPU jobs; worker queue wait excluded; incomparable to sequential baseline latency",
    }
    fingerprint = digest(metadata)
    destination = args.records / "locomo/jev-listwise.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(destination, fingerprint)
    write_json(destination.with_suffix(".metadata.json"), metadata)
    remote_args = SimpleNamespace(
        secrets_dir=args.secrets_dir, concurrency=8, requests_per_second=10
    )
    remotes = [Remote(config, remote_args) for _ in range(args.workers)]
    try:
        # Separate warmup, not included in measured query usage.
        await asyncio.gather(
            *(
                r.score(
                    "Which planet is red?", ["Mars is red.", "Jupiter is a gas giant."]
                )
                for r in remotes
            )
        )
        with destination.open("a") as f:

            def append(row):
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()

            await concurrent_queries(fixture, remotes, records, fingerprint, append)
    except Exception:
        write_json(
            destination.with_suffix(".failure.json"),
            {
                "completed_queries": len(records),
                "run_sha256": fingerprint,
                "message": "Concurrent run interrupted; no partial score published; failed-attempt billing unknown",
            },
        )
        raise
    finally:
        await asyncio.gather(*(r.close() for r in remotes))
    print(f"Complete: {len(records)} queries", flush=True)


def export(args):
    import pytrec_eval
    from summarize_rerankers import paired_interval

    all_rows = {}
    comparisons = 0
    output = EXPERIMENT / "results"
    for embedding in ["bge", "voyage", "granite", "bge-small", "voyage-1024", "leaf"]:
        fixture_path = (
            DATA / "locomo10-top100.json.gz"
            if embedding == "bge"
            else EXPERIMENT / embedding / "fixture.json.gz"
        )
        fixture = read_json(fixture_path)
        validate_fixture(fixture)
        queries = {q["id"]: q for q in fixture["queries"]}
        qrels = {qid: q["relevant"] for qid, q in queries.items()}
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {"ndcg_cut_10", "recip_rank", "recall_5", "recall_10"}
        )
        for model in ["jev-listwise", "ettin-150m"]:
            folder = output / embedding
            if embedding == "bge":
                source = ROOT.parent / "results/leaderboard/reranker"
                result = read_json(source / f"{model}--locomo.json")
                rows = read_json(source / f"records/{model}--locomo.json.gz")
                write_json(folder / f"{model}--locomo.json", result)
                write_json(folder / f"records/{model}--locomo.json.gz", rows)
            else:
                summarize(
                    SimpleNamespace(
                        fixture=fixture_path,
                        model=model,
                        records=args.records / embedding,
                        output=folder,
                        labels=None,
                    )
                )
                result = read_json(folder / f"{model}--locomo.json")
                rows = read_json(folder / f"records/{model}--locomo.json.gz")
            if result["fixture_sha256"] != fixture["fixture_sha256"] or result[
                "run_sha256"
            ] != digest(result["metadata"]):
                raise ValueError("Run provenance mismatch")
            by_id = {r["query_id"]: r for r in rows}
            if len(by_id) != len(rows) or by_id.keys() != queries.keys():
                raise ValueError("Incomplete query coverage")
            rankings = {}
            for qid, row in by_id.items():
                q = queries[qid]
                if (
                    row["run_sha256"] != result["run_sha256"]
                    or sorted(row["order"]) != list(range(100))
                    or row["order"] != rank_scores(row["scores"], 100)
                ):
                    raise ValueError("Invalid ranking or provenance")
                rankings[qid] = {
                    q["candidates"][index]: float(100 - pos)
                    for pos, index in enumerate(row["order"])
                }
            measured = evaluator.evaluate(rankings)
            for internal, trec in [
                ("ndcg_at_10", "ndcg_cut_10"),
                ("mrr", "recip_rank"),
                ("recall_at_5", "recall_5"),
                ("recall_at_10", "recall_10"),
            ]:
                for qid, row in by_id.items():
                    if abs(row["metrics"][internal] - measured[qid][trec]) > 1e-10:
                        raise ValueError("Independent metric mismatch")
                    comparisons += 1
                if (
                    abs(
                        result[internal]
                        - sum(r[trec] for r in measured.values()) / len(rows)
                    )
                    > 1e-10
                ):
                    raise ValueError("Aggregate mismatch")
                comparisons += 1
            all_rows[f"{embedding}/{model}"] = by_id
    pairs = []
    keys = list(all_rows)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "ndcg_at_10": paired_interval(
                        all_rows[a], all_rows[b], "ndcg_at_10"
                    ),
                }
            )
    write_json(
        EXPERIMENT / "validation.json",
        {
            "independent_trec_comparisons": comparisons,
            "complete_results": len(all_rows),
            "queries_per_result": 1533,
        },
    )
    write_json(
        EXPERIMENT / "paired-comparisons.json",
        {
            "method": "5000 paired whole-conversation bootstrap samples; 95% percentile intervals, unadjusted",
            "pairs": pairs,
        },
    )
    print(f"Validated {comparisons} independent metric comparisons", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "hosted", "export"])
    parser.add_argument("--embedding", choices=list(MODELS))
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("/mnt/optane/hindsight-reranker-work/embedding-runs"),
    )
    parser.add_argument("--secrets-dir", type=Path, default=Path.home() / ".secrets")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.command != "export" and not args.embedding:
        parser.error("--embedding is required")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "hosted":
        if not 1 <= args.workers <= 4:
            parser.error("Use 1 to 4 workers to bound provider request rate")
        asyncio.run(run_hosted(args))
    else:
        export(args)


if __name__ == "__main__":
    main()
