"""Frozen-candidate reranking with source labels and auditable, resumable records."""

from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import tarfile
import time
import zipfile

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "results" / "leaderboard" / "reranker"
DATA = ROOT / "datasets" / "reranker"
RETRIEVER = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
JEV_PROMPT = {
    "type": "noul",
    "instructions": "Does the candidate contain information useful for answering or verifying the query?",
    "criteria": {
        "true": "The candidate contains specific evidence useful for answering or verifying the query, including evidence that contradicts the query. Partial evidence for a question requiring several facts also counts.",
        "false": "The candidate is unrelated, or merely discusses the same broad topic without evidence useful for answering or verifying the query.",
    },
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".gz":
        # Stable gzip envelope: file name and wall clock must not affect checksums.
        with (
            temp.open("wb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as f,
        ):
            f.write(json.dumps(data, ensure_ascii=False, sort_keys=True).encode())
    else:
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def read_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def evidence_ids(evidence):
    """Normalize syntax only; never guess the intended location of a missing ID."""
    result = []
    for entry in evidence:
        for part in re.split(r"[;\s]+", entry.strip()):
            if not part:
                continue
            match = re.fullmatch(r"D:?(\d+):(\d+)", part)
            if not match:
                raise ValueError(f"Malformed evidence reference: {part}")
            result.append(f"D{int(match[1])}:{int(match[2])}")
    return list(dict.fromkeys(result))


def locomo_source(path):
    conversations = read_json(path)
    corpora, queries, excluded, corrections = {}, [], [], []
    for item in conversations:
        group = item["sample_id"]
        corpus = {}
        for key, turns in item["conversation"].items():
            if not re.fullmatch(r"session_\d+", key):
                continue
            date = item["conversation"][key + "_date_time"]
            for turn in turns:
                text = f"{turn['speaker']} on {date}: {turn['text']}"
                if turn.get("blip_caption"):
                    text += f" [Image caption: {turn['blip_caption']}]"
                if turn.get("query"):
                    text += f" [Image search description: {turn['query']}]"
                doc_id = group + "/" + turn["dia_id"]
                if doc_id in corpus:
                    raise ValueError(f"Duplicate turn ID: {doc_id}")
                corpus[doc_id] = text
        corpora[group] = corpus
        for i, qa in enumerate(item["qa"]):
            qid = f"{group}/q{i}"
            reason = None
            if qa["category"] == 5:
                reason = "adversarial"
            elif not qa.get("evidence"):
                reason = "no evidence annotations"
            else:
                try:
                    refs = evidence_ids(qa["evidence"])
                    if refs != qa["evidence"]:
                        corrections.append(
                            {
                                "query_id": qid,
                                "original": qa["evidence"],
                                "normalized": refs,
                            }
                        )
                    relevant = {group + "/" + ref: 1 for ref in refs}
                    if not relevant.keys() <= corpus.keys():
                        reason = "nonexistent evidence IDs: " + ", ".join(
                            sorted(relevant.keys() - corpus.keys())
                        )
                except ValueError as e:
                    reason = str(e)
            if reason:
                excluded.append(
                    {
                        "query_id": qid,
                        "reason": reason,
                        "evidence": qa.get("evidence", []),
                    }
                )
            else:
                queries.append(
                    {
                        "id": qid,
                        "group": group,
                        "category": qa["category"],
                        "query": qa["question"],
                        "relevant": relevant,
                    }
                )
    return corpora, queries, excluded, corrections


def scifact_source(path):
    with zipfile.ZipFile(path) as archive:
        corpus = {}
        for line in archive.read("scifact/corpus.jsonl").decode().splitlines():
            doc = json.loads(line)
            corpus[doc["_id"]] = (doc.get("title", "") + "\n" + doc["text"]).strip()
        questions = {
            q["_id"]: q["text"]
            for q in map(
                json.loads, archive.read("scifact/queries.jsonl").decode().splitlines()
            )
        }
        labels = collections.defaultdict(dict)
        for line in archive.read("scifact/qrels/test.tsv").decode().splitlines()[1:]:
            qid, docid, grade = line.split("\t")
            if int(grade) > 0:
                if docid not in corpus:
                    raise ValueError(f"Missing SciFact document: {docid}")
                labels[qid][docid] = int(grade)
    queries = [
        {
            "id": qid,
            "group": qid,
            "category": "fact-checking",
            "query": questions[qid],
            "relevant": labels[qid],
        }
        for qid in sorted(labels)
    ]
    return {"scifact": corpus}, queries, [], []


def prepare_evidence_labels(args):
    if args.output.exists():
        raise ValueError("Evidence labels already exist; choose a new output")
    fixture = read_json(args.fixture)
    validate_fixture(fixture)
    if fixture["dataset"] != "scifact":
        raise ValueError("Evidence labels require the SciFact fixture")
    with tarfile.open(args.source) as archive:
        original = {
            str(row["id"]): row
            for row in map(
                json.loads,
                archive.extractfile("data/claims_dev.jsonl")
                .read()
                .decode()
                .splitlines(),
            )
        }
        original_corpus = {
            str(row["doc_id"]): (
                row["title"] + "\n" + " ".join(row["abstract"])
            ).strip()
            for row in map(
                json.loads,
                archive.extractfile("data/corpus.jsonl").read().decode().splitlines(),
            )
        }
    if original_corpus.keys() != fixture["corpus"].keys() or any(
        " ".join(text.split()) != " ".join(fixture["corpus"][doc].split())
        for doc, text in original_corpus.items()
    ):
        raise ValueError("Original source abstracts do not match the scored corpus")
    queries = {q["id"]: q for q in fixture["queries"]}
    if original.keys() != queries.keys() or any(
        row["claim"] != queries[qid]["query"]
        or set(map(str, row["cited_doc_ids"])) != queries[qid]["relevant"].keys()
        for qid, row in original.items()
    ):
        raise ValueError("Original development claims do not match the BEIR fixture")
    if any(
        evidence["label"] not in {"SUPPORT", "CONTRADICT"}
        for row in original.values()
        for rationales in row["evidence"].values()
        for evidence in rationales
    ):
        raise ValueError("Unexpected original evidence label")
    labels = {
        qid: {str(doc): 1 for doc in row["evidence"]}
        for qid, row in original.items()
        if row["evidence"]
    }
    if any(not gold.keys() <= fixture["corpus"].keys() for gold in labels.values()):
        raise ValueError("Evidence document missing from scored corpus")
    view = {
        "schema_version": 1,
        "name": "scifact-evidence",
        "source": "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "source_split": "original SciFact claims_dev.jsonl",
        "labels": labels,
        "excluded_queries": {
            qid: "No explicit evidence annotation"
            for qid, row in original.items()
            if not row["evidence"]
        },
        "rule": "Only documents explicitly annotated SUPPORT or CONTRADICT count as relevant; questions without either are excluded uniformly before comparing models.",
        "query_sha256": {
            qid: hashlib.sha256(original[qid]["claim"].encode()).hexdigest()
            for qid in labels
        },
    }
    view["label_sha256"] = digest(view)
    write_json(args.output, view)
    print(
        f"Prepared explicit evidence labels: {len(labels)} queries; {view['label_sha256']}"
    )


def prepare(args):
    import numpy as np
    from huggingface_hub import HfApi
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    if args.output.exists():
        raise ValueError(
            "Fixture already exists; choose a new output instead of replacing published inputs"
        )
    corpora, queries, exclusions, corrections = (
        locomo_source if args.dataset == "locomo" else scifact_source
    )(args.source)
    revision = HfApi().model_info(RETRIEVER).sha
    model = SentenceTransformer(RETRIEVER, revision=revision, device=args.device)
    fixture = {
        "schema_version": 2,
        "dataset": args.dataset,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "retriever": {
            "model": RETRIEVER,
            "revision": revision,
            "query_prefix": QUERY_PREFIX,
            "max_seq_length": model.max_seq_length,
            "fusion": "1/(60+BM25_rank) + 1/(60+dense_rank); ranks start at 1; all corpus documents participate",
            "candidate_count": args.candidates,
        },
        "corpus": {},
        "queries": queries,
        "exclusions": exclusions,
        "corrections": corrections,
    }
    for group, corpus in corpora.items():
        ids = sorted(corpus)
        texts = [corpus[i] for i in ids]
        group_queries = [
            q for q in queries if args.dataset == "scifact" or q["group"] == group
        ]
        bm25 = BM25Okapi([re.findall(r"\w+", s.lower()) for s in texts])
        document_vectors = model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
        )
        query_vectors = model.encode(
            [QUERY_PREFIX + q["query"] for q in group_queries],
            normalize_embeddings=True,
            batch_size=64,
        )
        for q, vector in zip(group_queries, query_vectors):
            lexical = bm25.get_scores(re.findall(r"\w+", q["query"].lower()))
            dense = document_vectors @ vector
            ranks = []
            for scores in (lexical, dense):
                order = np.argsort(-scores, kind="stable")
                rank = np.empty(len(ids), dtype=int)
                rank[order] = np.arange(1, len(ids) + 1)
                ranks.append(rank)
            scores = 1 / (60 + ranks[0]) + 1 / (60 + ranks[1])
            q["candidates"] = [
                ids[i] for i in np.argsort(-scores, kind="stable")[: args.candidates]
            ]
        fixture["corpus"].update(corpus)
    fixture["fixture_sha256"] = digest(fixture)
    write_json(args.output, fixture)
    print(
        f"Prepared {args.dataset}: {len(queries)} queries, {len(corpora)} groups, {len(exclusions)} exclusions; {fixture['fixture_sha256']}",
        flush=True,
    )


def validate_fixture(fixture):
    expected = fixture["fixture_sha256"]
    if digest({k: v for k, v in fixture.items() if k != "fixture_sha256"}) != expected:
        raise ValueError("Fixture hash mismatch")
    if len({q["id"] for q in fixture["queries"]}) != len(fixture["queries"]):
        raise ValueError("Duplicate query IDs")
    for q in fixture["queries"]:
        if not q["relevant"] or not q["relevant"].keys() <= fixture["corpus"].keys():
            raise ValueError("Invalid relevance labels")
        if (
            len(q["candidates"]) != len(set(q["candidates"]))
            or not set(q["candidates"]) <= fixture["corpus"].keys()
        ):
            raise ValueError("Invalid candidate IDs")


def rank_scores(scores, count):
    if len(scores) != count or not all(
        isinstance(s, (float, int)) and math.isfinite(s) for s in scores
    ):
        raise ValueError("Provider did not return one finite score per candidate")
    return sorted(range(count), key=lambda i: (-scores[i], i))


def mapped_scores(response, count):
    scores = [None] * count
    for row in response:
        index = row["index"]
        if (
            not isinstance(index, int)
            or index < 0
            or index >= count
            or scores[index] is not None
        ):
            raise ValueError("Invalid or duplicate provider candidate index")
        scores[index] = row["relevance_score"]
    rank_scores(scores, count)
    return scores


class Remote:
    def __init__(self, config, args):
        import httpx

        self.config = config
        self.key = os.environ.get(config["key_env"], "")
        if not self.key and args.secrets_dir:
            self.key = (args.secrets_dir / config["key_file"]).read_text().strip()
        if not self.key:
            raise ValueError(f"Missing {config['key_env']}")
        self.client = httpx.AsyncClient(
            timeout=60,
            limits=httpx.Limits(
                max_connections=args.concurrency,
                max_keepalive_connections=args.concurrency,
            ),
            headers={"Authorization": "Bearer " + self.key},
        )
        self.semaphore = asyncio.Semaphore(args.concurrency)
        self.rate_lock = asyncio.Lock()
        self.next_request = 0.0
        self.rps = args.requests_per_second
        self.attempt_usage = collections.Counter()
        self.retries = 0

    async def post(self, body):
        import httpx

        for attempt in range(6):
            async with self.semaphore:
                async with self.rate_lock:
                    await asyncio.sleep(max(0, self.next_request - time.monotonic()))
                    self.next_request = time.monotonic() + 1 / self.rps
                started = time.perf_counter()
                try:
                    response = await self.client.post(
                        self.config["endpoint"], json=body
                    )
                except (httpx.TimeoutException, httpx.NetworkError):
                    if attempt == 5:
                        raise RuntimeError(
                            "Provider transport failure after six attempts"
                        ) from None
                    self.retries += 1
                    response = None
                duration = time.perf_counter() - started
            if response is None:
                await asyncio.sleep(2**attempt)
                continue
            if (
                response.status_code == 429 or 500 <= response.status_code < 600
            ) and attempt < 5:
                self.retries += 1
                try:
                    delay = float(response.headers.get("retry-after", 2**attempt))
                except ValueError:
                    delay = 2**attempt
                await asyncio.sleep(min(60, max(delay, 0.5)))
                continue
            if response.status_code >= 400:
                # The body and headers may reflect credentials; never persist them.
                raise RuntimeError(f"Provider HTTP {response.status_code}")
            return response.json(), duration
        raise RuntimeError("Retry budget exhausted")

    async def score(self, query, docs):
        provider = self.config["provider"]
        usage = collections.Counter()
        if (
            provider == "typesafe"
            and self.config.get("ranking_mode") == "hindsight-choice"
        ):
            from jev_listwise import HindsightChoice

            return await HindsightChoice(self).score(query, docs)
        if provider == "typesafe":

            async def pair(doc):
                data, duration = await self.post(
                    {
                        "model": self.config["model"],
                        "state": {"query": query, "candidate": doc},
                        "questions": {"relevance": JEV_PROMPT},
                    }
                )
                if data["model"] != self.config["model"]:
                    raise ValueError("Jev response model differs from pinned model")
                return (
                    data["answers"]["relevance"]["noul"],
                    data.get("usage", {}),
                    duration,
                )

            pairs = await asyncio.gather(*(pair(doc) for doc in docs))
            scores = [p[0] for p in pairs]
            for _, units, _ in pairs:
                usage.update(units)
            successful_duration = max(p[2] for p in pairs)
        else:
            body = {"model": self.config["model"], "query": query, "documents": docs}
            if provider == "voyage":
                body["truncation"] = False
            data, successful_duration = await self.post(body)
            scores = mapped_scores(
                data["data"] if provider == "voyage" else data["results"], len(docs)
            )
            usage.update(data.get("usage", {}))
            usage.update(data.get("meta", {}).get("billed_units", {}))
        rank_scores(scores, len(docs))
        return scores, dict(usage), successful_duration

    async def close(self):
        await self.client.aclose()


def metrics(order, relevant):
    if len(order) != len(set(order)):
        raise ValueError("Ranking has duplicate document IDs")
    hits = [i + 1 for i, doc in enumerate(order) if doc in relevant]
    result = {"mrr": 1 / hits[0] if hits else 0.0}
    ideal = sorted(relevant.values(), reverse=True)[:10]
    idcg = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(ideal))
    dcg = sum(
        (2 ** relevant.get(doc, 0) - 1) / math.log2(i + 2)
        for i, doc in enumerate(order[:10])
    )
    result["ndcg_at_10"] = dcg / idcg
    for k in (1, 3, 5, 10):
        found = len(set(order[:k]) & relevant.keys())
        result[f"recall_at_{k}"] = found / len(relevant)
        result[f"hit_at_{k}"] = float(found > 0)
    return result


def load_records(path, fingerprint):
    records = {}
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["run_sha256"] != fingerprint:
                raise ValueError(
                    "Resume refused: fixture, adapter, or configuration changed"
                )
            if row["query_id"] in records:
                raise ValueError("Duplicate query record")
            records[row["query_id"]] = row
    return records


def versions():
    names = (
        "sentence-transformers",
        "torch",
        "transformers",
        "numpy",
        "httpx",
        "rank-bm25",
    )
    return {name: importlib.metadata.version(name) for name in names}


async def run(args):
    fixture = read_json(args.fixture)
    validate_fixture(fixture)
    configs = {
        c["reranker_id"]: c
        for c in read_json(ROOT / "reranker_models.json")["rerankers"]
    }
    config = configs[args.model]
    local = config["provider"] == "local"
    encoder, remote, device_info = None, None, None
    metadata = {
        "config": config,
        "fixture_sha256": fixture["fixture_sha256"],
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": versions(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "max_length": config.get("max_length"),
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "requests_per_second": args.requests_per_second,
        "jev_question": JEV_PROMPT
        if config["provider"] == "typesafe" and not config.get("ranking_mode")
        else None,
        "listwise_adapter_sha256": hashlib.sha256(
            (ROOT / "jev_listwise.py").read_bytes()
        ).hexdigest()
        if config.get("ranking_mode")
        else None,
        "listwise_tokenizer": "toktok-rs==0.1.3/o200k_base"
        if config.get("ranking_mode")
        else None,
    }
    if local:
        import torch
        from huggingface_hub import HfApi
        from sentence_transformers import CrossEncoder

        metadata["model_revision"] = HfApi().model_info(config["model"]).sha
        metadata["precision"] = "bfloat16"
        metadata["score_activation"] = "identity (raw model score)"
        device_info = torch.cuda.get_device_properties(args.device)
        metadata["hardware"] = {
            "name": device_info.name,
            "memory_bytes": device_info.total_memory,
            "cuda_version": torch.version.cuda,
            "compute_capability": [device_info.major, device_info.minor],
        }
        encoder = CrossEncoder(
            config["model"],
            revision=metadata["model_revision"],
            device=args.device,
            max_length=config["max_length"],
            model_kwargs={"dtype": torch.bfloat16},
        )
        for _ in range(3):
            encoder.predict(
                [("Which planet is red?", "Mars is the red planet.")] * 16,
                batch_size=args.batch_size,
                activation_fn=torch.nn.Identity(),
            )
        torch.cuda.synchronize(args.device)
    elif config["provider"] != "rrf":
        remote = Remote(config, args)
        await remote.score(
            "Which planet is red?",
            ["Mars is the red planet.", "Jupiter is a gas giant."],
        )
    fingerprint = digest(metadata)
    destination = args.output / fixture["dataset"] / (args.model + ".jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(destination, fingerprint)
    write_json(destination.with_suffix(".metadata.json"), metadata)
    queries = fixture["queries"]
    selected = queries[: args.limit] if args.limit else queries
    started_at = time.time()
    try:
        for i, q in enumerate(selected):
            if q["id"] in records:
                continue
            docs = [fixture["corpus"][doc] for doc in q["candidates"]]
            started = time.perf_counter()
            attempts_before = remote.retries if remote else 0
            if local:
                import torch

                pairs = [(q["query"], doc) for doc in docs]
                lengths = [
                    len(tokens)
                    for tokens in encoder.tokenizer(
                        [p[0] for p in pairs], [p[1] for p in pairs], truncation=False
                    )["input_ids"]
                ]
                torch.cuda.synchronize(args.device)
                scoring_start = time.perf_counter()
                scores = encoder.predict(
                    pairs,
                    batch_size=args.batch_size,
                    show_progress_bar=False,
                    activation_fn=torch.nn.Identity(),
                ).tolist()
                torch.cuda.synchronize(args.device)
                successful_duration = time.perf_counter() - scoring_start
                usage = {
                    "truncated_pairs": sum(
                        length > config["max_length"] for length in lengths
                    )
                }
            elif remote:
                scores, usage, successful_duration = await remote.score(
                    q["query"], docs
                )
            else:
                scores, usage, successful_duration = (
                    [1 / (i + 1) for i in range(len(docs))],
                    {},
                    0.0,
                )
            duration = time.perf_counter() - started
            order = rank_scores(scores, len(docs))
            row = {
                "run_sha256": fingerprint,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "query_id": q["id"],
                "group": q["group"],
                "category": q["category"],
                "scores": scores,
                "order": order,
                "latency_s": duration,
                "scoring_latency_s": successful_duration,
                "retries": remote.retries - attempts_before if remote else 0,
                "usage": usage,
                "metrics": metrics([q["candidates"][j] for j in order], q["relevant"]),
            }
            with destination.open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
            records[q["id"]] = row
            if (i + 1) % 25 == 0 or i == 0:
                print(
                    f"{args.model} {fixture['dataset']} {i + 1}/{len(selected)} ({time.time() - started_at:.1f}s elapsed)",
                    flush=True,
                )
    except Exception as e:
        # No failed request is turned into an irrelevant document or a published score.
        write_json(
            destination.with_suffix(".failure.json"),
            {
                "query_id": q["id"],
                "error_type": type(e).__name__,
                "message": str(e)
                if isinstance(e, (ValueError, RuntimeError))
                else "Request or model failure; inspect locally",
                "completed_queries": len(records),
                "run_sha256": fingerprint,
            },
        )
        raise
    finally:
        if remote:
            await remote.close()
    write_json(destination.with_suffix(".metadata.json"), metadata)
    print(f"Saved {len(records)} records to {destination}", flush=True)


def confidence(rows, metric, dataset):
    import numpy as np

    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row["metrics"][metric])
    # Resample whole conversations, preserving the question-weighted estimand.
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(20261003)
    sample = rng.integers(0, len(groups), size=(5000, len(groups)))
    means = sums[sample].sum(axis=1) / counts[sample].sum(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize(args):
    fixture = read_json(args.fixture)
    validate_fixture(fixture)
    queries = {q["id"]: q for q in fixture["queries"]}
    source = args.records / fixture["dataset"] / (args.model + ".jsonl")
    metadata = read_json(source.with_suffix(".metadata.json"))
    if metadata["fixture_sha256"] != fixture["fixture_sha256"]:
        raise ValueError("Run metadata does not match the input fixture")
    records = load_records(source, digest(metadata))
    if records.keys() != queries.keys():
        raise ValueError(
            f"Incomplete run: {len(records)}/{len(queries)} queries; no quality summary published"
        )
    rows = []
    for qid, q in queries.items():
        row = records[qid]
        order = row["order"]
        if sorted(order) != list(range(len(q["candidates"]))):
            raise ValueError("Ranking is not a complete candidate permutation")
        if order != rank_scores(row["scores"], len(q["candidates"])):
            raise ValueError("Stored ordering does not match scores")
        computed = metrics([q["candidates"][j] for j in order], q["relevant"])
        if computed != row["metrics"]:
            raise ValueError("Stored metric mismatch")
        if row["group"] != q["group"] or row["category"] != q["category"]:
            raise ValueError("Stored query grouping mismatch")
        rows.append(row)
    dataset = fixture["dataset"]
    label_view = None
    if getattr(args, "labels", None):
        label_view = read_json(args.labels)
        if (
            digest({k: v for k, v in label_view.items() if k != "label_sha256"})
            != label_view["label_sha256"]
        ):
            raise ValueError("Evidence-label hash mismatch")
        labels = label_view["labels"]
        excluded = label_view["excluded_queries"]
        if dataset != "scifact" or label_view["name"] != "scifact-evidence":
            raise ValueError("Evidence-label view requires the SciFact fixture")
        if (
            set(labels) & excluded.keys()
            or set(labels) | excluded.keys() != queries.keys()
        ):
            raise ValueError(
                "Evidence-label view must account for every original query"
            )
        if any(
            not gold
            or not gold.keys() <= fixture["corpus"].keys()
            or any(grade != 1 for grade in gold.values())
            or label_view["query_sha256"][qid]
            != hashlib.sha256(queries[qid]["query"].encode()).hexdigest()
            for qid, gold in labels.items()
        ):
            raise ValueError("Invalid evidence labels or mismatched source query text")
        rows = [
            {
                **row,
                "source_metrics": row["metrics"],
                "evaluation_label_sha256": label_view["label_sha256"],
                "metrics": metrics(
                    [queries[row["query_id"]]["candidates"][i] for i in row["order"]],
                    labels[row["query_id"]],
                ),
            }
            for row in rows
            if row["query_id"] in labels
        ]
        queries = {
            qid: {**queries[qid], "relevant": gold} for qid, gold in labels.items()
        }
        dataset = label_view["name"]
    import numpy as np

    aggregate = {
        name: sum(r["metrics"][name] for r in rows) / len(rows)
        for name in rows[0]["metrics"]
    }
    intervals = {name: confidence(rows, name, fixture["dataset"]) for name in aggregate}
    usage = collections.Counter()
    for row in rows:
        usage.update(row["usage"])
    config = metadata["config"]
    tokens = usage.get("input_tokens", usage.get("total_tokens", 0))
    if config["pricing_type"] == "local":
        cost, cost_basis = None, "Local compute/electricity not measured"
    elif config["provider"] == "rrf":
        cost, cost_basis = 0.0, "No additional reranking"
    elif config.get("usd_per_million_tokens") is not None:
        cost, cost_basis = (
            tokens * config["usd_per_million_tokens"] / 1e6,
            "API reported tokens × published list rate; excludes credits and failed-request billing",
        )
    elif config.get("usd_per_search_unit") is not None:
        cost, cost_basis = (
            usage["search_units"] * config["usd_per_search_unit"],
            "API reported search units × published list rate; excludes credits and failed-request billing",
        )
    else:
        cost, cost_basis = None, "API usage recorded; monetary rate unverified"
    result = {
        "schema_version": 2,
        "reranker_id": args.model,
        "provider": config["provider"],
        "model": config.get("model"),
        "dataset": dataset,
        "fixture_sha256": fixture["fixture_sha256"],
        "run_sha256": digest(metadata),
        "total_questions": len(rows),
        "groups": len({r["group"] for r in rows}),
        "sample_id": "locomo10"
        if dataset == "locomo"
        else ("scifact-original-dev-evidence" if label_view else "scifact-beir-test"),
        "scored_queries": len(records),
        "evaluation_label_sha256": label_view["label_sha256"] if label_view else None,
        "label_view": {
            k: v
            for k, v in label_view.items()
            if k not in {"labels", "query_sha256", "excluded_queries"}
        }
        if label_view
        else None,
        **aggregate,
        "confidence_intervals": intervals,
        "confidence_method": "5000 bootstrap samples; whole conversations for LoCoMo, queries for SciFact; 95% percentile interval",
        "avg_latency_s": float(np.mean([r["latency_s"] for r in rows])),
        "p50_latency_s": float(np.median([r["latency_s"] for r in rows])),
        "p95_latency_s": float(np.quantile([r["latency_s"] for r in rows], 0.95)),
        "retries": sum(r["retries"] for r in rows),
        "failed_queries": 0,
        "usage": dict(usage),
        "estimated_api_cost_usd": cost,
        "cost_basis": cost_basis,
        "metadata": metadata,
        "retrieval_ceiling_recall": sum(
            len(set(q["candidates"]) & q["relevant"].keys()) / len(q["relevant"])
            for q in queries.values()
        )
        / len(queries),
    }
    result["by_category"] = {
        str(cat): {
            name: sum(r["metrics"][name] for r in rows if r["category"] == cat)
            / sum(r["category"] == cat for r in rows)
            for name in aggregate
        }
        for cat in {r["category"] for r in rows}
    }
    # SciFact groups are single queries, whose full details already live in records.
    result["by_group"] = (
        {
            group: {
                "count": sum(r["group"] == group for r in rows),
                **{
                    name: sum(r["metrics"][name] for r in rows if r["group"] == group)
                    / sum(r["group"] == group for r in rows)
                    for name in aggregate
                },
            }
            for group in sorted({r["group"] for r in rows})
        }
        if dataset == "locomo"
        else {}
    )
    write_json(args.output / (args.model + "--" + dataset + ".json"), result)
    write_json(
        args.output / "records" / (args.model + "--" + dataset + ".json.gz"),
        rows,
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "dataset": dataset,
                **aggregate,
                "p50_latency_s": result["p50_latency_s"],
                "cost": cost,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--dataset", choices=["locomo", "scifact"], required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidates", type=int, default=100)
    p.add_argument("--device", default="cuda:0")
    e = subs.add_parser("prepare-evidence-labels")
    e.add_argument("--source", type=Path, required=True)
    e.add_argument("--fixture", type=Path, default=DATA / "scifact-test-top100.json.gz")
    e.add_argument("--output", type=Path, required=True)
    r = subs.add_parser("run")
    r.add_argument("--fixture", type=Path, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--output", type=Path, default=ROOT / "reranker-runs")
    r.add_argument("--secrets-dir", type=Path)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--concurrency", type=int, default=64)
    r.add_argument("--requests-per-second", type=float, default=60)
    r.add_argument(
        "--limit", type=int, help="Smoke test only; partial runs cannot be published"
    )
    s = subs.add_parser("summarize")
    s.add_argument("--fixture", type=Path, required=True)
    s.add_argument(
        "--labels",
        type=Path,
        help="Optional independently sourced evidence-label view; scoring inputs remain the original fixture",
    )
    s.add_argument("--model", required=True)
    s.add_argument("--records", type=Path, default=ROOT / "reranker-runs")
    s.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "prepare-evidence-labels":
        prepare_evidence_labels(args)
    elif args.command == "run":
        asyncio.run(run(args))
    else:
        summarize(args)
