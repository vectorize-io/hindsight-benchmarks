"""Behavior checks for sampling, repository isolation and retrieval metrics."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memgovern_prepare import read_items, selected_queries, validate_vectors
from memgovern_experiment import (
    aggregate,
    checked_order,
    freeze_repository,
    single_positive_metrics,
    validate_pool,
)
from rerank_eval import digest, read_json, write_json


def test_query_sampling_ignores_input_order_and_keeps_full_corpus():
    queries = [{"id": str(i), "text": f"bug {i}"} for i in range(1000)]
    selected = selected_queries(queries, "project", 200)
    assert selected == selected_queries(list(reversed(queries)), "project", 200)
    assert len({r["id"] for r in selected}) == 200
    assert selected != selected_queries(queries, "other-project", 200)
    assert len(queries) == 1000
    with pytest.raises(ValueError):
        selected_queries(queries, "project", 1001)


def test_duplicate_source_ids_fail_instead_of_overwriting(tmp_path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"id": "a", "text": text}) for text in ("bug", "another bug")
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_items(path)


def test_embeddings_require_finite_unit_vectors():
    validate_vectors(np.array([[0, 1]], dtype=np.float32), 1)
    for vector in ([0, 0], [1, 1], [float("nan"), 0]):
        with pytest.raises(ValueError):
            validate_vectors(np.array([vector], dtype=np.float32), 1)


def test_miss_is_zero_and_ndcg_is_distinct_from_reciprocal_rank():
    order = ["other", "gold"]
    measured = single_positive_metrics(order, "gold")
    assert measured["ndcg_at_10"] == pytest.approx(1 / np.log2(3))
    assert measured["mrr"] == 0.5
    assert measured["recall_at_1"] == 0
    assert measured["recall_at_100"] == 1
    assert all(
        value == 0 for value in single_positive_metrics(order, "absent").values()
    )
    with pytest.raises(ValueError):
        single_positive_metrics(["gold", "gold"], "gold")


def test_repository_macro_does_not_overweight_larger_repository():
    rows = [{"query_id": "a/1", "group": "a", "metrics": {"mrr": 1}}]
    rows.extend(
        {"query_id": f"b/{i}", "group": "b", "metrics": {"mrr": 0}} for i in range(9)
    )
    summary = aggregate(rows)
    assert summary["repository_macro"]["mrr"] == 0.5
    assert summary["query_macro"]["mrr"] == 0.1
    with pytest.raises(ValueError):
        aggregate(rows + rows[:1])


def test_changed_pool_and_invalid_permutation_are_rejected():
    pool = {
        "queries": [
            {
                "id": "a",
                "dense": list(map(str, range(100))),
                "hybrid": list(map(str, range(100))),
            }
        ]
    }
    pool["pool_sha256"] = digest(pool)
    validate_pool(pool)
    pool["queries"][0]["dense"][0] = "changed"
    with pytest.raises(ValueError, match="checksum"):
        validate_pool(pool)
    with pytest.raises(ValueError):
        checked_order({"scores": [1, 2], "order": [0, 1]}, count=2)


def test_freezing_uses_all_documents_without_consulting_qrels(tmp_path):
    args = SimpleNamespace(work=tmp_path / "work", data=tmp_path / "data")
    repo = "project"
    base = args.data / "Procedural/MemGovern" / repo
    base.mkdir(parents=True)
    # 120 documents, one query: corpus is never restricted to sampled positives.
    corpus = [{"id": f"doc-{i:03}", "text": f"experience {i}"} for i in range(120)]
    queries = [{"id": "q", "text": "experience"}]
    for filename, rows in (("corpus.jsonl", corpus), ("queries.jsonl", queries)):
        (base / filename).write_text("\n".join(json.dumps(r) for r in rows))
    # Invalid label content deliberately proves inference does not parse it.
    (base / "qrels.tsv").write_text("not a relevance file")
    import hashlib

    files = {
        name: hashlib.sha256((base / name).read_bytes()).hexdigest()
        for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv")
    }
    write_json(args.work / "source.json", {"repositories": {repo: {"files": files}}})
    metadata = {"selection": {"count_per_repository": 1, "seed": 20261003}}
    enc = args.work / "bge-small/encodings"
    write_json(enc / "metadata.json", metadata)
    vectors = np.zeros((120, 2), dtype=np.float32)
    vectors[:, 0] = 1
    for kind, rows, values in (
        ("document", corpus, vectors),
        ("query", queries, vectors[:1]),
    ):
        np.savez(
            enc / f"{repo}.{kind}.npz",
            fingerprint=digest(metadata),
            ids=np.array([r["id"] for r in rows]),
            vectors=values,
        )
    assert freeze_repository(args, "bge-small", repo)
    pool = read_json(args.work / "bge-small/pools" / f"{repo}.json.gz")
    assert pool["corpus_count"] == 120
    assert pool["queries"][0]["dense"] == [r["id"] for r in corpus[:100]]
    assert len(pool["queries"][0]["hybrid"]) == 100


def test_checkpoint_can_change_hardware_but_not_embedding_semantics(tmp_path):
    from memgovern_gpu_prepare import checkpoint_metadata, SEMANTIC_FIELDS

    initial = {key: key for key in SEMANTIC_FIELDS}
    initial["device"] = "cpu"
    execution = {**initial, "device": "cuda:0", "hardware": "RTX A4000"}
    path = tmp_path / "vectors.npz"
    np.savez(
        path, execution_metadata=json.dumps(execution), fingerprint=digest(execution)
    )
    with np.load(path, allow_pickle=False) as checkpoint:
        assert checkpoint_metadata(checkpoint, initial)["device"] == "cuda:0"
    execution["config"] = "different model or prompt"
    np.savez(
        path, execution_metadata=json.dumps(execution), fingerprint=digest(execution)
    )
    with np.load(path, allow_pickle=False) as checkpoint:
        with pytest.raises(ValueError, match="semantics"):
            checkpoint_metadata(checkpoint, initial)
    np.savez(path, fingerprint="altered")
    with np.load(path, allow_pickle=False) as checkpoint:
        with pytest.raises(ValueError, match="provenance"):
            checkpoint_metadata(checkpoint, initial)


def test_reranking_shard_only_visits_its_owned_embedding(tmp_path):
    import asyncio
    from memgovern_experiment import process_pools

    args = SimpleNamespace(work=tmp_path, watch=False, embeddings=["leaf"])
    write_json(tmp_path / "source.json", {"repositories": {"repo": {}}})
    pool = {
        "queries": [
            {
                "id": "q",
                "dense": list(map(str, range(100))),
                "hybrid": list(map(str, range(100))),
            }
        ]
    }
    pool["pool_sha256"] = digest(pool)
    write_json(tmp_path / "leaf/pools/repo.json.gz", pool)
    visited = []

    async def callback(embedding, repo, pool):
        visited.append((embedding, repo))

    asyncio.run(process_pools(args, callback))
    assert visited == [("leaf", "repo")]
    args.embeddings = ["leaf", "leaf"]
    with pytest.raises(ValueError, match="ownership"):
        asyncio.run(process_pools(args, callback))


def test_pointwise_cache_maps_input_indices_not_ranked_positions():
    from memgovern_experiment import cached_pair_scores, merge_pair_scores

    baseline = {"id": "q", "text": "bug", "hybrid": [f"d{i}" for i in range(100)]}
    record = {
        "query_id": "q",
        "scores": list(map(float, range(100))),
        "order": list(reversed(range(100))),
    }
    query = {"id": "q", "text": "bug"}
    cache = cached_pair_scores(query, baseline, record)
    scores, indices = merge_pair_scores(["d7", "unseen", "d1"], cache, [0.5])
    assert scores == [7, 0.5, 1]
    assert indices == [0, 2]
    with pytest.raises(ValueError, match="query"):
        cached_pair_scores({"id": "q", "text": "different bug"}, baseline, record)
    with pytest.raises(ValueError, match="Incomplete"):
        merge_pair_scores(["d7", "unseen"], cache, [])
    with pytest.raises(ValueError):
        merge_pair_scores(["d7"], {"d7": float("nan")}, [])
