import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rerank_eval import (
    Remote,
    confidence,
    digest,
    evidence_ids,
    load_records,
    locomo_source,
    mapped_scores,
    metrics,
    rank_scores,
    validate_fixture,
)


def test_recall_counts_all_evidence_instead_of_hit_rate():
    values = metrics(["b", "x", "y"], {"a": 1, "b": 1})
    assert values["recall_at_5"] == 0.5
    assert values["hit_at_5"] == 1
    assert values["ndcg_at_10"] < 1


def test_retrieval_miss_counts_as_zero():
    assert all(v == 0 for v in metrics(["x", "y"], {"a": 1}).values())


def test_rank_matches_original_indices_and_breaks_ties_by_input_order():
    scores = mapped_scores(
        [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
            {"index": 1, "relevance_score": 0.9},
        ],
        3,
    )
    assert rank_scores(scores, 3) == [1, 2, 0]


@pytest.mark.parametrize(
    "rows",
    [
        [{"index": 0, "relevance_score": 0.5}],
        [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.7}],
        [{"index": -1, "relevance_score": 0.5}, {"index": 1, "relevance_score": 0.7}],
        [
            {"index": 0, "relevance_score": float("nan")},
            {"index": 1, "relevance_score": 0.7},
        ],
    ],
)
def test_invalid_provider_results_fail_visibly(rows):
    with pytest.raises(ValueError):
        mapped_scores(rows, 2)


def test_evidence_formatting_is_normalized_without_guessing():
    assert evidence_ids(["D8:6; D9:17", "D:11:26", "D30:05"]) == [
        "D8:6",
        "D9:17",
        "D11:26",
        "D30:5",
    ]
    with pytest.raises(ValueError):
        evidence_ids(["D1:18", "D", "D1:20"])


def test_dataset_exclusions_are_independent_of_retrieval(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "chat",
                    "conversation": {
                        "session_1_date_time": "today",
                        "session_1": [
                            {"dia_id": "D1:1", "speaker": "A", "text": "Evidence"}
                        ],
                    },
                    "qa": [
                        {"question": "valid", "category": 1, "evidence": ["D1:1"]},
                        {"question": "broken", "category": 1, "evidence": ["D1:9"]},
                        {"question": "empty", "category": 1, "evidence": []},
                    ],
                }
            ]
        )
    )
    _, questions, excluded, _ = locomo_source(path)
    assert [q["query"] for q in questions] == ["valid"]
    assert len(excluded) == 2


def test_fixture_tampering_is_rejected():
    fixture = {
        "corpus": {"a": "A"},
        "queries": [{"id": "q", "relevant": {"a": 1}, "candidates": ["a"]}],
    }
    fixture["fixture_sha256"] = digest(fixture)
    validate_fixture(fixture)
    fixture["corpus"]["a"] = "changed"
    with pytest.raises(ValueError, match="hash"):
        validate_fixture(fixture)


def test_resume_rejects_changed_adapter_or_fixture(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"run_sha256": "old", "query_id": "q"}) + "\n")
    with pytest.raises(ValueError, match="Resume refused"):
        load_records(path, "new")


def test_duplicate_resume_records_are_rejected(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text((json.dumps({"run_sha256": "same", "query_id": "q"}) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_records(path, "same")


def test_jev_receives_only_one_candidate_in_each_state():
    remote = object.__new__(Remote)
    remote.config = {"provider": "typesafe", "model": "jev-1.13.0"}
    bodies = []

    async def post(body):
        bodies.append(body)
        candidate = body["state"]["candidate"]
        return {
            "model": "jev-1.13.0",
            "answers": {"relevance": {"noul": 0.9 if candidate == "gold" else 0.1}},
            "usage": {"input_tokens": 10},
        }, 0.01

    remote.post = post
    scores, usage, _ = asyncio.run(remote.score("query", ["gold", "other"]))
    assert scores == [0.9, 0.1]
    assert usage["input_tokens"] == 20
    assert [body["state"] for body in bodies] == [
        {"query": "query", "candidate": "gold"},
        {"query": "query", "candidate": "other"},
    ]


def test_bootstrap_preserves_conversation_clustering():
    rows = [{"group": "a", "metrics": {"mrr": 1.0}}] * 100 + [
        {"group": "b", "metrics": {"mrr": 0.0}}
    ] * 100
    assert confidence(rows, "mrr", "locomo") == [0.0, 1.0]


def test_paired_interval_compares_same_queries_and_cancels_shared_variation():
    from summarize_rerankers import paired_interval

    a = {
        "q1": {"group": "chat1", "metrics": {"mrr": 0.8}},
        "q2": {"group": "chat2", "metrics": {"mrr": 0.4}},
    }
    b = {
        "q1": {"group": "chat1", "metrics": {"mrr": 0.6}},
        "q2": {"group": "chat2", "metrics": {"mrr": 0.2}},
    }
    result = paired_interval(a, b, "mrr")
    assert result["delta"] == pytest.approx(0.2)
    assert result["ci95"] == pytest.approx([0.2, 0.2])
    with pytest.raises(ValueError, match="identical queries"):
        paired_interval(a, {"q1": b["q1"]}, "mrr")


def test_partial_runs_cannot_publish_quality(tmp_path):
    from types import SimpleNamespace
    from rerank_eval import summarize, write_json

    fixture = {
        "schema_version": 2,
        "dataset": "scifact",
        "corpus": {"a": "Evidence", "b": "Other"},
        "queries": [
            {
                "id": "q1",
                "group": "q1",
                "category": "fact-checking",
                "query": "First",
                "relevant": {"a": 1},
                "candidates": ["a", "b"],
            },
            {
                "id": "q2",
                "group": "q2",
                "category": "fact-checking",
                "query": "Second",
                "relevant": {"a": 1},
                "candidates": ["a", "b"],
            },
        ],
    }
    fixture["fixture_sha256"] = digest(fixture)
    fixture_path = tmp_path / "fixture.json.gz"
    write_json(fixture_path, fixture)
    metadata = {
        "config": {"provider": "rrf", "pricing_type": "free"},
        "fixture_sha256": fixture["fixture_sha256"],
    }
    path = tmp_path / "records" / "scifact" / "rrf.jsonl"
    path.parent.mkdir(parents=True)
    write_json(path.with_suffix(".metadata.json"), metadata)
    path.write_text(
        json.dumps({"run_sha256": digest(metadata), "query_id": "q1"}) + "\n"
    )
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="Incomplete run"):
        summarize(
            SimpleNamespace(
                fixture=fixture_path,
                model="rrf",
                records=tmp_path / "records",
                output=output,
            )
        )
    assert not output.exists()


def tiny_complete_run(tmp_path):
    from types import SimpleNamespace
    from rerank_eval import write_json

    fixture = {
        "dataset": "scifact",
        "corpus": {"a": "First passage", "b": "Second passage"},
        "queries": [
            {
                "id": "q1",
                "group": "q1",
                "category": "fact-checking",
                "query": "Claim",
                "relevant": {"a": 1},
                "candidates": ["a", "b"],
            }
        ],
    }
    fixture["fixture_sha256"] = digest(fixture)
    args = SimpleNamespace(
        fixture=tmp_path / "fixture.json.gz",
        labels=None,
        model="rrf",
        records=tmp_path / "records",
        output=tmp_path / "published",
    )
    write_json(args.fixture, fixture)
    metadata = {
        "config": {"provider": "rrf", "pricing_type": "free"},
        "fixture_sha256": fixture["fixture_sha256"],
    }
    path = args.records / "scifact" / "rrf.jsonl"
    path.parent.mkdir(parents=True)
    write_json(path.with_suffix(".metadata.json"), metadata)
    row = {
        "run_sha256": digest(metadata),
        "query_id": "q1",
        "group": "q1",
        "category": "fact-checking",
        "scores": [0.9, 0.1],
        "order": [0, 1],
        "metrics": metrics(["a", "b"], {"a": 1}),
        "latency_s": 0.1,
        "retries": 0,
        "usage": {},
    }
    path.write_text(json.dumps(row) + "\n")
    return args, fixture


def test_summary_rejects_changed_passage_text_with_same_ids_and_labels(tmp_path):
    from rerank_eval import summarize, write_json

    args, fixture = tiny_complete_run(tmp_path)
    fixture["corpus"]["a"] = "Changed model input"
    fixture.pop("fixture_sha256")
    fixture["fixture_sha256"] = digest(fixture)
    write_json(args.fixture, fixture)
    with pytest.raises(ValueError, match="metadata does not match"):
        summarize(args)
    assert not args.output.exists()


def test_evidence_view_reuses_scores_and_preserves_original_metrics(tmp_path):
    from rerank_eval import read_json, summarize, write_json

    args, _ = tiny_complete_run(tmp_path)
    view = {
        "name": "scifact-evidence",
        "labels": {"q1": {"b": 1}},
        "excluded_queries": {},
        "query_sha256": {"q1": hashlib.sha256(b"Claim").hexdigest()},
    }
    view["label_sha256"] = digest(view)
    args.labels = tmp_path / "evidence-labels.json"
    write_json(args.labels, view)
    summarize(args)
    result = read_json(args.output / "rrf--scifact-evidence.json")
    rows = read_json(args.output / "records" / "rrf--scifact-evidence.json.gz")
    assert result["mrr"] == 0.5
    assert result["evaluation_label_sha256"] == view["label_sha256"]
    assert rows[0]["scores"] == [0.9, 0.1]
    assert rows[0]["source_metrics"]["mrr"] == 1
    assert rows[0]["metrics"]["mrr"] == 0.5
    view["query_sha256"]["q1"] = hashlib.sha256(b"Different claim").hexdigest()
    view.pop("label_sha256")
    view["label_sha256"] = digest(view)
    write_json(args.labels, view)
    with pytest.raises(ValueError, match="mismatched source query text"):
        summarize(args)


def test_server_overload_retries_without_turning_failure_into_irrelevance(monkeypatch):
    import httpx
    from types import SimpleNamespace

    monkeypatch.setenv("BENCHMARK_TEST_KEY", "fake-test-key")
    attempts = []

    def handle(request):
        attempts.append(request)
        if len(attempts) == 1:
            return httpx.Response(529, headers={"retry-after": "0"})
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "relevance_score": 0.1},
                    {"index": 0, "relevance_score": 0.9},
                ],
                "usage": {"total_tokens": 20},
            },
        )

    async def exercise():
        remote = Remote(
            {
                "provider": "voyage",
                "model": "fake",
                "endpoint": "https://provider.example/rerank",
                "key_env": "BENCHMARK_TEST_KEY",
            },
            SimpleNamespace(secrets_dir=None, concurrency=2, requests_per_second=1000),
        )
        await remote.client.aclose()
        remote.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            scores, usage, _ = await remote.score("query", ["gold", "other"])
            assert scores == [0.9, 0.1]
            assert usage["total_tokens"] == 20
            assert remote.retries == 1
            assert metrics(["gold", "other"], {"gold": 1})["mrr"] == 1
        finally:
            await remote.close()

    asyncio.run(exercise())
    assert len(attempts) == 2


def test_jev_listwise_ranks_shared_pool_and_counts_one_request():
    remote = object.__new__(Remote)
    remote.config = {
        "provider": "typesafe",
        "model": "jev-1.13.0",
        "ranking_mode": "hindsight-choice",
    }
    bodies = []

    async def post(body):
        bodies.append(body)
        return {
            "model": "jev-1.13.0",
            "answers": {"rank": {"probabilities": {"c0": 0.1, "c1": 0.9}}},
            "usage": {"input_tokens": 25},
        }, 0.02

    remote.post = post
    scores, usage, _ = asyncio.run(remote.score("query", ["other", "gold"]))
    assert rank_scores(scores, 2) == [1, 0]
    assert usage == {"input_tokens": 25, "choice_calls": 1}
    assert len(bodies) == 1
    question = bodies[0]["questions"]["rank"]
    assert question["type"] == "choice"
    assert question["criteria"] == {"c0": "other", "c1": "gold"}
    assert bodies[0]["state"] == "Question: query"


def test_jev_listwise_tournament_compares_finalists_and_preserves_rest_input_order():
    from jev_listwise import HindsightChoice

    remote = object.__new__(Remote)
    remote.config = {"model": "jev-1.13.0"}
    bodies = []

    async def post(body):
        bodies.append(body)
        options = body["questions"]["rank"]["criteria"]
        count = len(options)
        return {
            "model": "jev-1.13.0",
            "answers": {
                "rank": {
                    "probabilities": {
                        key: (i + 1) / (count * (count + 1) / 2)
                        for i, key in enumerate(options)
                    }
                }
            },
            "usage": {"input_tokens": count},
        }, 0.01

    remote.post = post
    choice = HindsightChoice(remote)
    choice.MAX_OPTIONS = 4
    choice.SHORTLIST = 1
    scores, usage, _ = asyncio.run(choice.score("query", [str(i) for i in range(9)]))
    # Last singleton advances without an invalid one-option preliminary Choice.
    # Finalists 3, 7, 8 compete together; all other candidates retain input order.
    assert rank_scores(scores, 9) == [8, 7, 3, 0, 1, 2, 4, 5, 6]
    assert usage["choice_calls"] == 3
    assert all(len(body["questions"]["rank"]["criteria"]) >= 2 for body in bodies)


def test_embedding_experiment_applies_model_specific_prompts():
    from embedding_experiment import MODELS, encode

    class Encoder:
        def encode(self, texts, **kwargs):
            return texts, kwargs

    _, query = encode(Encoder(), ["q"], MODELS["voyage"], "query", 8)
    _, document = encode(Encoder(), ["d"], MODELS["voyage"], "document", 8)
    _, granite = encode(Encoder(), ["q"], MODELS["granite"], "query", 8)
    assert query["prompt"] != document["prompt"]
    assert granite["prompt"] == ""
    assert query["normalize_embeddings"] is True
    _, nano = encode(Encoder(), ["q"], MODELS["voyage-1024"], "query", 8)
    assert nano["truncate_dim"] == 1024
    assert nano["normalize_embeddings"] is True
    _, small = encode(Encoder(), ["q"], MODELS["bge-small"], "query", 8)
    assert small["prompt"] == ""
    assert "truncate_dim" not in small
    _, leaf_query = encode(Encoder(), ["q"], MODELS["leaf"], "query", 8)
    _, leaf_doc = encode(Encoder(), ["d"], MODELS["leaf"], "document", 8)
    assert (
        leaf_query["prompt"]
        == "Represent this sentence for searching relevant passages: "
    )
    assert leaf_doc["prompt"] == ""
    assert leaf_query["normalize_embeddings"] is True


def test_embedding_candidate_selection_uses_ranks_without_gold_injection():
    import numpy as np
    from embedding_experiment import fused_candidates

    ids = ["a", "b", "gold"]
    chosen = fused_candidates(ids, np.array([3, 2, 1]), np.array([3, 2, 1]), 2)
    assert chosen == ["a", "b"]


def test_concurrent_embedding_run_writes_each_pending_query_once_and_resumes():
    from embedding_experiment import concurrent_queries

    fixture = {
        "corpus": {"a": "first", "b": "second"},
        "queries": [
            {
                "id": str(i),
                "query": "q",
                "group": "chat",
                "category": 1,
                "candidates": ["a", "b"],
                "relevant": {"b": 1},
            }
            for i in range(10)
        ],
    }

    class Fake:
        retries = 0

        async def score(self, query, docs):
            await asyncio.sleep(0)
            return [0.1, 0.9], {"input_tokens": 2}, 0.001

    records = {"0": {"already": "done"}}
    appended = []
    asyncio.run(
        concurrent_queries(
            fixture, [Fake(), Fake(), Fake()], records, "hash", appended.append
        )
    )
    assert len(appended) == 9
    assert set(records) == {str(i) for i in range(10)}
    assert all(row["order"] == [1, 0] for row in appended)
    assert records["0"] == {"already": "done"}
    asyncio.run(concurrent_queries(fixture, [Fake()], records, "hash", appended.append))
    assert len(appended) == 9
