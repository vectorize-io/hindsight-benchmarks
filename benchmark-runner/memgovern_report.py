"""Render the complete, independently checked MemGovern comparison."""

import numpy as np

from memgovern_experiment import OUTPUT, PATHS
from memgovern_prepare import MODEL_NAMES
from rerank_eval import ROOT, digest, read_json

NAMES = {
    "bge-small": "BGE-small (Hindsight default)",
    "granite": "Granite Small English R2",
    "leaf": "MongoDB Leaf IR",
    "voyage-1024": "Voyage 4 Nano (1024d)",
}
LABELS = {
    "dense": "Dense only",
    "jev-listwise": "Hybrid + Jev listwise",
    "ettin-150m": "Hybrid + Ettin 150M",
}


def absolute_interval(summary, metric):
    values = np.array(
        [row[metric] for _, row in sorted(summary["per_repository"].items())]
    )
    rng = np.random.default_rng(20261003)
    samples = rng.integers(0, len(values), size=(5000, len(values)))
    return np.quantile(values[samples].mean(axis=1), [0.025, 0.975])


def report():
    result = read_json(OUTPUT / "summary.json")
    if len(result["results"]) != 12 or result["independent_metric_checks"] != 691200:
        raise ValueError("Expected all twelve validated 9600-query combinations")
    lines = [
        "# MemGovern embeddings and rerankers",
        "",
        "All twelve combinations evaluate the same **9,600 queries: 200 per repository across 48 repositories**, selected by the lowest SHA256 of seed20261003, repository and original query ID. Each query searches its repository’s **entire corpus**, covering 121,475 experience cards in total. Candidate selection never uses relevance labels; no gold card is inserted into a candidate pool.",
        "",
        "The primary numbers are repository-macro averages. Query-macro averages are identical for this balanced sample. Dense-only uses normalized embedding cosine top100. Both rerankers receive the same top100 pool from full-corpus BM25/dense RRF, with k=60 and one-based ranks. Jev uses the vendored Hindsight listwise Choice algorithm with pruning off; Ettin uses independent query/document scores.",
        "",
        "| Embedding | Retrieval path | nDCG@10 (95% CI) | MRR@100 | Recall@5 | Recall@100 | Successful Jev API cost |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for embedding in MODEL_NAMES:
        for path in PATHS:
            summary = result["results"][embedding + "/" + path]
            m = summary["repository_macro"]
            lo, hi = absolute_interval(summary, "ndcg_at_10")
            cost = (
                f"${summary['usd_successful_requests']:.3f}"
                if path == "jev-listwise"
                else "Local compute unpriced"
            )
            lines.append(
                f"| {NAMES[embedding]} | {LABELS[path]} | {m['ndcg_at_10']:.4f} [{lo:.4f}, {hi:.4f}] | {m['mrr']:.4f} | {m['recall_at_5']:.4f} | {m['recall_at_100']:.4f} | {cost} |"
            )
    diagnostic_path = OUTPUT / "bm25-diagnostic.json"
    ablation_path = OUTPUT / "bm25-ettin" / "summary.json"
    if diagnostic_path.exists() and ablation_path.exists():
        diagnostic = read_json(diagnostic_path)
        ablation = read_json(ablation_path)
        if (
            diagnostic["independent_metric_checks"] != 288000
            or ablation["independent_metric_checks"] != 57600
            or ablation["summary"]["queries"] != 9600
            or ablation["summary"]["repositories"] != 48
            or diagnostic["source_sha256"] != ablation["source_sha256"]
            or ablation["source_sha256"] != digest(read_json(OUTPUT / "source.json"))
        ):
            raise ValueError("Expected complete independently checked BM25 ablations")
        lines.extend(
            [
                "",
                "## BM25 and reranking ablation",
                "",
                "These additional paths use the same 9,600 sampled queries and full repository corpora. BM25 uses default BM25Okapi with lowercase regex word tokens and stable descending top100. Hybrid rows reuse the original frozen RRF pools without reranking. BM25 + Ettin uses only BM25 candidates, with no embedding retrieval.",
                "",
                "| Retrieval path | nDCG@10 | MRR@100 | Recall@100 |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, summary in [
            ("BM25 only", diagnostic["results"]["bm25"]),
            *[
                ("BM25 + " + NAMES[name], diagnostic["results"][name + "/hybrid"])
                for name in MODEL_NAMES
            ],
            ("BM25 + Ettin 150M", ablation["summary"]),
        ]:
            metrics = summary["repository_macro"]
            lines.append(
                f"| {label} | {metrics['ndcg_at_10']:.4f} | {metrics['mrr']:.4f} | {metrics['recall_at_100']:.4f} |"
            )
        delta = ablation["paired_minus_bge_hybrid"]
        lo, hi = delta["ci95"]
        lines.extend(
            [
                "",
                f"BM25 + Ettin minus BM25/BGE-small + Ettin is {delta['delta']:+.4f} nDCG@10, with paired repository-bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]. BM25 improves BGE-small's initial ranking, while Ettin closes most of the remaining embedding-model differences. This is a retrieval result on MemGovern's single-positive labels, not evidence that embeddings are unnecessary for other workloads.",
                "",
                f"The ablation uses the same pinned Ettin revision, BF16, batch32, Identity activation and maximum length8192. It reuses {ablation['cached_pairs']:,} verified BGE-baseline query/card logits and scores {ablation['computed_pairs']:,} previously unseen pairs on the Blackwell. Frozen BM25 candidates, scores, complete permutations and cache-source provenance are retained. All six metrics are independently checked with pytrec_eval for every query: 57,600 comparisons for BM25 + Ettin and 288,000 for the five unreranked diagnostics. [Diagnostics](../results/experiments/memgovern/bm25-diagnostic.json), [Ettin ablation summary](../results/experiments/memgovern/bm25-ettin/summary.json), [full per-query records](../results/experiments/memgovern/bm25-ettin/records.json.gz).",
                "",
                "The exact executed diagnostic and GPU scoring scripts are preserved as [diagnostic source](../results/experiments/memgovern/audit-sources/bm25_diagnostic.py.txt) and [Ettin ablation source](../results/experiments/memgovern/audit-sources/bm25_ettin.py.txt). They import the pinned benchmark helpers from the repository root and use the same source/work directories and model environment as the main run.",
            ]
        )
    lines.extend(
        [
            "",
            "## Paired differences versus BGE-small",
            "",
            "Differences are nDCG@10, with 5,000 paired bootstrap samples of whole repositories. Intervals are 95% percentile intervals, unadjusted for multiple comparisons. They describe uncertainty across repositories for this fixed query sample, not the additional variation from drawing another sample of queries. Resampling units are the official repository corpora; historical names such as the two Airflow and two Home Assistant banks represent related projects, so these intervals should not be read as 48 fully independent project samples.",
            "",
            "| Embedding minus BGE-small | Retrieval path | nDCG@10 difference (95% CI) |",
            "|---|---|---:|",
        ]
    )
    for pair in result["paired_comparisons"]:
        if pair["a"] != "bge-small":
            continue
        delta = pair["ndcg_at_10"]
        # Stored comparisons are a minus b; present new model minus baseline.
        lo, hi = [-delta["ci95"][1], -delta["ci95"][0]]
        lines.append(
            f"| {NAMES[pair['b']]} | {LABELS[pair['path']]} | {-delta['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] |"
        )
    audit_path = OUTPUT / "a4000-cpu-agreement.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        lines.extend(
            [
                "",
                "## CPU and A4000 agreement check",
                "",
                f"The same {len(audit['query_ids'])} Granite queries were independently reencoded on CPU and compared with their actual A4000 checkpoints. Maximum absolute vector difference was {audit['maximum_absolute_vector_difference']:.3g}; minimum embedding cosine agreement was {audit['minimum_embedding_cosine_agreement']:.12f}. Both devices produced unit-normalized vectors. This sampled check supports numerical agreement of the device continuation; it does not prove every corpus ranking is identical. [Query IDs, model/device provenance and measurements](../results/experiments/memgovern/a4000-cpu-agreement.json).",
            ]
        )
    cache_audit_path = OUTPUT / "ettin-pointwise-cache-audit.json"
    if cache_audit_path.exists():
        cache_audit = read_json(cache_audit_path)["results"]
        granite_audit, leaf_audit = cache_audit["granite"], cache_audit["leaf"]
        lines.extend(
            [
                "",
                "## Pointwise cache and numerical sensitivity",
                "",
                f"The pre-cache runs independently scored {granite_audit['shared_pairs']:,} shared Granite/BGE pairs on the Blackwell and {leaf_audit['shared_pairs']:,} shared Leaf/BGE pairs across A4000/Blackwell. Replacing shared logits with their BGE baseline values would change mean nDCG@10 by {granite_audit['ndcg_at_10_delta_if_shared_pairs_replaced']:+.6f} for {granite_audit['queries']:,} Granite queries and {leaf_audit['ndcg_at_10_delta_if_shared_pairs_replaced']:+.6f} for {leaf_audit['queries']:,} Leaf queries. Granite shared logits were almost bit-identical; Leaf’s cross-device logits had p99 absolute difference {leaf_audit['logit_difference_p99']:.4f}, maximum {leaf_audit['maximum_absolute_logit_difference']:.4f}, and {leaf_audit['top1_changes']} top1 changes. These are BF16 batch/device effects, so tiny reranked differences should not be treated as model superiority. Actual completed rankings are retained; the remaining computations record explicit cache provenance. [Audit measurements](../results/experiments/memgovern/ettin-pointwise-cache-audit.json).",
            ]
        )
    total_cost = sum(
        row["usd_successful_requests"] for row in result["results"].values()
    )
    lines.extend(
        [
            "",
            "## What these labels establish",
            "",
            "MemGovern’s public qrels label one source experience card per original bug query. The task measures recovery of the card distilled from that bug’s fix. Other useful or semantically equivalent cards receive no credit. This does not directly measure whether retrieved experiences help solve a different bug, nor agent task success. Models may have encountered the underlying public repositories or benchmark data during training; this experiment does not test contamination.",
            "",
            "The source is [KaLM-Embedding/LMEB](https://huggingface.co/datasets/KaLM-Embedding/LMEB), pinned to `f9d4294be4a24b8c16d6bfb59d56a6fec4bd95c0`. The official per-repository corpus boundaries and positive associations are retained. Queries are sampled independently of text, labels and model scores, without filtering hard or unsuccessful queries. [Source hashes and counts](../results/experiments/memgovern/source.json).",
            "",
            "## Models and execution",
            "",
            "BGE-small uses 384 dimensions and empty query/document prompts, matching the current Hindsight default. Granite uses 384 dimensions, CLS pooling and empty prompts. Leaf uses its full native 768 dimensions and documented retrieval query prefix; documents are unprefixed. Voyage Nano uses its documented distinct query/document prompts and truncates to 1024 dimensions before normalization. All embeddings use float32 SDPA and unit normalization. All model revisions and prompts are recorded in each frozen pool’s encoding metadata; none are quantized.",
            "",
            "Nano runs on the RTX PRO 5000 Blackwell first. BGE-small, Granite and Leaf initially run concurrently on CPU, each with eight worker processes and four Torch threads per worker. Completed CPU vectors are retained; remaining embeddings are then computed on the idle RTX A4000, one model at a time, with the same float32 weights, prompts and normalization. Each GPU checkpoint records its actual execution metadata separately from the initial CPU configuration, and newly frozen pools retain that provenance. Ettin uses disjoint model shards: Leaf’s candidates are reranked on the A4000 after embedding preparation finishes, and the other three models on the Blackwell after Nano finishes, using pinned revision `025501c4e0f9bbeb4c5b198318e0089ff061cc14`, BF16 weights, batch32, native maximum context8192 and Identity activation. Hosted Jev1.13 runs concurrently with eight independent query workers, each paced at two requests per second. Repeated Ettin query/card pairs reuse verified raw logits from the completed BGE-small run; only previously unseen pairs require new inference. Cache reuse checks query text, source identity, scorer configuration and environment, and exported cached scores are checked against their baseline input indices. Records preserve cached indices and source-run hashes, so reused Blackwell logits remain distinguishable from fresh A4000 computation. Concurrent timings measure incremental work and are not isolated full-rerank speed benchmarks. Leaf’s initial worker pool failed before producing a checkpoint; its identical restart supplied the measured vectors. CPU preparation was later resumed with TOKENIZERS_PARALLELISM=false to prevent each worker spawning a full-host tokenizer thread pool; completed embeddings were reused and model/tokenization settings were unchanged.",
            "",
            f"Total successful Jev input-token cost across four embeddings is **${total_cost:.3f}**, at $0.042 per million input tokens. Output tokens are uncharged. Costs exclude warmup and unknown billing for failed requests/retries; local electricity and hardware are unpriced. Raw successful usage and retry counts are retained. Jev’s listwise algorithm may split oversized pools and use a final selection round; candidate pruning remains off and all100 candidates remain in the final ranking.",
            "",
            "## Validation and reproduction",
            "",
            "The exporter requires all 9,600 queries in all twelve combinations, exact source/pool/run provenance, finite scores, complete candidate permutations and exact provenance/value equality for reused pointwise scores. It independently recomputes nDCG@10, MRR@100 and Recall@1/5/10/100 with pytrec_eval: **691,200 metric comparisons**. Retrieval misses count as zero; incomplete runs produce no completed summary. [Summaries, paired intervals, frozen pools and full per-query records](../results/experiments/memgovern/summary.json).",
            "",
            "Download `Procedural/MemGovern/<repository>/{corpus.jsonl,queries.jsonl,qrels.tsv}` from the pinned dataset revision into `<data>/Procedural/MemGovern/`, and put the revision in `<data>/revision.txt`. Use the package versions recorded in pool metadata and cached pinned models. From the repository root:",
            "",
            "    python benchmark-runner/memgovern_prepare.py --embedding voyage-1024 --device cuda:0 --threads 8 --data <data> --work <work>",
            "    python benchmark-runner/memgovern_prepare.py --embedding bge-small --data <data> --work <work>",
            "    python benchmark-runner/memgovern_prepare.py --embedding granite --data <data> --work <work>",
            "    python benchmark-runner/memgovern_prepare.py --embedding leaf --data <data> --work <work>",
            "    python benchmark-runner/memgovern_gpu_prepare.py --embedding granite --data <data> --work <work>",
            "    python benchmark-runner/memgovern_gpu_prepare.py --embedding bge-small --data <data> --work <work>",
            "    python benchmark-runner/memgovern_gpu_prepare.py --embedding leaf --data <data> --work <work>",
            "    python benchmark-runner/memgovern_experiment.py freeze --data <data> --work <work>",
            "    python benchmark-runner/memgovern_experiment.py rerank --reranker jev-listwise --data <data> --work <work>",
            "    python benchmark-runner/memgovern_experiment.py rerank --reranker ettin-150m --data <data> --work <work>",
            "    python benchmark-runner/memgovern_experiment.py export --data <data> --work <work>",
            "    python benchmark-runner/memgovern_report.py",
            "",
            "For CPU preparation, set CUDA_VISIBLE_DEVICES to empty, OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=1 and TOKENIZERS_PARALLELISM=false. The three CPU commands can run concurrently; preparation checkpoints are reused only when their exact source/model/configuration fingerprints match. Model files are loaded from the local cache. For Nano, use the existing isolated Transformers4.57.6 compatibility overlay and the explicit Qwen3Config registration shim; forward computation and weights are unchanged. Other models and rerankers use Transformers5.18.0. Jev reads the authorized key from `~/.secrets/typesafe_api_key`. The GPU continuation commands require existing initial preparation metadata, accept hardware/scheduling changes, reject changed source/model/prompt/precision settings, and retain completed checkpoints. Select the A4000 by CUDA_VISIBLE_DEVICES for these continuation commands, and the Blackwell for Ettin. Source corpora and large embedding arrays are not duplicated into Git; the source manifest, frozen pools and records permit audit/reproduction.",
            "",
            "No production defaults, embedding stores or live installations changed.",
            "",
        ]
    )
    destination = ROOT / "MEMGOVERN_RESULTS.md"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(destination)
    print(f"Rendered {destination}")


if __name__ == "__main__":
    report()
