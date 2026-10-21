# MemGovern embeddings and rerankers

All twelve combinations evaluate the same **9,600 queries: 200 per repository across 48 repositories**, selected by the lowest SHA256 of seed20261003, repository and original query ID. Each query searches its repository’s **entire corpus**, covering 121,475 experience cards in total. Candidate selection never uses relevance labels; no gold card is inserted into a candidate pool.

The primary numbers are repository-macro averages. Query-macro averages are identical for this balanced sample. Dense-only uses normalized embedding cosine top100. Both rerankers receive the same top100 pool from full-corpus BM25/dense RRF, with k=60 and one-based ranks. Jev uses the vendored Hindsight listwise Choice algorithm with pruning off; Ettin uses independent query/document scores.

| Embedding | Retrieval path | nDCG@10 (95% CI) | MRR@100 | Recall@5 | Recall@100 | Successful Jev API cost |
|---|---|---:|---:|---:|---:|---:|
| BGE-small (Hindsight default) | Dense only | 0.8410 [0.8231, 0.8584] | 0.8109 | 0.9071 | 0.9894 | Local compute unpriced |
| BGE-small (Hindsight default) | Hybrid + Jev listwise | 0.9277 [0.9165, 0.9384] | 0.9091 | 0.9729 | 0.9965 | $6.414 |
| BGE-small (Hindsight default) | Hybrid + Ettin 150M | 0.9435 [0.9330, 0.9533] | 0.9302 | 0.9755 | 0.9965 | Local compute unpriced |
| Granite Small English R2 | Dense only | 0.8826 [0.8664, 0.8982] | 0.8584 | 0.9335 | 0.9945 | Local compute unpriced |
| Granite Small English R2 | Hybrid + Jev listwise | 0.9270 [0.9162, 0.9375] | 0.9083 | 0.9725 | 0.9973 | $6.295 |
| Granite Small English R2 | Hybrid + Ettin 150M | 0.9434 [0.9329, 0.9531] | 0.9301 | 0.9755 | 0.9973 | Local compute unpriced |
| MongoDB Leaf IR | Dense only | 0.8863 [0.8717, 0.9003] | 0.8632 | 0.9384 | 0.9945 | Local compute unpriced |
| MongoDB Leaf IR | Hybrid + Jev listwise | 0.9275 [0.9163, 0.9385] | 0.9092 | 0.9720 | 0.9972 | $6.371 |
| MongoDB Leaf IR | Hybrid + Ettin 150M | 0.9431 [0.9326, 0.9530] | 0.9298 | 0.9752 | 0.9972 | Local compute unpriced |
| Voyage 4 Nano (1024d) | Dense only | 0.8858 [0.8703, 0.9008] | 0.8627 | 0.9401 | 0.9945 | Local compute unpriced |
| Voyage 4 Nano (1024d) | Hybrid + Jev listwise | 0.9270 [0.9153, 0.9380] | 0.9085 | 0.9715 | 0.9978 | $6.322 |
| Voyage 4 Nano (1024d) | Hybrid + Ettin 150M | 0.9435 [0.9331, 0.9533] | 0.9301 | 0.9755 | 0.9978 | Local compute unpriced |

## BM25 and reranking ablation

These additional paths use the same 9,600 sampled queries and full repository corpora. BM25 uses default BM25Okapi with lowercase regex word tokens and stable descending top100. Hybrid rows reuse the original frozen RRF pools without reranking. BM25 + Ettin uses only BM25 candidates, with no embedding retrieval.

| Retrieval path | nDCG@10 | MRR@100 | Recall@100 |
|---|---:|---:|---:|
| BM25 only | 0.8831 | 0.8601 | 0.9938 |
| BM25 + BGE-small (Hindsight default) | 0.8918 | 0.8680 | 0.9965 |
| BM25 + Granite Small English R2 | 0.9075 | 0.8853 | 0.9973 |
| BM25 + MongoDB Leaf IR | 0.9086 | 0.8875 | 0.9972 |
| BM25 + Voyage 4 Nano (1024d) | 0.9119 | 0.8914 | 0.9978 |
| BM25 + Ettin 150M | 0.9419 | 0.9289 | 0.9938 |

BM25 + Ettin minus BM25/BGE-small + Ettin is -0.0015 nDCG@10, with paired repository-bootstrap 95% CI [-0.0023, -0.0008]. BM25 improves BGE-small's initial ranking, while Ettin closes most of the remaining embedding-model differences. This is a retrieval result on MemGovern's single-positive labels, not evidence that embeddings are unnecessary for other workloads.

The ablation uses the same pinned Ettin revision, BF16, batch32, Identity activation and maximum length8192. It reuses 654,957 verified BGE-baseline query/card logits and scores 305,043 previously unseen pairs on the Blackwell. Frozen BM25 candidates, scores, complete permutations and cache-source provenance are retained. All six metrics are independently checked with pytrec_eval for every query: 57,600 comparisons for BM25 + Ettin and 288,000 for the five unreranked diagnostics. [Diagnostics](../results/experiments/memgovern/bm25-diagnostic.json), [Ettin ablation summary](../results/experiments/memgovern/bm25-ettin/summary.json), [full per-query records](../results/experiments/memgovern/bm25-ettin/records.json.gz).

The exact executed diagnostic and GPU scoring scripts are preserved as [diagnostic source](../results/experiments/memgovern/audit-sources/bm25_diagnostic.py.txt) and [Ettin ablation source](../results/experiments/memgovern/audit-sources/bm25_ettin.py.txt). They import the pinned benchmark helpers from the repository root and use the same source/work directories and model environment as the main run.

## Paired differences versus BGE-small

Differences are nDCG@10, with 5,000 paired bootstrap samples of whole repositories. Intervals are 95% percentile intervals, unadjusted for multiple comparisons. They describe uncertainty across repositories for this fixed query sample, not the additional variation from drawing another sample of queries. Resampling units are the official repository corpora; historical names such as the two Airflow and two Home Assistant banks represent related projects, so these intervals should not be read as 48 fully independent project samples.

| Embedding minus BGE-small | Retrieval path | nDCG@10 difference (95% CI) |
|---|---|---:|
| Granite Small English R2 | Dense only | +0.0416 [+0.0365, +0.0466] |
| MongoDB Leaf IR | Dense only | +0.0452 [+0.0389, +0.0515] |
| Voyage 4 Nano (1024d) | Dense only | +0.0448 [+0.0400, +0.0497] |
| Granite Small English R2 | Hybrid + Jev listwise | -0.0007 [-0.0028, +0.0014] |
| MongoDB Leaf IR | Hybrid + Jev listwise | -0.0002 [-0.0023, +0.0020] |
| Voyage 4 Nano (1024d) | Hybrid + Jev listwise | -0.0007 [-0.0027, +0.0013] |
| Granite Small English R2 | Hybrid + Ettin 150M | -0.0001 [-0.0005, +0.0003] |
| MongoDB Leaf IR | Hybrid + Ettin 150M | -0.0004 [-0.0009, +0.0000] |
| Voyage 4 Nano (1024d) | Hybrid + Ettin 150M | -0.0000 [-0.0004, +0.0004] |

## CPU and A4000 agreement check

The same 64 Granite queries were independently reencoded on CPU and compared with their actual A4000 checkpoints. Maximum absolute vector difference was 2.38e-07; minimum embedding cosine agreement was 1.000000000000. Both devices produced unit-normalized vectors. This sampled check supports numerical agreement of the device continuation; it does not prove every corpus ranking is identical. [Query IDs, model/device provenance and measurements](../results/experiments/memgovern/a4000-cpu-agreement.json).

## Pointwise cache and numerical sensitivity

The pre-cache runs independently scored 473,688 shared Granite/BGE pairs on the Blackwell and 307,773 shared Leaf/BGE pairs across A4000/Blackwell. Replacing shared logits with their BGE baseline values would change mean nDCG@10 by +0.000000 for 6,185 Granite queries and +0.000214 for 4,001 Leaf queries. Granite shared logits were almost bit-identical; Leaf’s cross-device logits had p99 absolute difference 0.0625, maximum 0.6250, and 16 top1 changes. These are BF16 batch/device effects, so tiny reranked differences should not be treated as model superiority. Actual completed rankings are retained; the remaining computations record explicit cache provenance. [Audit measurements](../results/experiments/memgovern/ettin-pointwise-cache-audit.json).

## What these labels establish

MemGovern’s public qrels label one source experience card per original bug query. The task measures recovery of the card distilled from that bug’s fix. Other useful or semantically equivalent cards receive no credit. This does not directly measure whether retrieved experiences help solve a different bug, nor agent task success. Models may have encountered the underlying public repositories or benchmark data during training; this experiment does not test contamination.

The source is [KaLM-Embedding/LMEB](https://huggingface.co/datasets/KaLM-Embedding/LMEB), pinned to `f9d4294be4a24b8c16d6bfb59d56a6fec4bd95c0`. The official per-repository corpus boundaries and positive associations are retained. Queries are sampled independently of text, labels and model scores, without filtering hard or unsuccessful queries. [Source hashes and counts](../results/experiments/memgovern/source.json).

## Models and execution

BGE-small uses 384 dimensions and empty query/document prompts, matching the current Hindsight default. Granite uses 384 dimensions, CLS pooling and empty prompts. Leaf uses its full native 768 dimensions and documented retrieval query prefix; documents are unprefixed. Voyage Nano uses its documented distinct query/document prompts and truncates to 1024 dimensions before normalization. All embeddings use float32 SDPA and unit normalization. All model revisions and prompts are recorded in each frozen pool’s encoding metadata; none are quantized.

Nano runs on the RTX PRO 5000 Blackwell first. BGE-small, Granite and Leaf initially run concurrently on CPU, each with eight worker processes and four Torch threads per worker. Completed CPU vectors are retained; remaining embeddings are then computed on the idle RTX A4000, one model at a time, with the same float32 weights, prompts and normalization. Each GPU checkpoint records its actual execution metadata separately from the initial CPU configuration, and newly frozen pools retain that provenance. Ettin uses disjoint model shards: Leaf’s candidates are reranked on the A4000 after embedding preparation finishes, and the other three models on the Blackwell after Nano finishes, using pinned revision `025501c4e0f9bbeb4c5b198318e0089ff061cc14`, BF16 weights, batch32, native maximum context8192 and Identity activation. Hosted Jev1.13 runs concurrently with eight independent query workers, each paced at two requests per second. Repeated Ettin query/card pairs reuse verified raw logits from the completed BGE-small run; only previously unseen pairs require new inference. Cache reuse checks query text, source identity, scorer configuration and environment, and exported cached scores are checked against their baseline input indices. Records preserve cached indices and source-run hashes, so reused Blackwell logits remain distinguishable from fresh A4000 computation. Concurrent timings measure incremental work and are not isolated full-rerank speed benchmarks. Leaf’s initial worker pool failed before producing a checkpoint; its identical restart supplied the measured vectors. CPU preparation was later resumed with TOKENIZERS_PARALLELISM=false to prevent each worker spawning a full-host tokenizer thread pool; completed embeddings were reused and model/tokenization settings were unchanged.

Total successful Jev input-token cost across four embeddings is **$25.403**, at $0.042 per million input tokens. Output tokens are uncharged. Costs exclude warmup and unknown billing for failed requests/retries; local electricity and hardware are unpriced. Raw successful usage and retry counts are retained. Jev’s listwise algorithm may split oversized pools and use a final selection round; candidate pruning remains off and all100 candidates remain in the final ranking.

## Validation and reproduction

The exporter requires all 9,600 queries in all twelve combinations, exact source/pool/run provenance, finite scores, complete candidate permutations and exact provenance/value equality for reused pointwise scores. It independently recomputes nDCG@10, MRR@100 and Recall@1/5/10/100 with pytrec_eval: **691,200 metric comparisons**. Retrieval misses count as zero; incomplete runs produce no completed summary. [Summaries, paired intervals, frozen pools and full per-query records](../results/experiments/memgovern/summary.json).

Download `Procedural/MemGovern/<repository>/{corpus.jsonl,queries.jsonl,qrels.tsv}` from the pinned dataset revision into `<data>/Procedural/MemGovern/`, and put the revision in `<data>/revision.txt`. Use the package versions recorded in pool metadata and cached pinned models. From the repository root:

    python benchmark-runner/memgovern_prepare.py --embedding voyage-1024 --device cuda:0 --threads 8 --data <data> --work <work>
    python benchmark-runner/memgovern_prepare.py --embedding bge-small --data <data> --work <work>
    python benchmark-runner/memgovern_prepare.py --embedding granite --data <data> --work <work>
    python benchmark-runner/memgovern_prepare.py --embedding leaf --data <data> --work <work>
    python benchmark-runner/memgovern_gpu_prepare.py --embedding granite --data <data> --work <work>
    python benchmark-runner/memgovern_gpu_prepare.py --embedding bge-small --data <data> --work <work>
    python benchmark-runner/memgovern_gpu_prepare.py --embedding leaf --data <data> --work <work>
    python benchmark-runner/memgovern_experiment.py freeze --data <data> --work <work>
    python benchmark-runner/memgovern_experiment.py rerank --reranker jev-listwise --data <data> --work <work>
    python benchmark-runner/memgovern_experiment.py rerank --reranker ettin-150m --data <data> --work <work>
    python benchmark-runner/memgovern_experiment.py export --data <data> --work <work>
    python benchmark-runner/memgovern_report.py

For CPU preparation, set CUDA_VISIBLE_DEVICES to empty, OMP_NUM_THREADS=4, OPENBLAS_NUM_THREADS=1 and TOKENIZERS_PARALLELISM=false. The three CPU commands can run concurrently; preparation checkpoints are reused only when their exact source/model/configuration fingerprints match. Model files are loaded from the local cache. For Nano, use the existing isolated Transformers4.57.6 compatibility overlay and the explicit Qwen3Config registration shim; forward computation and weights are unchanged. Other models and rerankers use Transformers5.18.0. Jev reads the authorized key from `~/.secrets/typesafe_api_key`. The GPU continuation commands require existing initial preparation metadata, accept hardware/scheduling changes, reject changed source/model/prompt/precision settings, and retain completed checkpoints. Select the A4000 by CUDA_VISIBLE_DEVICES for these continuation commands, and the Blackwell for Ettin. Source corpora and large embedding arrays are not duplicated into Git; the source manifest, frozen pools and records permit audit/reproduction.

No production defaults, embedding stores or live installations changed.
