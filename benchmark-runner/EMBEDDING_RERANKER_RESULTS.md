# LoCoMo embeddings with Jev listwise and Ettin 150M

All twelve combinations use the same 1,533 LoCoMo questions, source evidence IDs, and complete corpus. Each embedding model supplies dense retrieval; its ranking is fused with unchanged BM25 using 1/(60+rank), then the top 100 candidates are reranked. Gold evidence is never inserted into the candidate pool. These are retrieval-plus-reranking comparisons: the candidate pools differ across embedding models.

| Embedding | Reranker | nDCG@10 (95% CI) | MRR | Recall@5 | Jev API cost, all queries |
|---|---|---:|---:|---:|---:|
| BGE-small-en-v1.5 (current default) | Jev 1.13 listwise | 0.802 [0.782, 0.822] | 0.822 | 0.834 | $0.445 |
| BGE-small-en-v1.5 (current default) | Ettin 150M | 0.711 [0.693, 0.729] | 0.712 | 0.769 | Local compute unpriced |
| Granite Small English R2 | Jev 1.13 listwise | 0.787 [0.765, 0.810] | 0.812 | 0.811 | $0.437 |
| Granite Small English R2 | Ettin 150M | 0.699 [0.681, 0.719] | 0.701 | 0.756 | Local compute unpriced |
| MongoDB Leaf IR | Jev 1.13 listwise | 0.809 [0.784, 0.834] | 0.829 | 0.839 | $0.460 |
| MongoDB Leaf IR | Ettin 150M | 0.720 [0.700, 0.739] | 0.718 | 0.778 | Local compute unpriced |
| Voyage 4 Nano (1024d) | Jev 1.13 listwise | 0.841 [0.822, 0.860] | 0.854 | 0.871 | $0.461 |
| Voyage 4 Nano (1024d) | Ettin 150M | 0.733 [0.717, 0.748] | 0.727 | 0.795 | Local compute unpriced |
| BGE-base-en-v1.5 (historical) | Jev 1.13 listwise | 0.802 [0.782, 0.822] | 0.826 | 0.828 | $0.447 |
| BGE-base-en-v1.5 (historical) | Ettin 150M | 0.713 [0.695, 0.730] | 0.712 | 0.770 | Local compute unpriced |
| Voyage 4 Nano (2048d, historical) | Jev 1.13 listwise | 0.839 [0.818, 0.858] | 0.851 | 0.870 | $0.463 |
| Voyage 4 Nano (2048d, historical) | Ettin 150M | 0.735 [0.718, 0.750] | 0.729 | 0.796 | Local compute unpriced |

## Retrieval before reranking

Top-100 evidence recall is the mean fraction of annotated evidence turns present in the candidate pool. It is the upper bound on evidence recall for either reranker on that pool.

| Embedding | Unreranked hybrid nDCG@10 | Top-100 evidence recall |
|---|---:|---:|
| BGE-small-en-v1.5 (current default) | 0.451 | 0.879 |
| Granite Small English R2 | 0.445 | 0.852 |
| MongoDB Leaf IR | 0.493 | 0.886 |
| Voyage 4 Nano (1024d) | 0.537 | 0.934 |
| BGE-base-en-v1.5 (historical) | 0.459 | 0.876 |
| Voyage 4 Nano (2048d, historical) | 0.539 | 0.933 |

## Paired embedding comparisons

Differences below are nDCG@10, using 5,000 paired bootstrap samples of whole conversations. Intervals are 95% percentile intervals, unadjusted for multiple comparisons. Only ten conversations underlie these estimates.

- jev-listwise, voyage-1024 minus BGE-small: **+0.040 [+0.030, +0.050]**.
- jev-listwise, granite minus BGE-small: **-0.015 [-0.028, -0.001]**.
- jev-listwise, leaf minus BGE-small: **+0.007 [-0.003, +0.017]**.
- ettin-150m, voyage-1024 minus BGE-small: **+0.022 [+0.014, +0.031]**.
- ettin-150m, granite minus BGE-small: **-0.012 [-0.022, -0.002]**.
- ettin-150m, leaf minus BGE-small: **+0.0090 [+0.0001, +0.0188]** (lower endpoint rounds from +0.0000757; borderline exclusion of zero).

## CPU inference probe

A fixed sample of 256 corpus documents and 64 queries was encoded in separate fresh CPU processes, with float32 SDPA, four Torch threads, batch size 32 for documents, three document passes, and individually timed queries after warmup. All models use identical source texts, their native maximum context and intended prompts; embeddings are unit normalized. These are indicative measurements on a shared host with concurrent work, not isolated speed claims. Peak process RSS includes Python/Torch and model loading.

| Model | Parameters | Dimensions | Documents/sec (median pass) | Single-query p50 / p95 (ms) | Peak process RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| BGE-small-en-v1.5 (current default) | 33.4M | 384 | 50.7 | 14.49 / 19.52 | 842.8 |
| Granite Small English R2 | 47.7M | 384 | 35.5 | 15.92 / 21.75 | 1016.8 |
| MongoDB Leaf IR | 22.9M | 768 | 62.1 | 11.28 / 20.25 | 811.7 |

CPU: AMD Ryzen Threadripper PRO 9985WX 64-Cores. Reproduce with `python benchmark-runner/cpu_embedding_probe.py --embedding <bge-small|granite|leaf> --output <file.json> --threads 4`. [Raw measurements](../results/experiments/embedding-reranker-locomo/cpu-probe/). Leaf’s native vectors have 768 dimensions, unlike BGE-small/Granite at 384; changing an existing store requires an explicit reembedding/migration strategy. No production defaults changed.

## Granite correctness audit

An independent CPU float32 audit used Hugging Face AutoModel directly, CLS pooling from the pinned model configuration, L2 normalization, and an independently implemented BM25/RRF merge. All 150 queries in conv-26 reproduced the frozen GPU top100 candidate sets exactly; 149 also reproduced their ordering exactly. Query and document vector norms were 0.99999988–1.00000012. Manual embeddings differed from Sentence Transformers by at most 5.6e-8. Native query/document prompts are empty, consistent with IBM’s documented usage. This sampled audit finds no normalization, pooling or prompt omission; it does not establish general model superiority. IBM’s [evaluation table](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2#evaluation-results) reports the same MTEB-v2 Retrieval score (53.9) for Granite Small R2 and BGE-small, despite Granite’s higher overall average across other task families. [Audit measurements](../results/experiments/embedding-reranker-locomo/granite-normalization-audit.json) and [exact executed audit source](../results/experiments/embedding-reranker-locomo/audit-sources/granite_normalization.py.txt).

## Models, execution and reproduction

Leaf uses `MongoDB/mdbr-leaf-ir` revision `4262131b32c3182bd06e67e92ae69d7bd66e0c5c`, full native 768 dimensions, 512-token context, mean pooling followed by the trained 384-to-768 projection and normalization. Its documented query prefix is `Represent this sentence for searching relevant passages: `; documents are unprefixed. This is the standard symmetric Leaf encoder, not the optional asymmetric larger Snowflake document model. No quantization or MRL truncation was applied. Native Sentence Transformers inference requires no remote code and works with Transformers 5.18.0. [Official Leaf model card](https://huggingface.co/MongoDB/mdbr-leaf-ir).

BGE-small uses pinned revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, 384 dimensions and no added prompt, matching the current Hindsight local embedding default. Voyage 1024 uses the same pinned official model and prompts as the 2048 run, truncating to 1024 dimensions before L2 normalization. Granite results are reused without new inference. BGE-base and Voyage 2048 remain as historical comparisons, and are not the production-default baseline.

BGE-base reuses the exact already published candidate fixture and two completed reranker runs, preserving their provenance and sequential timing. No historical BGE-base reranker API calls were repeated. Its revision and query prefix remain in [the original fixture](datasets/reranker/locomo10-top100.json.gz).

The historical Voyage 2048 run uses `voyageai/voyage-4-nano` revision `67fabc9bef010dabc5f6024aa1b1b6b93410426f`, full 2048-dimensional embeddings, and the documented distinct query/document prompts. Granite uses `ibm-granite/granite-embedding-small-english-r2` revision `2ab6fa8ea2d674564defd37171ae19079b864b33`, 384 dimensions and no additional prompt. New embeddings use float32, cosine normalization, batch size 32, SDPA attention and each model’s native maximum context. Preparation is local on the RTX PRO 5000 Blackwell. Official settings: [Voyage model card](https://huggingface.co/voyageai/voyage-4-nano), [Granite model card](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2).

Voyage’s pinned custom code needs the Transformers 4 attention API. Its isolated preparation environment uses Transformers 4.57.6, tokenizers 0.22.2, huggingface-hub 0.36.2 and requests 2.32.5, with the same PyTorch and Sentence Transformers as the existing environment. The adapter explicitly assigns Qwen3Config as the custom model’s config_class for AutoModel registration; weights and forward computation are unchanged. Granite preparation uses Transformers 5.18.0. All core versions and separate preparation hashes are recorded in the fixtures. Historical Granite source is preserved at commit 47534ba, and Voyage/concurrent inference at commit 13d6f2a. New BGE-small and Voyage1024 preparation and hosted adapter hashes are recorded in their fixtures and run metadata. Rerankers retain the original environment and models.

Jev is pinned to jev-1.13.0 and uses the existing Hindsight Choice/tournament ranking, pruning disabled. New Jev runs use three concurrent query workers, each owning a separate HTTP client, usage/retry counters, an eight-request concurrency limit and 10 requests/second pacing. One writer records completed queries. Warmup is excluded; interrupted queries cannot silently receive relevance zero. GPU preparation and Ettin work overlap hosted queries. New timings are measurements on a shared host with concurrent work and cannot be used as a controlled latency comparison with sequential BGE runs; no speed claim is made. Ettin uses the same pinned checkpoint, unquantized bfloat16 and batches of 32 as before.

The Granite Ettin process encountered a CUDA unknown error after 463 queries and resumed from its checkpoint with identical metadata on the same GPU. No query was dropped. The [interruption ledger](../results/experiments/embedding-reranker-locomo/run-interruptions.json) preserves the failure and resolution. Jev monetary values use successful recorded input tokens at the previously verified $0.042/million rate; output is free. Warmup, unknown failed/retried request billing and local compute costs are excluded.

Reproduce preparation with `python benchmark-runner/embedding_experiment.py prepare --embedding voyage` (using the isolated compatibility environment) or `--embedding voyage-1024` with the same environment. BGE-small, Granite and Leaf preparation use `--embedding bge-small`, `--embedding granite` or `--embedding leaf` in the main environment. Select Blackwell with `CUDA_VISIBLE_DEVICES=GPU-27368065-5d8b-6e5e-4e81-2e924bd9ce73`. Do not overwrite frozen fixtures. Score Jev with `embedding_experiment.py hosted --embedding <name> --workers 3 --records <run-root>/<name> --secrets-dir ~/.secrets`; score Ettin with `run_all_reranker.py run --fixture results/experiments/embedding-reranker-locomo/<name>/fixture.json.gz --model ettin-150m --output <run-root>/<name>`. Export with `embedding_experiment.py export --records <run-root>`. Dependencies are the existing pinned runner environment plus pytrec-eval-terrier 0.5.10 for independent validation.

Validation: **24 focused behavior tests pass**, Ruff and diff checks pass, and **73,632 independent TREC metric comparisons** match all twelve complete results. [Frozen fixtures, aggregates and raw rankings](../results/experiments/embedding-reranker-locomo/); [all paired comparisons](../results/experiments/embedding-reranker-locomo/paired-comparisons.json).

The original LoCoMo labels remain incomplete, and public benchmark exposure is unknown. This is not a clean holdout or end-to-end Hindsight evaluation. Changing retrieval also changes the candidate order and decoys seen by Jev’s listwise classifier, so differences do not isolate embedding quality from reranker interactions.
