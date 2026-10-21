# Direct reranker results, October 3, 2026

These measurements use the frozen inputs and relevance definitions in [RERANKER_BENCHMARK.md](RERANKER_BENCHMARK.md). Every scored run completed 1,533 LoCoMo queries or 300 SciFact queries. The 188-query SciFact evidence view reuses the complete scientific-domain runs. Brackets are 95% bootstrap intervals; the three tables are not three independent domains. Public data and known model-selection exposure prevent unseen-data claims.

## LoCoMo original dialogue evidence

| Model | nDCG@10 | MRR | Recall@5 | p50 seconds | API $ / 1K queries |
|---|---:|---:|---:|---:|---:|
| Unreranked hybrid candidates | 0.459 [0.431, 0.486] | 0.444 | 0.522 | 0.000 | $0.000 |
| MiniLM L6 | 0.647 [0.628, 0.667] | 0.647 | 0.697 | 0.069 | Unpriced local compute |
| MiniLM L12 | 0.658 [0.638, 0.679] | 0.657 | 0.713 | 0.083 | Unpriced local compute |
| Ettin 150M | 0.713 [0.695, 0.730] | 0.712 | 0.770 | 0.115 | Unpriced local compute |
| Ettin 400M | 0.721 [0.705, 0.737] | 0.724 | 0.772 | 0.136 | Unpriced local compute |
| Qwen3 0.6B | 0.664 [0.643, 0.686] | 0.658 | 0.725 | 0.243 | Unpriced local compute |
| Voyage rerank-3 | 0.773 [0.756, 0.789] | 0.783 | 0.811 | 0.165 | $0.306 |
| Voyage rerank-3-lite | 0.758 [0.743, 0.773] | 0.766 | 0.800 | 0.172 | $0.122 |
| Cohere v4 pro | 0.750 [0.733, 0.766] | 0.758 | 0.796 | 0.324 | Unverified rate |
| Cohere v4 fast | 0.730 [0.712, 0.747] | 0.734 | 0.780 | 0.155 | Unverified rate |
| Jev 1.13 pointwise (Noul) | 0.765 [0.746, 0.784] | 0.775 | 0.808 | 1.938 | $1.800 |
| Jev 1.13 Hindsight listwise | 0.802 [0.782, 0.822] | 0.826 | 0.828 | 0.214 | $0.292 |

## SciFact explicit human evidence

| Model | nDCG@10 | MRR | Recall@5 | p50 seconds | API $ / 1K queries |
|---|---:|---:|---:|---:|---:|
| Unreranked hybrid candidates | 0.858 [0.820, 0.893] | 0.833 | 0.906 | 0.000 | $0.000 |
| MiniLM L6 | 0.872 [0.835, 0.907] | 0.851 | 0.928 | 0.094 | Unpriced local compute |
| MiniLM L12 | 0.873 [0.837, 0.906] | 0.847 | 0.943 | 0.104 | Unpriced local compute |
| Ettin 150M | 0.902 [0.867, 0.933] | 0.882 | 0.949 | 0.286 | Unpriced local compute |
| Ettin 400M | 0.913 [0.882, 0.942] | 0.893 | 0.950 | 0.513 | Unpriced local compute |
| Qwen3 0.6B | 0.921 [0.890, 0.948] | 0.906 | 0.962 | 1.294 | Unpriced local compute |
| Voyage rerank-3 | 0.932 [0.905, 0.955] | 0.913 | 0.982 | 0.319 | $1.885 |
| Voyage rerank-3-lite | 0.933 [0.907, 0.957] | 0.915 | 0.975 | 0.344 | $0.754 |
| Cohere v4 pro | 0.933 [0.905, 0.957] | 0.915 | 0.977 | 0.861 | Unverified rate |
| Cohere v4 fast | 0.926 [0.897, 0.952] | 0.914 | 0.957 | 0.431 | Unverified rate |
| Jev 1.13 pointwise (Noul) | 0.936 [0.909, 0.960] | 0.923 | 0.972 | 2.129 | $3.193 |
| Jev 1.13 Hindsight listwise | 0.932 [0.903, 0.956] | 0.920 | 0.971 | 0.631 | $2.067 |

## BEIR SciFact cited documents

| Model | nDCG@10 | MRR | Recall@5 | p50 seconds | API $ / 1K queries |
|---|---:|---:|---:|---:|---:|
| Unreranked hybrid candidates | 0.716 [0.674, 0.757] | 0.688 | 0.780 | 0.000 | $0.000 |
| MiniLM L6 | 0.689 [0.644, 0.733] | 0.665 | 0.745 | 0.094 | Unpriced local compute |
| MiniLM L12 | 0.689 [0.644, 0.733] | 0.665 | 0.748 | 0.104 | Unpriced local compute |
| Ettin 150M | 0.754 [0.712, 0.795] | 0.731 | 0.820 | 0.282 | Unpriced local compute |
| Ettin 400M | 0.768 [0.727, 0.807] | 0.743 | 0.837 | 0.505 | Unpriced local compute |
| Qwen3 0.6B | 0.782 [0.743, 0.819] | 0.752 | 0.834 | 1.280 | Unpriced local compute |
| Voyage rerank-3 | 0.808 [0.772, 0.842] | 0.775 | 0.891 | 0.313 | $1.867 |
| Voyage rerank-3-lite | 0.813 [0.777, 0.847] | 0.783 | 0.864 | 0.348 | $0.747 |
| Cohere v4 pro | 0.815 [0.779, 0.849] | 0.784 | 0.901 | 0.864 | Unverified rate |
| Cohere v4 fast | 0.793 [0.755, 0.830] | 0.766 | 0.861 | 0.432 | Unverified rate |
| Jev 1.13 pointwise (Noul) | 0.800 [0.763, 0.834] | 0.768 | 0.854 | 2.121 | $3.177 |
| Jev 1.13 Hindsight listwise | 0.819 [0.783, 0.852] | 0.790 | 0.889 | 0.630 | $2.046 |

## Paired comparisons and limits

Differences below are nDCG@10 with paired 95% bootstrap intervals, unadjusted for multiple comparisons:

- locomo, voyage-rerank-3 minus local-minilm-l6: **+0.126 [+0.114, +0.141]**.
- locomo, voyage-rerank-3 minus jev-pointwise: **+0.007 [-0.003, +0.017]**.
- locomo, jev-listwise minus jev-pointwise: **+0.037 [+0.028, +0.045]**.
- locomo, jev-listwise minus voyage-rerank-3: **+0.029 [+0.022, +0.036]**.
- scifact-evidence, jev-listwise minus jev-pointwise: **-0.004 [-0.020, +0.011]**.
- scifact-evidence, voyage-rerank-3 minus cohere-rerank-v4-pro: **-0.001 [-0.013, +0.010]**.
- scifact-evidence, voyage-rerank-3 minus voyage-rerank-3-lite: **-0.002 [-0.010, +0.006]**.

Listwise Jev reproduces Hindsight’s Choice/tournament ranking with pruning disabled, pinned to the same Jev version as pointwise. Every LoCoMo query used one Choice call; every SciFact query used two preliminary groups and one finals call. LoCoMo listwise cost $0.447 total versus $2.760 pointwise; the full 300-query SciFact run cost $0.614 versus $0.953. Listwise had one internally retried LoCoMo query (101.424 seconds including the retry), and no SciFact retries. The two ranking methods match the upstream implementation structurally. Its initial validation attempt was restarted after removing an over-strict probability-sum check; excluded preflight usage is disclosed in the interruption ledger.

These are comparisons on the published fixtures, with different passages and labels from the retired Hindsight-pipeline benchmark. They cannot establish how much of the old ranking came from score handling, fact extraction, retrieval, question filtering, or labels. Small differences among the strongest hosted models are not a basis for a general model-choice claim.

Latency covers a completed whole-query attempt, including pacing, network calls, and internal retries. Pointwise Jev sends 100 independent Noul requests per query at a 60-request/second ceiling; listwise Jev uses Hindsight Choice/tournament ranking with pruning disabled, counting usage for all preliminary and final calls; its timing reflects that adapter configuration. Local timings use one RTX PRO 5000 Blackwell GPU, unquantized bfloat16, and batches of 32. Startup, downloads, retrieval, warmup, abandoned attempts, and time between checkpoint resumes are excluded. Reported costs are list-rate estimates for successful recorded API usage, not invoices; credits, unknown failed-attempt billing, local electricity, and unverified Cohere rates are excluded.

Jev had 4 whole-query interruptions, all resolved by checkpoint resume with unchanged scoring inputs and implementation. No failed query was scored as irrelevant or omitted from the final quality denominator. See [run interruptions](../results/leaderboard/reranker-analysis/run-interruptions.json).

Independent TREC validation passed 97,152 per-query and aggregate metric comparisons across all 36 exports. The focused behavior suite passes 21 tests, and the visualizer production build includes every completed model. [Validation report](../results/leaderboard/reranker-analysis/metric-validation.json); [all paired comparisons](../results/leaderboard/reranker-analysis/paired-comparisons.json).
