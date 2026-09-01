# Reranker distillation — spec and review notes

Training a small, fast, multilingual, long-context reranker to replace
`BAAI/bge-reranker-base` in Hindsight. This file is the working spec: what we
are building, why, what a review found wrong, and what is still open.

---

## 1. Why

Measured on one L4, TEI 1.8.3, fp16, two independent ground truths:

| Model | Ctx | Fact MRR | Chunk MRR | chunk pairs/s |
|---|---|---|---|---|
| bge-reranker-base **(production)** | 512 | 0.8550 | **0.2764** | 435 |
| ms-marco-MiniLM-L6 | 512 | 0.8587 | 0.2545 | 1420 |
| gte-reranker-modernbert-base (EN only) | 8192 | 0.8971 | 0.5666 | 270 |
| gte-multilingual-reranker-base | 8192 | 0.8602 | 0.5771 | 225 |
| bge-reranker-v2-m3 | 8192 | 0.9036 | 0.5938 | 150 |

The split is on **context length, not model quality**: every 512-token model
lands at 0.25–0.31 on chunks, every 8192-token model at 0.57–0.59. Hindsight's
`DEFAULT_RETAIN_CHUNK_SIZE` is 3000 chars (~750 tokens), so production truncates
about a third of every full chunk.

Every existing long-context option is 1.6–2.9x SLOWER than production. We want
long context AND multilingual AND faster. Nothing off the shelf does all three.

## 2. What we are building

- **Student:** `jhu-clsp/mmBERT-small` — ModernBERT, 22 layers x 384 hidden,
  140M total but **42M non-embedding** (256k Gemma-2 vocab), 8192 ctx via RoPE,
  1800+ languages. It is a masked LM with **no ranking ability at all**.
- **Teacher:** `BAAI/bge-reranker-v2-m3` — best on our chunk data (0.5938).
- **Loss:** MarginMSE on the teacher's score margin.
- **Stage 1:** general relevance, public IR triples.
- **Stage 2:** domain adaptation, synthetic conversational-memory docs at
  3000 chars, English + Chinese.
- **Serving:** must load in TEI so it is benchmarked on the identical harness
  as every other model.

## 3. Leakage policy — enforced in code

Never train on:

| Corpus | Why |
|---|---|
| LoCoMo, LongMemEval, BEAM | our evaluation benchmarks |
| **ShareGPT, UltraChat** | LongMemEval's chat histories are built FROM them |
| MSC | same genre and lineage, not worth the risk |

`assert_not_blocked()` runs on every dataset id and every provenance string, and
every emitted row carries `provenance` so the set can be audited after the fact.
Teacher check: bge-reranker-v2-m3 trained on MS MARCO, HotpotQA, TriviaQA, NQ,
COLIEE, PubMedQA, SQuAD, Mr.TyDi, MIRACL, MLDR, DuReader, T2-Ranking, CMedQAv2,
LeCaRDv2 — none is one of our evals. The teacher is clean.

## 4. Review findings — fix before the large run

Ranked by impact. (1) and (2) are bugs, not preferences.

### 4.1 Teacher scores were sigmoid-squashed — CONFIRMED BUG

`CrossEncoder.predict` applies `nn.Sigmoid()` by default when `num_labels == 1`.
We called it with no override, so every MarginMSE label was
`sigmoid(pos) - sigmoid(neg)`. Measured on our actual teacher:

```
DEFAULT (what v0 trained on):  [0.2919, 0.0000]  -> margin  0.29
RAW LOGITS:                    [-0.8862, -11.0156] -> margin 10.13
```

The irrelevant document squashed to **exactly 0.0**. Resolution collapses
precisely at the confident end, which is where MarginMSE gets its signal. v0
trained on margins compressed ~35x.

**Fix:** score with `activation_fn=torch.nn.Identity()`. Must land before
generating data, not after. Both Hofstätter's original MarginMSE and the Ettin
rerankers distil raw logits.

### 4.2 Streaming a prefix of `-all` configs collapses query diversity

The `-all` triplet configs enumerate every positive/negative combination grouped
by query, so a prefix returns all combinations of a few hundred queries:

| config | rows | unique queries | rows/query |
|---|---|---|---|
| miracl zh-triplet-all | 318,700 | 1,312 | ~243 |
| miracl en-triplet-all | 789,900 | 2,863 | ~276 |
| hotpotqa triplet-all | 3,381,494 | 84,516 | ~40 |

At the planned 2M, the zh target (600k) exceeds the whole zh config, so Chinese
would be ~1,300 unique queries repeated hundreds of times — not the "native
Chinese ranking data" the comment claims.

**Fix:** cap rows per query (4–8) and shuffle instead of taking a prefix. Add a
real Chinese source at scale: T2Ranking (300k+ human-annotated Chinese ranking
queries) or DuReader-retrieval.

### 4.3 Chinese stage-2 docs overflow the teacher window

3000 Chinese characters is ~2000+ tokens, but the teacher scores at
`max_length=1024`. So Chinese labels and mined negatives are computed on the
front half of each document — reproducing, inside our label pipeline, the exact
truncation failure this project exists to fix. English (~750 tokens) is fine.

**Fix:** teacher `max_length` >= 2560 for stage 2; `--stage2-maxlen` 2048–2560;
print a per-language token histogram before training.

### 4.4 Eval config burns ~1/3 of the budget and nothing is checkpointed

1% of 2M = 20k held out, evaluated every 200 steps over ~31k steps means ~155
evals and roughly 50% wall-clock overhead. `save_strategy="no"` means a crash at
hour 2.5 loses everything.

**Fix:** eval on 2–5k rows every 2000–5000 steps; checkpoint every ~5k steps;
prefer a ranking metric (`CrossEncoderNanoBEIREvaluator`) over eval loss, with
`load_best_model_at_end`.

### 4.5 Negative mining selects for false negatives, and wastes the pool

Taking the teacher's top-scoring non-positive from 24 random docs is the policy
*most likely* to pick a document that genuinely answers the query — with 1500
docs over 60 topics, ~25 docs share a topic and recall-style queries are often
generic.

**Fix (NV-Retriever style):** score each query against the full per-language
pool, drop candidates scoring above ~95% of the positive, then take 4–8
negatives from ranks ~2–30. This also grows stage 2 from ~18k to ~70–140k
triples for free. MarginMSE partially self-corrects surviving false negatives
(margin ~0 gives a weak gradient rather than a wrong one), which is a real
advantage of this loss, but the filter is cheap.

### 4.6 Stage 2 may regress short-fact reranking

Stage 2 is exclusively 3000-char transcripts, but production also reranks short
facts. Mix 5–10% replayed stage-1 rows into stage 2 (rehearsal) and measure
facts MRR before and after.

### 4.7 Synthetic data failure modes

- **Lexically easy positives** — "questions this excerpt answers" produces
  queries quoting the doc. Require paraphrase, ban verbatim phrases, add
  temporal displacement ("what did X say last month about…").
- **Entity mode collapse** — Gemini reuses a small name set, causing cross-doc
  collisions. Seed each prompt with explicitly sampled names/places/dates.
- **Topic collisions** — 25 docs per topic drives false negatives; expand topics
  or add a sampled sub-facet per document.
- **Format mismatch** — real Hindsight chunks carry speaker tags and timestamps;
  format a share of synthetic docs that way.

### 4.8 Smaller notes

- `_row_to_triple` is dead code (no call site).
- `--stage1-per-lang` is actually a *total* target (shares sum to 0.7, MS MARCO
  fills the remainder). Rename or fix the accounting.

## 5. Settled decisions

- **MarginMSE stays.** A 2026 reproduction study puts pairwise MarginMSE and
  listwise InfoNCE in the top tier, above pointwise and above LLM-listwise
  objectives. Distillation beats contrastive learning exactly when the teacher
  is stronger than the student — 568M teacher, 42M non-embedding student.
- **Hyperparameters are inside the published envelope.** Ettin: 68M → lr 3e-5 at
  batch 256; 150M → 1.5e-5 at batch 192; 1 epoch, 3% warmup, bf16. Ours (lr
  2e-5, batch 64, 1 epoch, 10% warmup, bf16) is fine. Raise stage-2 batch from
  16 to 32–64. If time is tight, batch 128 + lr 3e-5 beats cutting steps.
- **2M stage-1 triples is enough** for our bar (a domain benchmark, not BEIR).
  Ettin's 143M is a general-purpose SOTA budget; do not copy it.
- **One teacher across both stages.** Never mix teachers inside a MarginMSE stage.
- **No RoPE extrapolation risk:** mmBERT ran a context-extension phase at 8192
  during pretraining. Whether the *ranking head* stays accurate at unseen
  lengths is unproven, so stage-2 length must cover real serving lengths and the
  chunk benchmark must run at full length.

## 6. TEI serving gotchas (all hit, all fixed in `make_tei_servable`)

1. `sentence_bert_config.json` ships without `max_seq_length` → TEI refuses to boot.
2. transformers 5.x rewrites ModernBERT RoPE into nested `rope_parameters`;
   TEI 1.8.3 reads flat `global_rope_theta` / `local_rope_theta` → "Model is not
   supported".
3. sentence-transformers 6.x writes module paths TEI cannot parse (non-fatal).
4. **Image must match compute capability**: A100 is cc 8.0 → tag `1.8.3`;
   L4 is cc 8.9 → tag `89-1.8.3`. Mismatch fails with "Runtime compute cap 80 is
   not compatible with compile time compute cap 89".

**Benchmark the student on the L4**, never the A100 — every comparison model was
measured on an L4 and cross-hardware numbers are not comparable.

## 7. Deliverables

- Model weights and all checkpoints, downloaded locally (VMs are ephemeral).
- Training data and eval results, downloaded locally.
- A plain-English model card: base model, datasets, evals, how it was made.
- Our own benchmark (fact + chunk, EN + ZH).
- **A broader general-reranker eval** on datasets we did NOT train on — SciFact,
  NFCorpus, TREC-COVID (not HotpotQA or MS MARCO, which are in our mix).

---

## 8. Results — student v0 (first run, trained WITH the bugs below active)

Measured on one L4, TEI 1.8.3, same bank and ground truth as the five baselines.

| Model | Fact MRR | Chunk MRR | chunk pairs/s | Ctx | Multiling |
|---|---|---|---|---|---|
| bge-reranker-v2-m3 | 0.9036 | 0.5938 | 150 | 8192 | yes |
| gte-reranker-modernbert | 0.8971 | 0.5666 | 270 | 8192 | **no** |
| **student v0** | **0.8664** | **0.5558** | **435** | 8192 | yes |
| gte-multilingual | 0.8602 | 0.5771 | 225 | 8192 | yes |
| ms-marco-MiniLM-L6 | 0.8587 | 0.2545 | 1420 | 512 | **no** |
| **bge-reranker-base (production)** | 0.8550 | 0.2764 | 435 | 512 | yes |

Against production at **identical throughput (435 pairs/s)**: chunk MRR +101%,
fact MRR +1.3%, plus 8192 context and 1800 languages. It is also the fastest
long-context model by 1.6x and the only one not slower than what we run today.

v0 was trained with the sigmoid bug (margins compressed ~35x), an English-only
stage 1, 200k triples, one unfiltered negative per query, and the inverted
false-negative ceiling. All fixed for v1.

## 9. Review log — five adversarial rounds

| Round | Findings |
|---|---|
| 1 | sigmoid-squashed teacher labels; `-all` prefix collapsed query diversity; MIRACL cannot carry Chinese; teacher truncating zh labels at 1024; quadratic mining; 50% eval overhead with no checkpointing |
| 2 | false-negative ceiling **inverted for negative logits**; Chinese silently collapsing to 15% via English backfill |
| 3 | `except Exception` defeated the fail-fast; `--skip-stage1` trained the raw MLM; eval holdout leaked queries; local checkpoints could not be served by TEI |
| 4 | the parity check itself would fail healthy models (f32 sigmoid ties at ~-89, not -17); ground truth could be silently re-annotated, voiding the comparison |
| 5 | **clean** — "no material correctness bug remains, safe to run" |

Roughly a third of all findings were bugs in code written to *check* other code.
The fail-fast guard would have hard-failed its own run; the parity gate would
have rejected a healthy checkpoint. Verification code needs verifying.

## 10. Open items, accepted for this run

- SPEC 4.7 synthetic-prompt hardening (paraphrase requirement, entity seeding,
  speaker-tag formatting) is not implemented.
- No per-language token histogram before training.
- Stage-1 teacher labels at 1024 while the student trains at 512, so long
  passages carry label signal the student cannot see. MarginMSE tolerates it.
- No `resume_from_checkpoint` flag; restart after a crash is manual.
- `--stage1-per-lang` is a total, not per-language.
- Quality eval is English-only. Chinese quality is UNMEASURED — the largest
  remaining measurement hole, and the one to close next.

---

# v2 — the 6-layer student

## 11. Why depth, and why now

v1 hit 435 pairs/s, exactly the throughput of the `bge-reranker-base` it would
replace, and roughly a third of the open-source default `ms-marco-MiniLM-L-6-v2`
at 1420 pairs/s. Nicolo's ordering after the 2026-08-28 call is throughput
first, quality second: a fast model with adequate quality ships, a good model
with no throughput does not.

The bet that 6 layers is enough rests on one comparison, on facts, where context
length is not a factor:

| Model | Layers | Fact MRR |
|---|---|---|
| hindsight-v1 | 22 | 0.8698 |
| ms-marco-MiniLM-L6 | 6 | 0.8587 |

Six layers is 1.3% behind twenty-two. The whole 0.2545 vs 0.5724 chunk gap is
MiniLM's 512-token context, not its depth. So: keep 8192 context, cut depth.

**The risk this rests on.** That comparison is at 512 tokens. It says nothing
about whether 6 layers suffice for long-range reasoning at 2048. If v2 holds
fact MRR but drops chunk MRR, depth was load-bearing after all and the speed has
to come from quantization instead.

No teacher inference: the labels in `stage1.jsonl` and `stage2.jsonl` are raw
teacher logits already, and they are model-agnostic. v1's 900k scorings and 2.5h
of negative mining carry over unchanged.

## 12. Layer selection

Keep source layers **[0, 4, 8, 12, 16, 20]** of 22. Even spacing, and every kept
layer's global/sliding role at its new index equals its role at its old one.
`select_layers()` derives this: a stride coprime to `global_attn_every_n_layers`
is invertible modulo it, so `(i*stride) % g == 0` exactly when `i % g == 0`.
Stride 3 would have selected only global layers and then assigned four of six a
local role.

## 13. What the smoke test found in transformers 5.16.1

The first prune was written against a ModernBERT layout that no longer exists.
Roles are **not** baked onto the attention module as `local_attention`. They come
from `config.layer_types`, a list of `"full_attention"` / `"sliding_attention"`
indexed by `layer_idx`, resolved at construction into `layer.attention_type` and
`attn.sliding_window` (None or 65).

Three defects followed, all in the prune:

1. **`config.layer_types` was not truncated.** A 6-layer model saved with a
   22-entry list. On reload every layer's role is read by position in the old
   list. Our selection happens to make the first six entries correct, so this
   would have round-tripped *by luck*; any other selection silently serves a
   different model than it trained.
2. **`layer_idx` left stale.** New position 1 still claimed index 4. Any path
   re-deriving from `layer_idx` reads the wrong entry or indexes off the end of
   the shortened list.
3. **Source layer 0 is special.** `ModernBertEncoderLayer.__init__` gives
   `layer_idx == 0` an `nn.Identity()` for `attn_norm` where every other layer
   gets a LayerNorm. A selection not starting at 0 trains with a LayerNorm in
   position 0 and reloads with an Identity, dropping a normalisation and its
   weights. Now a hard error.

Rewriting `layer_types` demotes the coprime-stride rule from a correctness
requirement to a quality one: any selection now reloads as what it trained as,
but preserving each layer's original receptive field keeps every block doing the
job it was pretrained for.

The reported role, before the fix, came from
`getattr(l.attn, "local_attention", None)` — a default that returned `None` for
every layer and printed a confident, empty report. Sixth round, and the finding
is again in the code written to check the code.

## 14. Measured cost of the depth cut

Probed on the A100 before launching, batch 64 at a 512-token window:

| | steps/s | peak mem | stage 1 projected |
|---|---|---|---|
| 22-layer (v1) | 2.04 | 30.6 GB | 115 min |
| 6-layer (v2) | 5.31 | 9.3 GB | 44 min |

The 22-layer projection lands next to v1's actual 108 minutes, which validates
the probe method.

Stage-2 shape, 6-layer at a 2048-token window:

| batch | steps/s | peak mem | epoch wall time |
|---|---|---|---|
| 16 | 7.09 | 7.9 GB | 9.5 min |
| 32 | 3.58 | 13.4 GB | 9.4 min |
| 48 | 2.53 | 18.9 GB | 8.9 min |
| 64 | 1.97 | 24.4 GB | 8.6 min |

**Wall time is flat across batch size.** The GPU is saturated at every setting,
so the memory freed by pruning buys under a minute. Stage 2 therefore keeps v1's
batch 16 / lr 1e-5 / 1 epoch / 2048 and depth stays the only variable. Round 1
separately caught that the file's DEFAULTS (epochs 2.0, batch 32, maxlen 2560)
differ from what v1 actually ran on three axes, so every value is pinned on the
command line.

## 15. Attention budget, and the hedge

Preserving the 1-in-3 rhythm at 6 layers leaves **2 full-attention layers**, at
new indices 0 and 3. v1 had 8 of 22. `ms-marco-MiniLM-L-6-v2`, the entire basis
for "6 layers is enough", has 6 layers of FULL attention and was measured only on
short documents. So the planned config departs from that evidence on both axes
at once, sparse attention and long documents, and new layer 0 is the
embedding-adjacent layer with the Identity attn_norm, leaving effectively one
deep global mix for a 2048-token chunk.

FLOPs at 2048 tokens, per the round-2 review:

| Option | rhythm | keep | roles | global mixes | GFLOP | vs planned |
|---|---|---|---|---|---|---|
| Planned | 3 | [0,4,8,12,16,20] | G L L G L L | 2 | 61.6 | — |
| Hedge | 2 | [0,4,9,13,18,20] | G L G L G L | 3 | 67.7 | +10% |
| MiniLM shape | 1 | [0,3,6,12,18,21] | all G | 6 | 85.7 | +39% |

The hedge is legal because `prune_to_layers` writes `layer_types` and TEI derives
roles from `index % global_attn_every_n_layers`, which is a value we also write.
Set `cfg.global_attn_every_n_layers = 2` before pruning and pass the keep list
manually; the existing TEI check then validates the new rhythm, and refuses the
selection if the assignment is forgotten. Every kept layer still holds its
pretrained role: 0, 9 and 18 are all global in the source.

Both configs get trained and benchmarked. At ~53 minutes each the A/B is nearly
free next to the decision it informs.

## 16. In-trainer eval works. The log just buffers it.

**Superseded claim, kept because it misled a later review.** During the v2 run
this section said eval had stopped firing. It had not.

Mid-run greps for `eval_loss` came back empty well past step 2500, while
training-loss lines were present. The cause is buffering, not behaviour:
transformers writes training loss through tqdm to unbuffered stderr, and the
eval dict through a plain print to stdout, which is block-buffered when
redirected to a file. The eval line existed and had not flushed yet.

v2 logged all six stage-1 eval points (2500 through 14016) and both stage-2
points. A controlled test on the same VM, same versions, same MarginMSE and
pruned-model setup, logged 6 eval points at `eval_steps=5`.

Consequence for planning: a 2-epoch stage 1 gets eval every 2500 steps, so 11
points, and a matched comparison against v2 at step 14016 comes from the eval
curve rather than from a saved checkpoint. `eval_checkpoints.py` remains useful
as an independent cross-check, and its holdout reproduction (same seed, same
group-split, row and query counts printed for comparison against the training
log) is the part worth keeping.

**To read the log, split on carriage returns first**, or tqdm's progress bar
hides everything on one line:
`tr '\r' '\n' < train.log | grep -a eval_loss`

## 17. BEAM comparison — what the reviews found

Two rounds on `run_beam_reranker.py`. The first invalidated the plan rather than
confirming it.

Round 1, all confirmed:

1. `start_token_proxy(port, upstream)` had its arguments swapped. Immediate
   TypeError, first minute.
2. **Bank reuse broken two independent ways.** `DaemonEmbedManager` overwrites
   `HINDSIGHT_API_DATABASE_URL` unless `HINDSIGHT_EMBED_API_DATABASE_URL` is set,
   so every arm got its own empty database. And `QualityBenchmark.run` stamped
   banks with its own wall clock at call time, so no later arm could name arm 1's
   banks. `run()` now takes an explicit `bank_ts`.
3. **The failure is silent.** A missing bank does not raise: recall 404s,
   quality.py fabricates `results: []`, and the arm completes all 80 questions on
   "No memories found" context, reporting a plausible low accuracy that reads
   exactly like a reranker regression.
4. Both student arms would block 900s in `start_tei` and abort the run, because
   the HF repo is private and TEI gets no token. The guard against this was
   literally `pass` on both branches.
5. Proxy bound loopback while the daemon was pointed at the Docker bridge.
6. `proxy.terminate()` does not exist on `ThreadingHTTPServer`.

Round 2, after the rewrite:

7. **`ingested = True` fired only after the whole arm succeeded.** An arm that
   ingested and then failed left it False, so the next arm re-ingested into the
   same bank ids. `create_bank` is create-or-update and retain document ids embed
   a fresh wall clock, so that is a second copy of every session, not an upsert.
   The run then completes on doubled banks.
8. **`documents_scored` proves TEI answered once, not that it survived.** Recall
   failures count toward no abort threshold, so TEI dying at question 10 finishes
   the arm with a depressed score and no error. Now also checks the probe's
   `forward_errors`.
9. `probe.wait(timeout=30)` inside `finally` could raise and kill every remaining
   arm. Cleanup must not be able to end the run.

## 18. What BEAM can and cannot answer

Verified against product source, not assumed:

- **Chunks never enter the ranked pool.** `_facts_only_recall` passes
  `include_chunks=False`, and chunks are fetched at Step 5.5 from the chunk_ids
  of already-reranked facts. `include_chunks` changes response content only. The
  student's long-document advantage cannot appear here.
- **"300 candidates" is the cap, not the pool.** `budget="low"` maps to a fixed
  thinking budget of 100 per retrieval arm, and Step 5 truncates to 200
  afterwards. Say "production cap, low-budget retrieval".
- **n=80 cannot rank two cross-encoders.** Minimum detectable difference is
  roughly 7-12 points paired, worse with conversation clustering. The expected
  student-versus-production gap is under 2 points. Going from 4 conversations to
  20 costs 5x and still would not resolve it.
- What it CAN certify: the RRF-versus-cross-encoder gap if it is 10+ points, and
  the absence of a catastrophic regression at 15+ points.
- Retain never consults the reranker, which is what makes one shared bank
  legitimate. Entity resolution scores with Jaro-Winkler and pg_trgm, the
  consolidator's recall passes `reranking="interleave"` which skips the
  cross-encoder branch, and observations are disabled in this config.

## 19. Reranker facts confirmed from the product source

- The scored document is `"[Date: June 05, 2022 (2022-06-05)] {context}: {text}"`
  (`engine/search/reranking.py:272`). **Not `text_signals`**, which is BM25-only
  and exists to feed entity names into the tsvector index.
- `DEFAULT_RERANKER_MAX_CANDIDATES = 300` (`config.py:995`). The quality
  harness's 30 is a benchmark override that predates the BEAM rebuild.
- `DEFAULT_RETAIN_EXTRACTION_MODE = "concise"` (`config.py:1172`). `chunks` is
  one of five modes and is what dev runs for speed. **In a default deployment the
  reranked document is a short extracted fact**, so the +107% chunk result
  applies to chunks-mode deployments and to long `context` fields, not to the
  default. The default gets the +1.7% fact number.

TEI 1.8.3 ignores `config.layer_types` entirely and re-derives each layer's role
as `index % global_attn_every_n_layers` (`flash_modernbert.rs:226`). A layer
selection that disagrees with that arithmetic trains one model and serves a
different one, silently, with `tei_parity_check` as the only backstop.

## 20. Both trained checkpoints return NaN on long documents in float16

**Severity: this affects v1, which is already published on HuggingFace and
described as an 8192-context reranker.**

TEI serves float16 by default and offers only `float16` or `float32`. Both of
our trained checkpoints produce `{"error":"score is NaN"}` above a length
threshold in float16:

| model | last good | first NaN |
|---|---|---|
| v1 (22 layers) | 1024 tok | 2048 tok |
| v2 (6 layers) | 768 tok | 896 tok |

The 6-layer throughput run surfaced it: the `chunk_max` profile failed 96 of 96
requests. That profile sizes documents with `bge-reranker-base`'s tokenizer for
corpus comparability, and mmBERT's 256k vocabulary splits the same text into
more tokens, pushing every document over the cliff.

### What it is not

- **Not a training failure.** The same weights are correct at every length in
  bfloat16 and float32, matching to three decimals. We train in bf16, which
  carries float32's exponent range.
- **Not batch-related.** 128 documents at 750 tokens succeed; one document at
  896 fails. Purely sequence length.
- **Not activation overflow in the residual stream.** Per-layer max magnitude at
  1024 tokens peaks at 98.1 against fp16's 65504 ceiling, with no NaN or inf at
  any layer output.
- **Not attention-logit overflow.** Peak pre-softmax score at 2048 tokens is
  20 (v2) and 46 (v1).
- **Not the attention kernel.** Reproduces with `attn_implementation="eager"`,
  so it is not SDPA or flash-specific.
- **Not a general ModernBERT-on-TEI problem.**
  `Alibaba-NLP/gte-reranker-modernbert-base`, a published 8192-context
  ModernBERT reranker, is correct in float16 at 768 / 1024 / 2048 / 4096.

### What it is

**Our fine-tuning causes it.** The untrained `jhu-clsp/mmBERT-small` backbone
is correct in float16 at every length tested. Both of our trained checkpoints
are not, and the shallower one fails earlier. Every magnitude that is easy to
instrument stays small, which points at an underflow (a LayerNorm variance
collapsing to zero, giving `1/sqrt(0)`) rather than an overflow. Unproven.

One backbone difference worth chasing: mmBERT-small sets
`local_rope_theta = 160000`, equal to its global theta, where standard
ModernBERT uses 10000 for sliding-window layers.

### Working fix

`--dtype float32` with `--max-batch-tokens 16384`. Correct at 768 through 4096.
The default 32768 budget OOMs a 24GB L4 during warmup at double the bytes per
value, which is why the first attempt looked like a boot failure rather than a
memory limit.

### Consequences

- The v1 model card, the `vectorize-io/hindsight-reranker-small` README, and
  every "8192 context" claim are wrong as served. **Correct them before Nicolo
  swaps anything into dev.**
- Published quality numbers stand: fact documents are ~45 tokens and the chunk
  profile is 380, both far under the cliff.
- Hindsight treats a 424 as a failed rerank, so in chunks mode with 3000-char
  chunks this degrades recall quietly instead of failing loudly.
- bfloat16 would fix it and TEI cannot serve bf16. vLLM and Infinity can, which
  makes the engine comparison relevant to correctness and not only to speed.

## 21. 8-bit is a dead end. The engine is not.

Measured on a dedicated L4, `BAAI/bge-reranker-base` and our student, three
engines, fp16 against 8-bit. One request = a 300-candidate rerank.

### Every 8-bit configuration that engaged was slower

| config | versus its own fp16 |
|---|---|
| vLLM 0.28.0 fp8, bge | 23% slower on chunk |
| Infinity 0.0.77 int8, bge, full coverage | 28% slower on chunk |
| vLLM fp8, our ModernBERT student | flag accepted, quantizes nothing |
| Infinity int8, our student | fails to load |

Ada's 2x FP8 ratio applies to dense GEMM. At hidden size 384 or 768 these
matmuls are small enough that the quantize and scale steps cost more than the
multiply saves. Weight-only int8 is worse still: it dequantizes every forward
pass and gives up the fp16 kernel.

### Two of the four are SILENT no-ops

This is the part worth remembering. Both engines accept the flag and report
success while changing nothing:

- **vLLM `--quantization fp8` on ModernBERT**: byte-identical 268.54 MiB
  footprint, all 66 linear layers still `UnquantizedLinearMethod`, 134 fp16
  weight tensors. On bge the same flag genuinely engages (48 `float8_e4m3fn`
  tensors, 531.71 → 450.71 MiB), so the flag works, just not on this
  architecture.
- **Infinity int8 with BetterTransformer on (the default)**: replaces
  `WeightOnlyInt8Linear` in **2 of 74** linear layers, because BetterTransformer
  fuses the encoder before the int8 pass reaches it. 529.77 MiB against fp16's
  530.34. With BetterTransformer off it is 74 of 74 and 448.93 MiB, and that is
  the configuration that is 28% slower.

Anyone trusting the flag would have published two fabricated "8-bit results".
Verify quantization by introspecting weight dtypes and parameter bytes, never by
reading a log line.

### Where it engages, it also costs ranking

vLLM fp8 on bge changes the **top-ranked document on 4 of 12 queries**
(top-1 agreement 0.67, Spearman 0.9887) while being 23% slower. Infinity int8
drops score distinctness from 0.704 to 0.544, creating real ties.

### The real finding: 171k tok/s was TEI's ceiling, not the L4's

vLLM reaches ~244,000 tok/s on the same GPU and weights, roughly 17% MFU against
121 TFLOPS fp16. Both engines are far from the hardware limit, so headroom
exists and quantization is the wrong tool to reach it.

| profile | TEI + student | vLLM + student | |
|---|---|---|---|
| chunk (380 tok) | 1.40 req/s | 2.10 | **1.50x** |
| chunk_max (750 tok) | 0.75 req/s | 1.10 | **1.47x** |
| fact (45 tok) | 8.60 req/s | 5.10 | **0.59x** |

**vLLM wins on long documents and loses badly on short ones.** Its advantage
appears only above concurrency 1, where continuous batching fills a forward pass
from several in-flight requests and TEI's does not. So the engine choice depends
on the production token-length distribution, which nobody has measured. That is
what `rerank_length_probe.py` exists to capture.

Cross-engine fp16 parity is sound: TEI vs vLLM on the student is Spearman
0.99986 with top-1 agreement 1.0, despite vLLM logging `pooling_type=CLS` where
the config says `classifier_pooling: mean`.

### Other facts worth keeping

- TEI's `--max-batch-tokens` default (32768) is already its best: the sweep
  16384/32768/65536/131072 gives 1.40/1.35/1.35/1.35 req/s. The dial flagged as
  untested early on is a dead end.
- Infinity's own warning: `quantization to int8 mode currently yields incorrect
  results`.
- Infinity cannot load ModernBERT at int8 because its handler assumes every
  `nn.Linear` has a bias, and ModernBERT is bias-free.
