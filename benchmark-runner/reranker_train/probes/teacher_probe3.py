"""Does the probe-1 finding hold in the coding-agent genre, and across document genres?

Probe 1 and 2 used the concise pipe-fact genre with personal-life content and found the
teacher carries essentially no supersession signal (-0.4 to +0.07 logits) against a
relevance scale of +13.6 logits.

This probe repeats the knowledge-update construction in:
  - `obs`    the observation genre the Claude Code plugin reranks exclusively
             (clean prose, [Date: ...] prefix, no pipe template, no context prefix)
  - `fact`   the concise pipe template with `claude-code: ` context prefix
  - `bare`   the same sentence with no date prefix at all, as a control on whether the
             prefix contributes anything

with coding-domain content: architecture decisions, dependency versions, CI, API contracts.
"""

import json
import statistics
from datetime import date

import torch
from sentence_transformers import CrossEncoder

MODEL = "BAAI/bge-reranker-v2-m3"

# (subject, old state, new state, why, "now" query, short query)
CODE = [
    ("the ingest worker", "retries failed messages three times with a fixed 5 second backoff",
     "retries failed messages six times with exponential backoff capped at 90 seconds",
     "after the duplicate-delivery incident",
     "what is the retry policy in the ingest worker now", "ingest worker retry"),
    ("the search service", "stores embeddings in Postgres with pgvector",
     "stores embeddings in Qdrant behind a thin adapter", "after the pgvector index blew past 40 GB",
     "where does the search service store embeddings now", "search embeddings store"),
    ("the deploy pipeline", "runs on Jenkins with a self-hosted agent pool",
     "runs on GitHub Actions with hosted runners", "after the Jenkins host was decommissioned",
     "what CI does the deploy pipeline use now", "deploy pipeline CI"),
    ("the public API", "exposes /v1/recall with a flat results array",
     "exposes /v2/recall with a paginated envelope and a cursor",
     "after the pagination RFC was accepted",
     "what does the public recall endpoint return now", "recall endpoint shape"),
    ("the auth middleware", "validates JWTs with a shared HS256 secret",
     "validates JWTs against JWKS with RS256 and key rotation", "after the secret leaked in a log",
     "how does the auth middleware validate tokens now", "auth token validation"),
    ("the reranker", "runs bge-reranker-base on TEI in float16",
     "runs the 6-layer distilled student on vLLM in bfloat16",
     "after the float16 pooling overflow was traced",
     "what reranker model is deployed now", "reranker deployment"),
    ("the migration runner", "applies Alembic revisions at process start",
     "applies Alembic revisions in a separate init container",
     "after two pods raced the same migration",
     "when do database migrations run now", "migration runner"),
    ("the chunker", "splits documents at 512 tokens with no overlap",
     "splits documents at 1024 tokens with 128 tokens of overlap",
     "after recall dropped on cross-boundary facts",
     "what chunk size does the chunker use now", "chunk size"),
    ("the rate limiter", "allows 100 requests per minute per API key",
     "allows 600 requests per minute per API key with a burst bucket of 50",
     "after the batch ingest customers complained",
     "what is the rate limit now", "rate limit"),
    ("the test suite", "mocks the LLM provider with recorded fixtures",
     "runs against a local ollama model in CI", "after the fixtures drifted from the real schema",
     "how does the test suite handle the LLM now", "test suite LLM"),
    ("the consolidation job", "runs hourly over the whole bank",
     "runs every fifteen minutes over a dirty-set queue", "after the hourly run started timing out",
     "how often does consolidation run now", "consolidation schedule"),
    ("the client SDK", "pins httpx at 0.24 and retries on 5xx only",
     "pins httpx at 0.28 and retries on 5xx plus connection errors",
     "after intermittent connection resets in production",
     "what does the client SDK retry on now", "client SDK retries"),
    ("the embedding model", "uses bge-small-en-v1.5 at 384 dimensions",
     "uses mmBERT-small at 768 dimensions with Matryoshka truncation",
     "after the multilingual requirement landed",
     "what embedding model is in use now", "embedding model"),
    ("the graph expansion arm", "follows one hop from seed facts",
     "follows two hops with a decay factor of 0.6", "after multi-hop recall measured poorly",
     "how many hops does graph expansion follow now", "graph expansion hops"),
    ("the config resolver", "reads settings from environment variables only",
     "reads settings from a layered chain of defaults, file, and environment",
     "after per-bank overrides were needed",
     "where does the config resolver read settings from now", "config resolution"),
    ("the error handling", "returns a bare 500 with the exception string",
     "returns a typed error code with a request id and no internal detail",
     "after an internal path leaked in a customer trace",
     "what does the API return on an internal error now", "API error format"),
    ("the cache layer", "keys on the raw query string",
     "keys on a normalised query hash plus the bank id and budget",
     "after cross-bank cache bleed was reported",
     "what does the cache key on now", "cache key"),
    ("the worker pool", "runs four processes with a thread per request",
     "runs two processes with an async event loop", "after memory per pod doubled",
     "what concurrency model does the worker pool use now", "worker concurrency"),
    ("the schema", "stores tags as a comma-separated text column",
     "stores tags as a GIN-indexed text array", "after tag filters went sequential-scan",
     "how are tags stored now", "tag storage"),
    ("the release process", "tags and publishes from a maintainer laptop",
     "tags and publishes from a signed CI job with provenance attestation",
     "after supply-chain review",
     "how are releases published now", "release process"),
]

OLD_D = [date(2024, 2, 8), date(2024, 6, 19), date(2025, 1, 23), date(2025, 3, 6)]
NEW_D = [date(2026, 1, 15), date(2026, 3, 30), date(2026, 6, 11), date(2026, 7, 24)]


def r_obs(d, subj, state, why):
    """Observation genre: clean prose, date prefix, no context prefix."""
    return (f"[Date: {d.strftime('%B %d, %Y')} ({d.isoformat()})] "
            f"{subj.capitalize()} {state}, decided {why}.")


def r_fact(d, subj, state, why):
    """Concise pipe fact with the claude-code context prefix."""
    t = (f"{subj.capitalize()} {state} | When: {d.strftime('%B %Y')} "
         f"| Involving: the platform team | {why}")
    return f"[Date: {d.strftime('%B %d, %Y')} ({d.isoformat()})] claude-code: {t}"


def r_bare(d, subj, state, why):
    """No date prefix at all."""
    return f"{subj.capitalize()} {state}, decided {why}."


RENDERERS = {"obs": r_obs, "fact": r_fact, "bare": r_bare}


def main():
    print(f"loading {MODEL} ...", flush=True)
    ce = CrossEncoder(MODEL, max_length=512, device="cpu", activation_fn=torch.nn.Identity())

    pairs = []
    for i, (subj, s_old, s_new, why, q_now, q_short) in enumerate(CODE):
        d_o, d_n = OLD_D[i % 4], NEW_D[i % 4]
        for gname, rend in RENDERERS.items():
            for q in (q_now, q_short):
                pairs += [(q, rend(d_o, subj, s_old, why)), (q, rend(d_n, subj, s_new, why))]
        # relevance control on the observation genre
        j = (i + 9) % len(CODE)
        o_subj, _, o_new, o_why, _, _ = CODE[j]
        pairs.append((q_now, r_obs(NEW_D[j % 4], o_subj, o_new, o_why)))
        # explicit-date query on the observation genre (does it read the prefix?)
        q_dated = f"what changed about {subj} in {d_n.strftime('%B %Y')}"
        pairs += [(q_dated, r_obs(d_o, subj, s_new, why)), (q_dated, r_obs(d_n, subj, s_new, why))]

    print(f"scoring {len(pairs)} pairs ...", flush=True)
    sc = list(map(float, ce.predict(pairs, batch_size=16, show_progress_bar=False)))

    per = 3 * 2 * 2 + 1 + 2
    res = {g: {"now": [], "short": []} for g in RENDERERS}
    ctrl, dated = [], []
    for i in range(len(CODE)):
        k = i * per
        for g in RENDERERS:
            for qs in ("now", "short"):
                res[g][qs].append(sc[k + 1] - sc[k])   # new - old, positive is correct
                k += 2
        ctrl.append(sc[k]); k += 1
        dated.append(sc[k + 1] - sc[k]); k += 2

    out = {}
    print("\n=== KNOWLEDGE UPDATE in the coding domain (correct = NEWER) ===")
    print(f"{'genre':6s} {'query':6s}  newer wins    mean       median")
    for g in RENDERERS:
        for qs in ("now", "short"):
            d = res[g][qs]
            w = sum(1 for x in d if x > 0)
            print(f"{g:6s} {qs:6s}  {w:2d}/{len(d)} ({100*w/len(d):3.0f}%)  "
                  f"{statistics.mean(d):+.3f}   {statistics.median(d):+.3f}")
            out[f"{g}_{qs}"] = {"wins": w, "n": len(d), "mean": statistics.mean(d)}

    print("\n=== does the date prefix contribute at all? obs vs bare, same content ===")
    for qs in ("now", "short"):
        a, b = res["obs"][qs], res["bare"][qs]
        diff = [x - y for x, y in zip(a, b)]
        print(f"  {qs:6s}: mean margin with prefix {statistics.mean(a):+.3f}, "
              f"without {statistics.mean(b):+.3f}, difference {statistics.mean(diff):+.3f}")
        out[f"prefix_effect_{qs}"] = statistics.mean(diff)

    print("\n=== EXPLICIT-DATE QUERY on the observation genre (correct = matching date) ===")
    w = sum(1 for x in dated if x > 0)
    print(f"  matching date wins {w}/{len(dated)} ({100*w/len(dated):.0f}%), "
          f"mean {statistics.mean(dated):+.3f}")
    out["dated"] = {"wins": w, "n": len(dated), "mean": statistics.mean(dated)}

    print("\n=== RELEVANCE CONTROL (observation genre) ===")
    rel = []
    for i in range(len(CODE)):
        k = i * per
        rel.append(statistics.mean([sc[k], sc[k + 1]]) - ctrl[i])
    print(f"  mean(relevant) - unrelated subsystem: {statistics.mean(rel):+.3f} "
          f"(median {statistics.median(rel):+.3f}, min {min(rel):+.3f})")
    out["relevance_margin"] = statistics.mean(rel)

    with open("teacher_probe3.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote teacher_probe3.json")


if __name__ == "__main__":
    main()
