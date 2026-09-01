#!/usr/bin/env python3
"""
Reranker performance benchmark — throughput and latency under concurrency.

This is the performance counterpart to run_all_reranker.py, which measures
reranking *quality* (MRR, Recall@K). This script measures how fast a reranker
endpoint answers, and where it stops keeping up.

It replays the shape of Hindsight's real recall path against a Text Embeddings
Inference (TEI) /rerank endpoint:

  * one "recall" = one query plus N candidate documents
  * the client splits candidates into batches of `--batch-size` (Hindsight's
    RemoteTEICrossEncoder default is 128) and issues those requests together
  * each request is POST /rerank {"query", "texts", "return_text": false},
    the same body cross_encoder.py sends
  * documents carry the same "[Date: ...] context: text" prefix the engine
    builds in search/reranking.py
  * `--semaphore` reproduces the per-process cap on in-flight requests
    (Hindsight uses 8). Set 0 to remove it and measure the server's own ceiling.

Load is closed-loop: `--concurrency` workers each run recalls back to back.

Measurement discipline:
  * a warmup phase runs real load and is discarded entirely; it ends only once
    the clock has elapsed AND every worker has completed at least one recall,
    so no cold recall can leak into the measured set
  * the measured window is a fixed `--duration` seconds. Throughput uses that
    fixed denominator, so a slow drain at the end cannot understate it
  * only recalls that COMPLETED inside the window are scored, and only
    SUCCESSFUL ones count toward throughput. Failures are reported separately,
    which matters at overload, where counting failures as throughput would
    otherwise show a server "keeping up" while it sheds load
  * requests in flight when the window closes are discarded, not waited for

Retries: Hindsight retries 429 and 5xx up to 3 times with backoff. This tool
does NOT retry, so offered load is retry-free and the reported saturation point
is the server's first-failure point rather than production's effective one.

Documents are built from real BEAM conversation prose, sliced to an exact token
count with the reranker's own tokenizer, so a "380 token" document really is
380 tokens to the model.

Usage:
  python reranker_perf.py --url http://localhost:8081 --out results.json

Dependencies: httpx, tokenizers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import httpx
from tokenizers import Tokenizer

BENCHMARK_RUNNER_DIR = Path(__file__).parent
DEFAULT_BEAM = BENCHMARK_RUNNER_DIR / "datasets" / "beam_128k_subset.json"

# Hindsight's RemoteTEICrossEncoder defaults (hindsight_api/config.py).
HINDSIGHT_BATCH_SIZE = 128
HINDSIGHT_MAX_CANDIDATES = 300
HINDSIGHT_CLIENT_SEMAPHORE = 8
HINDSIGHT_REQUEST_TIMEOUT = 30.0

# Document length profiles, in tokens of the reranker's own tokenizer.
#   fact      — an extracted Hindsight fact. Measured from the annotated LoCoMo
#               ground truth: p50 166 chars, p90 266.
#   chunk     — the ~1500-character case Nicolo measured, about 380 tokens.
#   chunk_max — a full-size chunk at Hindsight's DEFAULT_RETAIN_CHUNK_SIZE of
#               3000 chars, about 750 tokens. A 512-token model truncates this,
#               so it is the profile where context length actually matters.
DOC_PROFILES: dict[str, int] = {"fact": 45, "chunk": 380, "chunk_max": 750}


@dataclass(frozen=True)
class Scenario:
    profile: str
    candidates: int
    concurrency: int

    @property
    def name(self) -> str:
        return f"{self.profile}/{self.candidates}cand/{self.concurrency}conc"


@dataclass
class Sample:
    """One completed unit of work, with the clock readings needed to place it."""

    start: float
    end: float
    ok: bool

    @property
    def latency_ms(self) -> float:
        return (self.end - self.start) * 1000.0


@dataclass
class ScenarioResult:
    scenario: str
    profile: str
    doc_tokens: int
    candidates: int
    concurrency: int

    # What actually bound in-flight requests: the worker count or the semaphore.
    requests_per_recall: int
    max_inflight_requests: int
    binding_constraint: str

    duration_s: float
    recalls_ok: int
    recalls_failed: int
    recall_success_rate: float
    requests_ok: int
    requests_failed: int
    pairs: int

    recalls_per_s: float
    requests_per_s: float
    pairs_per_s: float

    recall_p50_ms: float
    recall_p90_ms: float
    recall_p99_ms: float
    recall_max_ms: float

    request_p50_ms: float
    request_p90_ms: float
    request_p99_ms: float
    request_max_ms: float

    errors_by_kind: dict[str, int] = field(default_factory=dict)


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile. Never invents a value the run did not see."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def load_tokenizer(name: str) -> Tokenizer:
    """Load a tokenizer with padding and truncation switched off.

    Some tokenizer.json files ship padding enabled to a fixed length —
    Alibaba-NLP/gte-multilingual-reranker-base pads to 512. Left on, every
    encode() returns that length no matter the input, which silently breaks
    document sizing: the prefix measures 512 tokens, the body budget collapses
    to 1, and the benchmark ends up reranking near-empty documents while
    reporting the requested length. Truncation would cap long documents for the
    same reason. Both have to be off to measure real token counts.
    """
    tokenizer = Tokenizer.from_pretrained(name)
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def load_beam_text(beam_path: Path) -> tuple[list[str], list[str]]:
    """Return (queries, prose_passages) pulled from the BEAM subset."""
    with beam_path.open() as fh:
        samples = json.load(fh)

    queries: list[str] = []
    prose: list[str] = []
    for sample in samples:
        for qa in sample.get("qa", []):
            question = qa.get("question", "").strip()
            if question:
                queries.append(question)
        for session in sample.get("sessions", []):
            for message in session.get("messages", []):
                content = message.get("content", "").strip()
                if content:
                    prose.append(content)

    if not queries:
        raise RuntimeError(f"No questions found in {beam_path}")
    if not prose:
        raise RuntimeError(f"No message prose found in {beam_path}")
    return queries, prose


def build_documents(
    tokenizer: Tokenizer,
    prose: list[str],
    target_tokens: int,
    count: int,
    rng: random.Random,
) -> tuple[list[str], bool]:
    """Build `count` documents of `target_tokens` tokens each.

    Returns (documents, prose_was_recycled). The date/context prefix Hindsight
    prepends is inside the budget, so the document that reaches the model is the
    length this asks for. Documents are non-overlapping windows of a shuffled
    stream of real prose, so no two are identical. If the source prose is too
    small the stream wraps and repeats verbatim, which the caller is told about
    rather than left to discover in the numbers.
    """
    prefix = "[Date: June 05, 2022 (2022-06-05)] conversation: "
    prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False).ids)
    body_tokens = max(1, target_tokens - prefix_len)

    shuffled = prose[:]
    rng.shuffle(shuffled)

    needed = body_tokens * count
    stream: list[int] = []
    passages_used = 0
    while len(stream) < needed:
        passage = shuffled[passages_used % len(shuffled)]
        stream.extend(tokenizer.encode(passage, add_special_tokens=False).ids)
        passages_used += 1
    recycled = passages_used > len(shuffled)

    documents = [
        prefix + tokenizer.decode(stream[i * body_tokens : (i + 1) * body_tokens]) for i in range(count)
    ]
    return documents, recycled


def measure_document_lengths(tokenizer: Tokenizer, documents: list[str]) -> tuple[int, int, float]:
    """Real token lengths, since decode/encode is not always round-trip exact."""
    lengths = [len(tokenizer.encode(d, add_special_tokens=False).ids) for d in documents]
    return min(lengths), max(lengths), statistics.mean(lengths)


# ---------------------------------------------------------------------------
# Load generation
# ---------------------------------------------------------------------------


class RerankClient:
    """Issues /rerank calls the way Hindsight's RemoteTEICrossEncoder does."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        batch_size: int,
        semaphore: asyncio.Semaphore | None,
    ) -> None:
        self._client = client
        self._url = base_url.rstrip("/") + "/rerank"
        self._batch_size = batch_size
        self._semaphore = semaphore
        self.request_samples: list[Sample] = []
        self.errors_by_kind: dict[str, int] = {}

    def _record_error(self, kind: str) -> None:
        self.errors_by_kind[kind] = self.errors_by_kind.get(kind, 0) + 1

    async def _post(self, query: str, texts: list[str]) -> bool:
        payload = {"query": query, "texts": texts, "return_text": False}
        started = time.perf_counter()
        try:
            if self._semaphore is not None:
                async with self._semaphore:
                    response = await self._client.post(self._url, json=payload)
            else:
                response = await self._client.post(self._url, json=payload)
        except httpx.HTTPError as exc:
            self.request_samples.append(Sample(started, time.perf_counter(), False))
            self._record_error(type(exc).__name__)
            return False

        finished = time.perf_counter()
        ok = response.status_code == 200
        if not ok:
            self._record_error(f"HTTP {response.status_code}")
        self.request_samples.append(Sample(started, finished, ok))
        return ok

    async def recall(self, query: str, documents: list[str]) -> Sample:
        batches = [documents[i : i + self._batch_size] for i in range(0, len(documents), self._batch_size)]
        started = time.perf_counter()
        outcomes = await asyncio.gather(*(self._post(query, batch) for batch in batches))
        return Sample(started, time.perf_counter(), all(outcomes))


async def run_scenario(
    scenario: Scenario,
    base_url: str,
    documents: list[str],
    queries: list[str],
    doc_tokens: int,
    batch_size: int,
    semaphore_size: int,
    duration_s: float,
    warmup_s: float,
    request_timeout: float,
    seed: int,
) -> ScenarioResult:
    """Closed-loop load with a discarded warmup and a fixed measured window."""
    requests_per_recall = math.ceil(scenario.candidates / batch_size)
    demanded_inflight = scenario.concurrency * requests_per_recall
    if semaphore_size > 0 and semaphore_size < demanded_inflight:
        max_inflight = semaphore_size
        binding = "client_semaphore"
    else:
        max_inflight = demanded_inflight
        binding = "concurrency"

    # Enough connections for every request that can be in flight, plus headroom.
    # Without this, excess requests queue inside httpx and that wait would be
    # misreported as server latency.
    conn_cap = max(16, max_inflight * 2)
    limits = httpx.Limits(max_connections=conn_cap, max_keepalive_connections=conn_cap)
    # An explicit pool timeout makes connection starvation surface as an error
    # instead of silently inflating the latency numbers.
    timeout = httpx.Timeout(request_timeout, pool=5.0)

    recall_samples: list[Sample] = []
    warmup_done: list[bool] = [False] * scenario.concurrency

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as http_client:
        semaphore = asyncio.Semaphore(semaphore_size) if semaphore_size > 0 else None
        client = RerankClient(http_client, base_url, batch_size, semaphore)

        run_started = time.perf_counter()
        # Filled in once warmup ends; workers stop issuing at `stop_at`.
        window: dict[str, float] = {}
        stop_event = asyncio.Event()

        def warmup_over() -> bool:
            return (time.perf_counter() - run_started) >= warmup_s and all(warmup_done)

        async def worker(index: int, worker_seed: int) -> None:
            local_rng = random.Random(worker_seed)
            while not stop_event.is_set():
                query = local_rng.choice(queries)
                # Rotate the window into the pool so successive recalls do not
                # send byte-identical payloads.
                offset = local_rng.randrange(len(documents))
                picked = [documents[(offset + i) % len(documents)] for i in range(scenario.candidates)]
                sample = await client.recall(query, picked)
                warmup_done[index] = True
                if window:
                    recall_samples.append(sample)

        async def controller() -> None:
            # Phase A: warmup. Real load, entirely discarded.
            while not warmup_over():
                await asyncio.sleep(0.05)
            # Phase B: measured window with a fixed denominator.
            start = time.perf_counter()
            window["start"] = start
            window["end"] = start + duration_s
            client.request_samples.clear()
            client.errors_by_kind.clear()
            await asyncio.sleep(duration_s)
            stop_event.set()

        await asyncio.gather(
            controller(),
            *(worker(i, seed + i) for i in range(scenario.concurrency)),
        )

        window_start, window_end = window["start"], window["end"]
        request_samples = [s for s in client.request_samples if window_start <= s.end <= window_end]
        errors_by_kind = dict(client.errors_by_kind)

    # Only work that COMPLETED inside the window is scored. In-flight work at
    # the close is discarded rather than waited for.
    scored = [s for s in recall_samples if window_start <= s.end <= window_end]
    ok_recalls = [s for s in scored if s.ok]
    failed_recalls = len(scored) - len(ok_recalls)

    ok_requests = [s for s in request_samples if s.ok]
    failed_requests = len(request_samples) - len(ok_requests)

    recall_latencies = [s.latency_ms for s in ok_recalls]
    request_latencies = [s.latency_ms for s in ok_requests]
    pairs = len(ok_recalls) * scenario.candidates
    total_scored = len(scored)

    return ScenarioResult(
        scenario=scenario.name,
        profile=scenario.profile,
        doc_tokens=doc_tokens,
        candidates=scenario.candidates,
        concurrency=scenario.concurrency,
        requests_per_recall=requests_per_recall,
        max_inflight_requests=max_inflight,
        binding_constraint=binding,
        duration_s=duration_s,
        recalls_ok=len(ok_recalls),
        recalls_failed=failed_recalls,
        recall_success_rate=round(len(ok_recalls) / total_scored, 4) if total_scored else 0.0,
        requests_ok=len(ok_requests),
        requests_failed=failed_requests,
        pairs=pairs,
        recalls_per_s=round(len(ok_recalls) / duration_s, 2),
        requests_per_s=round(len(ok_requests) / duration_s, 2),
        pairs_per_s=round(pairs / duration_s, 1),
        recall_p50_ms=round(_pct(recall_latencies, 0.50), 1),
        recall_p90_ms=round(_pct(recall_latencies, 0.90), 1),
        recall_p99_ms=round(_pct(recall_latencies, 0.99), 1),
        recall_max_ms=round(max(recall_latencies), 1) if recall_latencies else 0.0,
        request_p50_ms=round(_pct(request_latencies, 0.50), 1),
        request_p90_ms=round(_pct(request_latencies, 0.90), 1),
        request_p99_ms=round(_pct(request_latencies, 0.99), 1),
        request_max_ms=round(max(request_latencies), 1) if request_latencies else 0.0,
        errors_by_kind=errors_by_kind,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def probe(base_url: str, timeout: float) -> dict:
    """Confirm the endpoint is a reranker and report what it is serving."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        info = await client.get(base_url.rstrip("/") + "/info")
        info.raise_for_status()
        details = info.json()

        check = await client.post(
            base_url.rstrip("/") + "/rerank",
            json={
                "query": "what is the deployment process",
                "texts": ["the deploy runs from CI", "unrelated text about cats"],
                "return_text": False,
            },
        )
        check.raise_for_status()
        scored = check.json()
        if not isinstance(scored, list) or not scored or "score" not in scored[0]:
            raise RuntimeError(f"/rerank did not return scores: {scored!r}")
    return details


def parse_int_list(raw: str, what: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError(f"--{what} is empty")
    if any(v < 1 for v in values):
        raise ValueError(f"--{what} values must be >= 1, got {values}")
    return values


def write_results(out: Path, payload: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(payload, fh, indent=2)


async def main_async(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    out = Path(args.out)

    try:
        candidate_counts = parse_int_list(args.candidates, "candidates")
        concurrencies = parse_int_list(args.concurrency, "concurrency")
        unknown = [p for p in args.profiles.split(",") if p.strip() not in DOC_PROFILES]
        if unknown:
            raise ValueError(f"unknown profile(s) {unknown}; choose from {sorted(DOC_PROFILES)}")
        profiles = {p.strip(): DOC_PROFILES[p.strip()] for p in args.profiles.split(",") if p.strip()}
    except ValueError as exc:
        print(f"BAD ARGUMENTS: {exc}", file=sys.stderr)
        return 1

    # Load the tokenizer BEFORE probing, so a cold HF cache or a network hiccup
    # fails now rather than after the GPU has been billing through a warmup.
    print(f"Loading tokenizer {args.tokenizer} ...", flush=True)
    tokenizer = load_tokenizer(args.tokenizer)

    print(f"Probing {args.url} ...", flush=True)
    try:
        info = await probe(args.url, args.timeout)
    except Exception as exc:
        print(f"PROBE FAILED: {exc}", file=sys.stderr)
        return 1

    max_input = info.get("max_input_length")
    max_client_batch = info.get("max_client_batch_size")
    print(f"  model            : {info.get('model_id', 'unknown')}")
    print(f"  max_input_length : {max_input}")
    print(f"  max_client_batch : {max_client_batch}")
    print(f"  max_batch_tokens : {info.get('max_batch_tokens')}")
    print(f"  dtype            : {info.get('model_dtype')}")

    if max_client_batch is not None and args.batch_size > max_client_batch:
        print(
            f"REFUSING TO RUN: --batch-size {args.batch_size} exceeds the server's "
            f"max_client_batch_size {max_client_batch}; every request would fail.",
            file=sys.stderr,
        )
        return 1

    queries, prose = load_beam_text(Path(args.beam))
    longest_query = max(len(tokenizer.encode(q, add_special_tokens=False).ids) for q in queries)
    print(f"Corpus: {len(queries)} real queries (longest {longest_query} tok), {len(prose)} prose passages")

    pool_size = max(candidate_counts) * 4
    corpus: dict[str, list[str]] = {}
    corpus_stats: dict[str, dict] = {}
    for name, target in profiles.items():
        docs, recycled = build_documents(tokenizer, prose, target, pool_size, rng)
        lo, hi, mean = measure_document_lengths(tokenizer, docs)
        corpus[name] = docs
        corpus_stats[name] = {
            "target_tokens": target,
            "actual_min": lo,
            "actual_max": hi,
            "actual_mean": round(mean, 1),
            "pool_size": len(docs),
            "prose_recycled": recycled,
        }
        print(f"  profile {name:6s}: target {target} tok, actual {lo}-{hi} tok (mean {mean:.1f}), {len(docs)} docs")
        # A corpus that is not the requested length makes every throughput
        # number meaningless, and it fails silently — the run still completes,
        # just against the wrong documents. Refuse rather than report it.
        if not (0.9 * target <= mean <= 1.1 * target):
            print(
                f"REFUSING TO RUN: profile {name} targeted {target} tokens but built {mean:.1f}. "
                f"Check the tokenizer for baked-in padding or truncation.",
                file=sys.stderr,
            )
            return 1
        if recycled:
            print("    note: source prose was too small for a unique pool and wrapped; documents repeat.")
        # The model's limit covers the whole "[CLS] query [SEP] doc [SEP]" pair,
        # so the real document budget is the limit minus the query and specials.
        if max_input is not None and hi + longest_query + 3 > max_input:
            corpus_stats[name]["server_truncates"] = True
            print(
                f"    note: longest pair ({hi} doc + {longest_query} query + 3) exceeds the model's "
                f"{max_input}-token limit, so the server truncates (auto_truncate).",
                flush=True,
            )

    scenarios = [
        Scenario(profile=name, candidates=cands, concurrency=conc)
        for name in profiles
        for cands in candidate_counts
        for conc in concurrencies
    ]
    per_scenario_s = args.warmup + args.duration
    print(
        f"\nRunning {len(scenarios)} scenarios "
        f"({args.warmup:g}s warmup + {args.duration:g}s measured each, "
        f"~{len(scenarios) * per_scenario_s / 60:.0f} min minimum)\n",
        flush=True,
    )

    payload = {
        "endpoint_info": info,
        "config": {
            "url": args.url,
            "tokenizer": args.tokenizer,
            "batch_size": args.batch_size,
            "client_semaphore": args.semaphore,
            "duration_s": args.duration,
            "warmup_s": args.warmup,
            "request_timeout_s": args.timeout,
            "retries": "none (production retries 429/5xx up to 3x)",
            "seed": args.seed,
            "hindsight_defaults": {
                "batch_size": HINDSIGHT_BATCH_SIZE,
                "max_candidates": HINDSIGHT_MAX_CANDIDATES,
                "client_semaphore": HINDSIGHT_CLIENT_SEMAPHORE,
                "request_timeout_s": HINDSIGHT_REQUEST_TIMEOUT,
            },
        },
        "corpus": corpus_stats,
        "results": [],
        "failed_scenarios": [],
    }

    for i, scenario in enumerate(scenarios, start=1):
        print(f"[{i}/{len(scenarios)}] {scenario.name} ... ", end="", flush=True)
        try:
            result = await run_scenario(
                scenario,
                args.url,
                corpus[scenario.profile],
                queries,
                profiles[scenario.profile],
                args.batch_size,
                args.semaphore,
                args.duration,
                args.warmup,
                args.timeout,
                args.seed + i * 1000,
            )
        except Exception as exc:  # one bad scenario must not discard the sweep
            print(f"FAILED: {type(exc).__name__}: {exc}", flush=True)
            payload["failed_scenarios"].append({"scenario": scenario.name, "error": f"{type(exc).__name__}: {exc}"})
            write_results(out, payload)
            continue

        payload["results"].append(asdict(result))
        # Written after every scenario so a crash never discards completed work.
        write_results(out, payload)

        flags = []
        if result.recall_success_rate < 0.99:
            flags.append(f"SUCCESS {result.recall_success_rate:.0%}")
        if result.errors_by_kind:
            flags.append(str(result.errors_by_kind))
        suffix = ("  " + "  ".join(flags)) if flags else ""
        print(
            f"{result.recalls_per_s:6.2f} recall/s  "
            f"p50 {result.recall_p50_ms:7.1f}ms  p90 {result.recall_p90_ms:7.1f}ms  "
            f"{result.pairs_per_s:8.1f} pairs/s  [{result.binding_constraint}]{suffix}",
            flush=True,
        )

    print(f"\nWrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="TEI reranker base URL, e.g. http://localhost:8081")
    parser.add_argument("--out", default="reranker_perf_results.json", help="Where to write the result JSON")
    parser.add_argument("--tokenizer", default="BAAI/bge-reranker-base", help="HF tokenizer id for sizing documents")
    parser.add_argument("--beam", default=str(DEFAULT_BEAM), help="BEAM subset JSON supplying real text")
    parser.add_argument("--profiles", default="fact,chunk,chunk_max",
                        help="Comma-separated document profiles. chunk_max (750 tok) is the "
                             "regime this project exists to fix, so it is on by default.")
    parser.add_argument("--candidates", default="50,100,300", help="Comma-separated candidate counts per recall")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32", help="Comma-separated concurrent-recall levels")
    parser.add_argument("--batch-size", type=int, default=HINDSIGHT_BATCH_SIZE, help="Candidates per HTTP request")
    parser.add_argument(
        "--semaphore",
        type=int,
        default=HINDSIGHT_CLIENT_SEMAPHORE,
        help="Per-process cap on in-flight requests (0 = unlimited, measures the server's own ceiling)",
    )
    parser.add_argument("--duration", type=float, default=20.0, help="Measured seconds per scenario")
    parser.add_argument("--warmup", type=float, default=8.0, help="Discarded warmup seconds per scenario")
    parser.add_argument(
        "--timeout", type=float, default=HINDSIGHT_REQUEST_TIMEOUT, help="HTTP request timeout in seconds"
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed for corpus and query selection")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
