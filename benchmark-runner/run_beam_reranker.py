#!/usr/bin/env python3
"""
BEAM end-to-end quality, varying only the reranker.

run_all_quality.py answers "which LLM extracts the best memories" by holding the
reranker at RRF and sweeping LLMs. This asks the opposite question: hold the LLM
fixed and sweep the reranker.

WHAT THIS CAN AND CANNOT MEASURE
--------------------------------
Read this before quoting any number out of it.

**It cannot rank one cross-encoder against another.** The frozen subset is 4
conversations and 80 questions. For a paired comparison between two arms the
minimum detectable difference is roughly 7 to 12 points, and conversation
clustering makes even that optimistic. The expected gap between our student and
production is 1.7% fact MRR, diluted through an LLM answering and a judge
grading, which lands under 2 points. It is not detectable here and no feasible
sample size in this harness would make it so.

**It can measure two things.** The RRF-versus-cross-encoder gap, if it is 10+
points, which quantifies what the reranker is worth end to end. And the absence
of a catastrophic regression, 15+ points, which is the real question when
swapping a model into production.

**The reranker only ever scores extracted facts here.** `_facts_only_recall`
passes ``include_chunks=False``, and chunks are fetched after ranking from the
chunk_ids of already-reranked facts. Chunks never enter the ranked pool. Our
student's long-document advantage cannot appear in this benchmark.

**"300 candidates" is the cap, not the pool.** ``budget="low"`` sets a thinking
budget of 100 per retrieval arm, so the reranker sees roughly 100-300 unique
documents rather than a reliable 300, and Step 5 truncates to 200 afterwards.
Describe this as "production cap, low-budget retrieval", never as "production
settings".

The run is still worth its GPU hours: it is the configuration Nicolo runs next
week, it proves the student breaks nothing in the real pipeline, and the length
probe captures the one number nobody has, which is how long the documents
production actually reranks really are.

INGEST ONCE, EVALUATE MANY
--------------------------
Retain never consults the reranker (verified: entity resolution scores with
Jaro-Winkler and pg_trgm, the consolidator's recall uses interleave, and
observations are off), so every arm can share one ingested bank.

Two things make that work, and both were wrong in the first draft:

* The daemon manager ignores ``HINDSIGHT_API_DATABASE_URL`` and overwrites it
  with a per-profile name unless ``HINDSIGHT_EMBED_API_DATABASE_URL`` is set.
* ``QualityBenchmark.run`` stamps new banks with its own wall clock taken at
  call time, so a later arm could never name the banks the first arm wrote. It
  now takes an explicit ``bank_ts``.

Without both, arms 2+ recall against banks that do not exist, quality.py
fabricates empty results, and the run reports a confident low score that reads
exactly like a reranker regression.

Run on a GPU host with Docker:
  python run_beam_reranker.py
  python run_beam_reranker.py --arms rrf,production
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hindsight_embed import DaemonEmbedManager

from hindsight_benchmark.gcp import start_token_proxy, vertex_openai_upstream
from hindsight_benchmark.quality import QualityBenchmark
from run_reranker_eval import start_tei, stop_tei

BENCHMARK_RUNNER_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_RUNNER_DIR.parent / "results" / "leaderboard" / "beam-reranker"

VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "model-benchmark-506614")
LLM_MODEL = "google/gemini-3.7-flash"

# Everything in this script is a host process except TEI, which makes no
# outbound calls. So every URL is loopback and nothing needs the Docker bridge.
# Pointing the daemon at the bridge address while the proxy binds loopback was
# the first draft's silent connection failure.
HOST = "127.0.0.1"
PROXY_PORT = 8898
PROBE_PORT = 8097

MAX_CANDIDATES = 300

# label -> (reranker provider, model id or local checkpoint path)
ARMS: dict[str, tuple[str, str | None]] = {
    "rrf": ("rrf", None),
    "production": ("tei", "BAAI/bge-reranker-base"),
    "student-v1": ("tei", str(Path.home() / "models" / "hindsight-reranker-small")),
    "student-6layer": ("tei", str(Path.home() / "train" / "out6" / "final")),
}

# The student weights live in a PRIVATE HF repo. TEI gets no token, so it cannot
# pull them; the download happens on the host and TEI mounts the local path.
HF_STUDENT_REPO = "vectorize-io/hindsight-reranker-small"

BASE_CONFIG = {
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "local",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL": "BAAI/bge-small-en-v1.5",
    "HINDSIGHT_API_ENABLE_OBSERVATIONS": "false",
    "HINDSIGHT_API_EXTRACT_CAUSAL_LINKS": "false",
    "HINDSIGHT_API_LLM_TIMEOUT": "45",
    "HINDSIGHT_API_RETAIN_LLM_TIMEOUT": "600",
    "HINDSIGHT_API_RETAIN_WALL_TIMEOUT": "0",
    "HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT": "8",
    "HINDSIGHT_API_RETAIN_BATCH_TOKENS": "50000",
    "HINDSIGHT_API_DB_POOL_MIN_SIZE": "20",
    "HINDSIGHT_API_DB_COMMAND_TIMEOUT": "15",
    "HINDSIGHT_API_RERANKER_MAX_CANDIDATES": str(MAX_CANDIDATES),
}


def make_config(arm: str, llm_base_url: str, db_url: str) -> dict[str, str]:
    provider, _model = ARMS[arm]
    cfg = {
        **BASE_CONFIG,
        "HINDSIGHT_API_LLM_PROVIDER": "openai",
        "HINDSIGHT_API_LLM_MODEL": LLM_MODEL,
        "HINDSIGHT_API_LLM_BASE_URL": llm_base_url,
        "HINDSIGHT_API_LLM_API_KEY": "unused-adc",
        # DaemonEmbedManager reads the EMBED-prefixed key and overwrites the
        # plain one. Setting only HINDSIGHT_API_DATABASE_URL gives every arm its
        # own empty database and silently defeats the shared bank.
        "HINDSIGHT_EMBED_API_DATABASE_URL": db_url,
        "HINDSIGHT_API_DATABASE_URL": db_url,
        "HINDSIGHT_API_RERANKER_PROVIDER": provider,
    }
    if provider == "tei":
        cfg["HINDSIGHT_API_RERANKER_TEI_URL"] = f"http://{HOST}:{PROBE_PORT}"
    return cfg


def preflight(arms: list[str]) -> None:
    """Resolve every model before arm 1 runs.

    start_tei polls for 900 seconds before giving up, and it treats a
    nonexistent local path as a hub id. Discovering a missing model at arm 3
    costs fifteen minutes and, because start_tei raises, the rest of the run.
    """
    for arm in arms:
        provider, model = ARMS[arm]
        if provider != "tei":
            continue
        path = Path(model)
        if path.exists() and path.is_dir():
            weights = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
            if not weights or not (path / "config.json").exists():
                raise SystemExit(
                    f"{arm}: {path} exists but has no weights or no config.json "
                    "(an interrupted download looks exactly like this). Remove it and retry."
                )
            print(f"  {arm}: local checkpoint {path} ({weights[0].name})", flush=True)
            continue
        if arm == "student-v1":
            print(f"  {arm}: downloading {HF_STUDENT_REPO} -> {path}", flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["hf", "download", HF_STUDENT_REPO, "--local-dir", str(path),
                 "--exclude", "v0/*"],
                check=True,
            )
            continue
        if "/" in model and not path.is_absolute():
            r = requests.get(f"https://huggingface.co/api/models/{model}", timeout=30)
            if r.status_code != 200:
                raise SystemExit(f"{arm}: hub model {model!r} not reachable (HTTP {r.status_code})")
            print(f"  {arm}: hub model {model} reachable", flush=True)
            continue
        raise SystemExit(
            f"{arm}: {model!r} is neither an existing directory nor a hub id. "
            "start_tei would treat it as a hub id and block for 900s."
        )


def start_probe(tei_url: str, out_path: Path, tokenizer_id: str, log_path: Path):
    script = BENCHMARK_RUNNER_DIR / "reranker_train" / "rerank_length_probe.py"
    log = log_path.open("w")
    proc = subprocess.Popen(
        [sys.executable, str(script), "--upstream", tei_url,
         "--port", str(PROBE_PORT), "--out", str(out_path),
         "--tokenizer", tokenizer_id],
        stdout=log, stderr=subprocess.STDOUT,
    )
    deadline = time.time() + 90
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"length probe exited early; see {log_path}")
        try:
            if requests.get(f"http://{HOST}:{PROBE_PORT}/health", timeout=2).status_code == 200:
                print(f"  length probe up on :{PROBE_PORT} -> {tei_url}", flush=True)
                return proc
        except Exception:
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError(f"length probe did not come up; see {log_path}")


def probe_stats() -> dict:
    try:
        return requests.get(f"http://{HOST}:{PROBE_PORT}/stats", timeout=5).json()
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-conversations", type=int, default=None)
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--bank-ts", type=int, default=None,
                    help="Reuse banks from an earlier run with this stamp instead of ingesting")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {list(ARMS)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== preflight", flush=True)
    preflight(arms)

    run_ts = args.bank_ts or int(time.time())
    db_url = f"pg0://beam-rerank-{run_ts}"
    print(f"database: {db_url}   bank stamp: {run_ts}", flush=True)

    proxy = start_token_proxy(vertex_openai_upstream(VERTEX_PROJECT), PROXY_PORT)
    llm_base_url = f"http://{HOST}:{PROXY_PORT}/v1"

    benchmark = QualityBenchmark(vertex_project=VERTEX_PROJECT)
    mgr = DaemonEmbedManager()
    results: dict[str, dict] = {}
    ingested = args.bank_ts is not None

    try:
        for i, arm in enumerate(arms, 1):
            provider, model = ARMS[arm]
            print(f"\n{'=' * 60}\n[{i}/{len(arms)}] arm={arm} provider={provider} model={model}\n{'=' * 60}", flush=True)

            tei_started = probe = None
            profile = f"beam-rerank-{arm}-{run_ts}"
            reusing = ingested
            try:
                if provider == "tei":
                    tei_url = start_tei(model)
                    tei_started = True
                    probe = start_probe(
                        tei_url.replace("172.17.0.1", HOST),
                        RESULTS_DIR / f"lengths__{arm}.json",
                        model,
                        RESULTS_DIR / f"probe__{arm}.log",
                    )

                if not mgr.ensure_running(make_config(arm, llm_base_url, db_url), profile):
                    raise RuntimeError("daemon did not start")

                # Flip before evaluating, not after. Ingest is done once
                # run() returns, whatever the evaluation outcome, and treating a
                # late failure as "never ingested" makes the next arm duplicate
                # every session into the same banks.
                ingest_arm = not reusing
                result = benchmark.run(
                    model_id=LLM_MODEL.split("/")[-1],
                    provider_id="vertex",
                    api_url=mgr.get_url(profile),
                    max_questions_per_conversation=args.max_questions,
                    max_conversations=args.max_conversations,
                    save=False,
                    bank_ts=run_ts,
                    reuse_bank_ts=run_ts if reusing else None,
                )

                # A missing bank does not raise: recall 404s, quality.py
                # fabricates empty results, and the arm finishes with a low
                # score that looks like a model result. These two checks are the
                # difference between a failed arm and a fabricated finding.
                if ingest_arm:
                    ingested = True

                stats = probe_stats() if probe else {}
                if provider == "tei" and not stats.get("documents_scored"):
                    raise RuntimeError(
                        f"reranker scored 0 documents this arm (probe: {stats}); "
                        "the TEI path was never exercised, so the score is meaningless"
                    )
                # documents_scored proves TEI answered at least once, not that it
                # survived the arm. quality.py retries a failed recall twice and
                # then fabricates an empty result set, and recall failures count
                # toward no abort threshold, so a TEI that dies at question 10
                # finishes the arm with a depressed score and no error anywhere.
                # forward_errors is the probe's record of exactly that.
                if stats.get("forward_errors"):
                    raise RuntimeError(
                        f"reranker upstream failed {stats['forward_errors']} times mid-arm; "
                        "questions after the failure ranked nothing, so the score is depressed "
                        "by an outage rather than by the model"
                    )
                facts = result.get("stored_fact_tokens")
                if reusing:
                    # _count_stored_fact_tokens swallows a 404 and returns None,
                    # so a bank this arm cannot see reads as "count unavailable"
                    # rather than as an error. On a reuse arm that is precisely
                    # the symptom of the bank not being there, and the arm would
                    # otherwise finish with a confident low score.
                    if not facts:
                        raise RuntimeError(
                            f"reused bank stamp {run_ts} holds no readable facts "
                            f"(stored_fact_tokens={facts!r}); arm 1's bank is not visible here"
                        )
                elif facts == 0:
                    raise RuntimeError("ingest stored 0 fact tokens; recall had nothing to rank")

                results[arm] = result
                print(f"\n  {arm}: accuracy={result.get('accuracy')}% "
                      f"({result.get('correct')}/{result.get('total')})"
                      f"  reranked {stats.get('documents_scored', 0)} docs", flush=True)
                (RESULTS_DIR / f"{arm}.json").write_text(json.dumps(
                    {"arm": arm, "reranker_provider": provider, "reranker_model": model,
                     "max_candidates_cap": MAX_CANDIDATES, "llm": LLM_MODEL,
                     "bank_ts": run_ts, "probe": stats, **result}, indent=2))
            except Exception as exc:
                import traceback

                traceback.print_exc()
                print(f"  ARM FAILED {arm}: {type(exc).__name__}: {exc}", flush=True)
                results[arm] = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                mgr.stop(profile)
                if probe:
                    probe.terminate()
                    try:
                        probe.wait(timeout=30)
                    except Exception:
                        probe.kill()
                if tei_started:
                    try:
                        stop_tei()
                    except Exception as exc:
                        print(f"  stop_tei failed: {exc}", flush=True)
                time.sleep(3)
    finally:
        proxy.shutdown()

    print(f"\n{'=' * 60}")
    print(f"BEAM reranker comparison  cap={MAX_CANDIDATES} (low-budget retrieval)  {LLM_MODEL}")
    print(f"{'=' * 60}")
    for arm in arms:
        r = results.get(arm) or {}
        if "error" in r:
            print(f"  {arm:16s} FAILED  {r['error']}")
        elif r:
            print(f"  {arm:16s} {r.get('accuracy'):6.2f}%  {r.get('correct')}/{r.get('total')}")
        else:
            print(f"  {arm:16s} not run")
    print("\nDifferences under ~10 points are not resolvable at n=80. This run "
          "certifies the RRF gap and the absence of a large regression, nothing finer.")
    print(f"results -> {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
