#!/usr/bin/env python3
"""
Reranker benchmark orchestrator — Docker-based, shared data volume.

Architecture:
  Phase 0 (annotation, skipped if ground truth file already exists):
    - Start a Hindsight container with rrf reranker
    - Ingest conv-43 into a temp bank
    - Annotate ground truth: for each of 178 non-adversarial QA pairs,
      recall with budget=high and use LLM to mark relevant facts
    - Save to datasets/locomo_reranker_gt.json

  Phase 1 (ingest):
    - Start a fresh Hindsight container (rrf, shared volume)
    - Ingest conv-43 into a shared persistent bank (same pattern as reflect benchmark)
    - Stop container; data persists on volume

  Phase 2 (per reranker):
    - For each reranker config: start container with shared volume + reranker-specific env
    - Run recall() on all annotated questions, compute MRR and Recall@K
    - Save results to results/leaderboard/reranker/{reranker_id}.json
    - Stop container

Run all rerankers:
  PYTHONUNBUFFERED=1 uv run python3 -u run_all_reranker.py > /tmp/reranker_run.log 2>&1 &

Run a single reranker (substring match on reranker_id):
  uv run python3 -u run_all_reranker.py flashrank
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import requests
from testcontainers.core.container import DockerContainer

from hindsight_benchmark.reranker import GT_PATH, RerankerBenchmark
from hindsight_benchmark.quality import _parse_date

BENCHMARK_RUNNER_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_RUNNER_DIR.parent / "results" / "leaderboard" / "reranker"
DATASETS_DIR = BENCHMARK_RUNNER_DIR / "datasets"

HINDSIGHT_IMAGE = "ghcr.io/vectorize-io/hindsight:latest"
HINDSIGHT_PORT = 8888

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "")
ZEROENTROPY_API_KEY = os.environ.get("ZEROENTROPY_KEY", "")

# Constant retain model — always gemini-2.5-flash
RETAIN_PROVIDER = "gemini"
RETAIN_MODEL = "gemini-2.5-flash"
RETAIN_API_KEY = GEMINI_API_KEY

# Rerankers to benchmark: (reranker_id, provider, model_or_None)
RERANKERS = [
    ("rrf",                           "rrf",        None),
    ("local-minilm-l6",               "local",      "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ("flashrank",                     "flashrank",  "ms-marco-MiniLM-L-12-v2"),
    ("cohere-rerank-english-v3",        "cohere",       "rerank-english-v3.0"),
    ("cohere-rerank-multilingual-v3",   "cohere",       "rerank-multilingual-v3.0"),
    ("jina-reranker-v2-multilingual",   "litellm-sdk",    "jina_ai/jina-reranker-v2-base-multilingual"),
    ("voyage-rerank-2",                 "litellm-sdk",    "voyage/rerank-2"),
    ("voyage-rerank-2-lite",            "litellm-sdk",    "voyage/rerank-2-lite"),
    ("zerank-2",                        "zeroentropy",    "zerank-2"),
    ("zerank-2-small",                  "zeroentropy",    "zerank-2-small"),
    ("cohere-rerank-v4-pro",            "cohere",         "rerank-v4.0-pro"),
    ("cohere-rerank-v4-fast",           "cohere",         "rerank-v4.0-fast"),
    # Too slow on CPU in Docker (>3 min/question for 300 candidates):
    # ("local-bge-reranker-v2-m3",   "local",      "BAAI/bge-reranker-v2-m3"),
    # Requires `einops` not present in image (use litellm-sdk instead):
    # ("local-jina-reranker-v2",     "local",      "jinaai/jina-reranker-v2-base-multilingual"),
]

# Note: To run a subset:
#   uv run python3 -u run_all_reranker.py jina
#   uv run python3 -u run_all_reranker.py cohere

BASE_ENV = {
    # Set HOME so pg0 stores its data at /app/data/.pg0/ (the mounted volume)
    # Without this, pg0 defaults to /home/hindsight/.pg0/ inside the container
    # and data does NOT persist across container restarts.
    "HOME": "/app/data",
    "HINDSIGHT_ENABLE_CP": "false",
    "HINDSIGHT_API_HOST": "0.0.0.0",
    "HINDSIGHT_API_PORT": str(HINDSIGHT_PORT),
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "local",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL": "BAAI/bge-small-en-v1.5",
    "HINDSIGHT_API_RERANKER_PROVIDER": "rrf",
    "HINDSIGHT_API_ENABLE_OBSERVATIONS": "false",
    "HINDSIGHT_API_EXTRACT_CAUSAL_LINKS": "false",
    "HINDSIGHT_API_LLM_TIMEOUT": "45",
    "HINDSIGHT_API_DB_POOL_MIN_SIZE": "20",
    # Use 300 candidates for benchmark runs (realistic ranking conditions)
    "HINDSIGHT_API_RERANKER_MAX_CANDIDATES": "300",
    "HINDSIGHT_API_DB_COMMAND_TIMEOUT": "15",
    # Retain LLM: always gemini-2.5-flash (constant baseline)
    "HINDSIGHT_API_RETAIN_LLM_PROVIDER": RETAIN_PROVIDER,
    "HINDSIGHT_API_RETAIN_LLM_MODEL": RETAIN_MODEL,
    "HINDSIGHT_API_RETAIN_LLM_API_KEY": RETAIN_API_KEY,
    "HINDSIGHT_API_RETAIN_LLM_TIMEOUT": "120",
    # LLM fallback
    "HINDSIGHT_API_LLM_PROVIDER": RETAIN_PROVIDER,
    "HINDSIGHT_API_LLM_MODEL": RETAIN_MODEL,
    "HINDSIGHT_API_LLM_API_KEY": RETAIN_API_KEY,
}


def _make_reranker_env(provider: str, model: str = None) -> dict:
    env = {**BASE_ENV, "HINDSIGHT_API_RERANKER_PROVIDER": provider}
    if provider == "local" and model:
        env["HINDSIGHT_API_RERANKER_LOCAL_MODEL"] = model
        env["HINDSIGHT_API_LAZY_RERANKER"] = "true"  # avoid startup delay
        if "jina" in model.lower():
            env["HINDSIGHT_API_RERANKER_LOCAL_TRUST_REMOTE_CODE"] = "true"
    if provider == "flashrank" and model:
        env["HINDSIGHT_API_RERANKER_FLASHRANK_MODEL"] = model
    if provider == "cohere":
        env["HINDSIGHT_API_RERANKER_COHERE_API_KEY"] = COHERE_API_KEY
        if model:
            env["HINDSIGHT_API_RERANKER_COHERE_MODEL"] = model
    if provider == "litellm-sdk":
        if model:
            env["HINDSIGHT_API_RERANKER_LITELLM_SDK_MODEL"] = model
        if "jina" in (model or "").lower():
            env["HINDSIGHT_API_RERANKER_LITELLM_SDK_API_KEY"] = JINA_API_KEY
        elif "voyage" in (model or "").lower():
            env["HINDSIGHT_API_RERANKER_LITELLM_SDK_API_KEY"] = VOYAGE_API_KEY
    if provider == "zeroentropy":
        env["HINDSIGHT_API_RERANKER_ZEROENTROPY_API_KEY"] = ZEROENTROPY_API_KEY
        if model:
            env["HINDSIGHT_API_RERANKER_ZEROENTROPY_MODEL"] = model
    return env


def _start_hindsight(data_dir: str, env: dict) -> tuple[DockerContainer, str]:
    """Start a Hindsight container with the given data volume and env, wait for health."""
    c = DockerContainer(HINDSIGHT_IMAGE)
    c.with_volume_mapping(data_dir, "/app/data", "rw")
    c.with_exposed_ports(HINDSIGHT_PORT)
    for k, v in env.items():
        c.with_env(k, v)
    c.start()
    port = c.get_exposed_port(HINDSIGHT_PORT)
    api_url = f"http://localhost:{port}"
    print(f"  Container started on {api_url}, waiting for health...")
    _wait_health(api_url)
    return c, api_url


def _wait_health(api_url: str, timeout: int = 300):
    """Poll /health until 200 or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200:
                print(f"  ✓ Hindsight healthy at {api_url}")
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError(f"Hindsight did not become healthy at {api_url} within {timeout}s")


def _ingest_shared_bank(api_url: str, run_ts: int) -> str:
    """Ingest LoComo conv-43 into the shared bank. Returns the bank_id."""
    from hindsight_client import Hindsight

    with open(DATASETS_DIR / "locomo_quality.json") as f:
        dataset = json.load(f)

    item = next((i for i in dataset if i["sample_id"] == "conv-43"), None)
    if item is None:
        raise ValueError("conv-43 not found in dataset")

    client = Hindsight(base_url=api_url)
    bank_id = f"rrb_shared_conv43_{run_ts}"
    print(f"\nCreating shared bank: {bank_id}")
    client.create_bank(bank_id=bank_id)

    conv = item["conversation"]
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]
    session_keys = sorted(k for k in conv if k.startswith("session_") and not k.endswith("_date_time"))

    print("Ingesting conversation (retain with gemini-2.5-flash, once for all rerankers)...")
    for session_key in session_keys:
        if not isinstance(conv.get(session_key), list):
            continue
        session_date = _parse_date(conv[f"{session_key}_date_time"])
        client.retain(
            bank_id=bank_id,
            content=json.dumps(conv[session_key]),
            context=f"Conversation between {speaker_a} and {speaker_b} ({session_key})",
            timestamp=session_date,
            document_id=f"conv43_{session_key}_{run_ts}",
        )
        print(f"  ✓ Ingested {session_key}", flush=True)

    return bank_id


def _load_gt_meta() -> tuple[str, str] | tuple[None, None]:
    """Read data_dir and bank_id saved in the GT file. Returns (None, None) if not present."""
    if not GT_PATH.exists():
        return None, None
    with open(GT_PATH) as f:
        gt = json.load(f)
    data_dir = gt.get("data_dir")
    bank_id = gt.get("bank_id")
    if data_dir and bank_id and Path(data_dir).exists():
        return data_dir, bank_id
    return None, None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    filter_reranker = sys.argv[1] if len(sys.argv) > 1 else None
    rerankers_to_run = RERANKERS
    if filter_reranker:
        rerankers_to_run = [r for r in RERANKERS if filter_reranker in r[0]]
        print(f"Filtered to rerankers matching '{filter_reranker}': {[r[0] for r in rerankers_to_run]}")

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is required (used for retain LLM and annotation)")
        sys.exit(1)

    benchmark = RerankerBenchmark(gemini_api_key=GEMINI_API_KEY)
    run_ts = int(time.time())

    # Check if we can reuse an existing bank from a previous run
    saved_data_dir, saved_bank_id = _load_gt_meta()
    if saved_data_dir and saved_bank_id:
        data_dir = saved_data_dir
        bank_id = saved_bank_id
        print(f"\nReusing persistent data directory from GT: {data_dir}")
        print(f"Reusing bank: {bank_id}")
        print(f"\n{'='*60}")
        print(f"Phase 0: Ground truth already exists at {GT_PATH}, skipping annotation")
        print(f"Phase 1: Skipping ingest — reusing existing bank {bank_id}")
        print(f"{'='*60}")
    else:
        # Fresh run: ingest + annotate
        data_dir = tempfile.mkdtemp(prefix="hindsight-reranker-bench-")
        print(f"\nUsing new persistent data directory: {data_dir}")

        # ── Phase 1: ingest shared bank ──────────────────────────────────────────
        # Must run before annotation so GT fact texts come from the same bank
        # that Phase 2 benchmark runs will query against.
        print(f"\n{'='*60}")
        print("Phase 1: Starting Hindsight container and ingesting shared bank")
        print(f"  Retain LLM: {RETAIN_PROVIDER}/{RETAIN_MODEL} (constant baseline)")
        print(f"{'='*60}")

        retain_env = {**BASE_ENV, "HINDSIGHT_API_RERANKER_PROVIDER": "rrf"}
        retain_container = None
        bank_id = None
        try:
            retain_container, retain_url = _start_hindsight(data_dir, retain_env)
            bank_id = _ingest_shared_bank(retain_url, run_ts)

            # ── Phase 0: annotate ground truth (uses the same shared bank) ───────
            print(f"\n{'='*60}")
            print("Phase 0: Annotating ground truth (one-time step, using shared bank)")
            print(f"  GT path: {GT_PATH}")
            print(f"  Bank: {bank_id} (same bank as Phase 2 — fact texts will match)")
            print(f"{'='*60}")
            benchmark.create_annotations(retain_url, bank_id, GT_PATH)

            # Save data_dir and bank_id into the GT file for future runs
            with open(GT_PATH) as f:
                gt_data = json.load(f)
            gt_data["data_dir"] = data_dir
            gt_data["bank_id"] = bank_id
            with open(GT_PATH, "w") as f:
                json.dump(gt_data, f, indent=2)
            print(f"  Saved data_dir and bank_id to GT file for future reuse")
        except Exception as e:
            print(f"ERROR in Phase 1: {e}")
            if retain_container:
                retain_container.stop()
            return
        finally:
            if retain_container:
                print("Stopping retain container...")
                retain_container.stop()
                print(f"  Retain container stopped. Data persists at {data_dir}")

    print(f"\nShared bank ready: {bank_id}")

    # ── Phase 2: run each reranker against the shared bank ─────────────────────
    total = len(rerankers_to_run)
    for i, (reranker_id, provider, model) in enumerate(rerankers_to_run, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {reranker_id}")
        print(f"  Provider: {provider}, Model: {model or 'default'}")
        print(f"  Bank: {bank_id} (shared, volume persists)")
        print(f"{'='*60}")

        if provider == "cohere" and not COHERE_API_KEY:
            print("  SKIP: COHERE_API_KEY not set")
            continue
        if provider == "litellm-sdk" and "jina" in (model or "").lower() and not JINA_API_KEY:
            print("  SKIP: JINA_API_KEY not set")
            continue
        if provider == "litellm-sdk" and "voyage" in (model or "").lower() and not VOYAGE_API_KEY:
            print("  SKIP: VOYAGE_API_KEY not set")
            continue
        if provider == "zeroentropy" and not ZEROENTROPY_API_KEY:
            print("  SKIP: ZEROENTROPY_KEY not set")
            continue

        reranker_container = None
        try:
            env = _make_reranker_env(provider, model)
            reranker_container, api_url = _start_hindsight(data_dir, env)
            print(f"  Container ready at {api_url}. Running reranker benchmark...")
            result = benchmark.run_with_bank(
                reranker_id=reranker_id,
                provider=provider,
                model=model,
                api_url=api_url,
                bank_id=bank_id,
                gt_path=GT_PATH,
            )
            print(
                f"\n  Result: MRR={result.get('mrr')}, "
                f"R@1={result.get('recall_at_1')}, "
                f"R@3={result.get('recall_at_3')}, "
                f"R@5={result.get('recall_at_5')}, "
                f"avg_latency={result.get('avg_latency_s')}s"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            if reranker_container is not None:
                print("  Stopping reranker container...")
                reranker_container.stop()

    print(f"\n{'='*60}")
    print("All reranker benchmarks complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Data dir (can be removed): {data_dir}")


if __name__ == "__main__":
    main()
