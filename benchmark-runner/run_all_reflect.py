#!/usr/bin/env python3
"""
Reflect benchmark orchestrator — Docker-based, stable, shared data volume.

Architecture:
  1. Create a host directory as the persistent Hindsight data volume.
  2. Start a Hindsight Docker container (ghcr.io/vectorize-io/hindsight:latest)
     with that volume mounted to /app/data. No external DB needed — Hindsight
     uses its embedded pg0 database stored in /app/data.
  3. Ingest all LoComo conv-43 sessions into a shared bank (once).
  4. For each model under test:
       a. Stop the Hindsight container.
       b. Start a fresh Hindsight container with the SAME data volume mounted,
          but a different REFLECT_LLM config (provider/model/key).
       c. Run 242 reflect() calls against the shared bank.
       d. Save results.

The data volume persists across container restarts — no re-ingestion per model.
All pg0 / shared-memory / port-conflict issues are solved by using Docker.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import requests
from testcontainers.core.container import DockerContainer

from hindsight_benchmark.reflect import ReflectBenchmark
from hindsight_benchmark.quality import _parse_date

BENCHMARK_RUNNER_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_RUNNER_DIR.parent / "results" / "leaderboard" / "llm"
DATASETS_DIR = BENCHMARK_RUNNER_DIR / "datasets"

HINDSIGHT_IMAGE = "ghcr.io/vectorize-io/hindsight:latest"
HINDSIGHT_PORT = 8888

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Constant retain model — always gemini-2.5-flash
RETAIN_PROVIDER = "gemini"
RETAIN_MODEL = "gemini-2.5-flash"
RETAIN_API_KEY = GEMINI_API_KEY

# Models to benchmark: (provider_id, model_id, reflect_provider, reflect_model, reflect_api_key)
MODELS = [
    # OpenAI
    ("openai", "gpt-4o-mini",       "openai", "gpt-4o-mini",       OPENAI_API_KEY),
    ("openai", "gpt-4.1-mini",      "openai", "gpt-4.1-mini",      OPENAI_API_KEY),
    ("openai", "gpt-4.1-nano",      "openai", "gpt-4.1-nano",      OPENAI_API_KEY),
    ("openai", "gpt-5-nano",        "openai", "gpt-5-nano",        OPENAI_API_KEY),
    ("openai", "gpt-5-mini",        "openai", "gpt-5-mini",        OPENAI_API_KEY),
    ("openai", "gpt-5.2",           "openai", "gpt-5.2",           OPENAI_API_KEY),
    ("openai", "gpt-5.4",           "openai", "gpt-5.4",           OPENAI_API_KEY),
    ("openai", "gpt-5.4-mini",      "openai", "gpt-5.4-mini",      OPENAI_API_KEY),
    ("openai", "gpt-5.4-nano",      "openai", "gpt-5.4-nano",      OPENAI_API_KEY),
    # Groq
    ("groq", "openai-gpt-oss-20b",       "groq", "openai/gpt-oss-20b",       GROQ_API_KEY),
    ("groq", "openai-gpt-oss-120b",      "groq", "openai/gpt-oss-120b",      GROQ_API_KEY),
    ("groq", "llama-3.1-8b-instant",     "groq", "llama-3.1-8b-instant",     GROQ_API_KEY),
    ("groq", "llama-3.3-70b-versatile",  "groq", "llama-3.3-70b-versatile",  GROQ_API_KEY),
    # Gemini
    ("gemini", "gemini-2.5-flash",       "gemini", "gemini-2.5-flash",       GEMINI_API_KEY),
    ("gemini", "gemini-2.5-flash-lite",  "gemini", "gemini-2.5-flash-lite",  GEMINI_API_KEY),
    ("gemini", "gemini-3-flash-preview", "gemini", "gemini-3-flash-preview", GEMINI_API_KEY),
]

BASE_ENV = {
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
    "HINDSIGHT_API_RERANKER_MAX_CANDIDATES": "30",
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


def _make_env(reflect_provider: str, reflect_model: str, reflect_api_key: str) -> dict:
    return {
        **BASE_ENV,
        "HINDSIGHT_API_REFLECT_LLM_PROVIDER": reflect_provider,
        "HINDSIGHT_API_REFLECT_LLM_MODEL": reflect_model,
        "HINDSIGHT_API_REFLECT_LLM_API_KEY": reflect_api_key,
        "HINDSIGHT_API_REFLECT_LLM_TIMEOUT": "60",
    }


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
    bank_id = f"rb_shared_conv43_{run_ts}"
    print(f"\nCreating shared bank: {bank_id}")
    client.create_bank(bank_id=bank_id)

    conv = item["conversation"]
    speaker_a = conv["speaker_a"]
    speaker_b = conv["speaker_b"]
    session_keys = sorted(k for k in conv if k.startswith("session_") and not k.endswith("_date_time"))

    print("Ingesting conversation (retain with gemini-2.5-flash, once for all models)...")
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


def _write_error(provider_id: str, model_id: str, message: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{provider_id}-{model_id}.json"
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing["model_id"] = model_id
    existing["provider_id"] = provider_id
    existing["reflect"] = {
        "accuracy": 0.0, "correct": 0, "total": 0,
        "avg_latency_s": 0.0,
        "model_id": model_id, "provider_id": provider_id,
        "error": message,
    }
    path.write_text(json.dumps(existing, indent=2))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    filter_model = sys.argv[1] if len(sys.argv) > 1 else None
    models_to_run = MODELS
    if filter_model:
        models_to_run = [m for m in MODELS if filter_model in m[1]]
        print(f"Filtered to models matching '{filter_model}': {[m[1] for m in models_to_run]}")

    benchmark = ReflectBenchmark(
        gemini_api_key=GEMINI_API_KEY,
        openai_api_key=OPENAI_API_KEY,
    )
    run_ts = int(time.time())

    # Persistent data directory — mounted to /app/data in every container.
    # The embedded pg0 database and bank data live here across container restarts.
    data_dir = tempfile.mkdtemp(prefix="hindsight-reflect-bench-")
    print(f"\nUsing persistent data directory: {data_dir}")

    # ── Phase 1: ingest once ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Phase 1: Starting Hindsight container and ingesting shared bank")
    print(f"  Retain LLM: {RETAIN_PROVIDER}/{RETAIN_MODEL} (constant baseline)")
    print(f"{'='*60}")

    retain_env = _make_env(RETAIN_PROVIDER, RETAIN_MODEL, RETAIN_API_KEY)
    retain_container = None
    try:
        retain_container, retain_url = _start_hindsight(data_dir, retain_env)
        bank_id = _ingest_shared_bank(retain_url, run_ts)
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

    # ── Phase 2: run each model against the shared bank ───────────────────────
    total = len(models_to_run)
    for i, (provider_id, model_id, r_provider, r_model, r_api_key) in enumerate(models_to_run, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {provider_id}/{model_id}")
        print(f"  Reflect LLM: provider={r_provider}, model={r_model}")
        print(f"  Bank: {bank_id} (shared, volume persists)")
        print(f"{'='*60}")

        if not r_api_key:
            print("  SKIP: No API key available")
            continue

        model_container = None
        try:
            env = _make_env(r_provider, r_model, r_api_key)
            model_container, api_url = _start_hindsight(data_dir, env)
            print(f"  Container ready at {api_url}. Running reflect benchmark...")
            result = benchmark.run_with_bank(
                model_id=model_id,
                provider_id=provider_id,
                api_url=api_url,
                bank_id=bank_id,
            )
            print(f"\n  Result: accuracy={result.get('accuracy')}%, correct={result.get('correct')}/{result.get('total')}, avg_latency={result.get('avg_latency_s')}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            _write_error(provider_id, model_id, str(e))
        finally:
            if model_container is not None:
                print("  Stopping model container...")
                model_container.stop()

    print(f"\n{'='*60}")
    print("All reflect benchmarks complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Data dir (can be removed): {data_dir}")


if __name__ == "__main__":
    main()
