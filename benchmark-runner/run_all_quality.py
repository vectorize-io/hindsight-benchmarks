#!/usr/bin/env python3
"""
Quality benchmark orchestrator — runs quality benchmarks sequentially for all cloud models.
Uses HindsightEmbedded (DaemonEmbedManager) to start/stop an embedded Hindsight instance
per model run, eliminating the need for a manually managed external API server.
"""
import json
import os
import sys
import time
from pathlib import Path

from hindsight_embed import DaemonEmbedManager

from hindsight_benchmark.quality import QualityBenchmark

BENCHMARK_RUNNER_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_RUNNER_DIR.parent / "results" / "leaderboard" / "llm"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Models to benchmark: (provider_id, model_id, hindsight_provider, hindsight_model, api_key)
MODELS = [
    # OpenAI
    ("openai", "gpt-4o-mini",       "openai", "gpt-4o-mini",       OPENAI_API_KEY),
    ("openai", "gpt-4.1-mini",      "openai", "gpt-4.1-mini",      OPENAI_API_KEY),
    ("openai", "gpt-4.1-nano",      "openai", "gpt-4.1-nano",      OPENAI_API_KEY),
    ("openai", "gpt-5-nano",        "openai", "gpt-5-nano",        OPENAI_API_KEY),
    ("openai", "gpt-5-mini",        "openai", "gpt-5-mini",        OPENAI_API_KEY),
    ("openai", "gpt-5.2",           "openai", "gpt-5.2",           OPENAI_API_KEY),
    ("openai", "gpt-5.4",           "openai", "gpt-5.4",           OPENAI_API_KEY),
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

BASE_CONFIG = {
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "local",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL": "BAAI/bge-small-en-v1.5",
    "HINDSIGHT_API_RERANKER_PROVIDER": "rrf",
    "HINDSIGHT_API_ENABLE_OBSERVATIONS": "false",
    "HINDSIGHT_API_EXTRACT_CAUSAL_LINKS": "false",
    "HINDSIGHT_API_LLM_TIMEOUT": "45",
    "HINDSIGHT_API_RETAIN_LLM_TIMEOUT": "120",
    "HINDSIGHT_API_DB_POOL_MIN_SIZE": "20",
    "HINDSIGHT_API_RERANKER_MAX_CANDIDATES": "30",
    "HINDSIGHT_API_DB_COMMAND_TIMEOUT": "15",
}


def make_config(hindsight_provider: str, hindsight_model: str, api_key: str, run_ts: int) -> dict:
    return {
        **BASE_CONFIG,
        "HINDSIGHT_API_LLM_PROVIDER": hindsight_provider,
        "HINDSIGHT_API_LLM_MODEL": hindsight_model,
        "HINDSIGHT_API_LLM_API_KEY": api_key,
        "HINDSIGHT_API_DATABASE_URL": f"pg0://quality-bench-{run_ts}",
    }


def _write_error(provider_id: str, model_id: str, message: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{provider_id}-{model_id}.json"
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing["model_id"] = model_id
    existing["provider_id"] = provider_id
    existing["quality"] = {
        "accuracy": 0.0, "correct": 0, "total": 0,
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

    benchmark = QualityBenchmark(
        gemini_api_key=GEMINI_API_KEY,
        openai_api_key=OPENAI_API_KEY,
    )
    mgr = DaemonEmbedManager()

    total = len(models_to_run)
    for i, (provider_id, model_id, h_provider, h_model, api_key) in enumerate(models_to_run, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {provider_id}/{model_id}")
        print(f"  Hindsight: provider={h_provider}, model={h_model}")
        print(f"{'='*60}")

        if not api_key:
            print(f"  SKIP: No API key available")
            continue

        run_ts = int(time.time())
        profile = f"qb-{run_ts}"
        config = make_config(h_provider, h_model, api_key, run_ts)

        print(f"  Starting Hindsight daemon (profile={profile})...")
        if not mgr.ensure_running(config, profile):
            print(f"  ERROR: Daemon did not start in time")
            _write_error(provider_id, model_id, "Daemon failed to start")
            continue

        api_url = mgr.get_url(profile)
        print(f"  Daemon ready at {api_url}. Running benchmark...")
        try:
            result = benchmark.run(model_id=model_id, provider_id=provider_id, api_url=api_url)
            print(f"\n  Result: accuracy={result.get('accuracy')}%, correct={result.get('correct')}/{result.get('total')}")
        except Exception as e:
            print(f"  ERROR running benchmark: {e}")
            _write_error(provider_id, model_id, str(e))
        finally:
            print(f"  Stopping daemon...")
            mgr.stop(profile)
            time.sleep(3)

    print(f"\n{'='*60}")
    print("All benchmarks complete!")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
