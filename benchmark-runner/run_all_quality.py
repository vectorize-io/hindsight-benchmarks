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
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
# Judge route: Gemini 3.7 Flash on Vertex AI, billed to GCP credits. Set to
# empty to fall back to GEMINI_API_KEY.
VERTEX_JUDGE_PROJECT = os.environ.get("VERTEX_JUDGE_PROJECT", "model-benchmark-506614")

# Model list is sourced from benchmark_models.json (single source of truth).
# Add a model with `rag-benchmark add ...` — no edits needed here.
# Tuple shape: (provider_id, model_id, hindsight_provider, hindsight_model, api_key)
from hindsight_benchmark.models_config import load_models

_QUALITY_MODELS = load_models("quality")
MODELS = [
    (m["provider_id"], m["model_id"], m["hindsight_provider"], m["hindsight_model"], m["api_key"])
    for m in _QUALITY_MODELS
]

# Models that need a custom LLM base URL (e.g. Ollama Cloud)
MODEL_BASE_URLS = {m["model_id"]: m["base_url"] for m in _QUALITY_MODELS if m["base_url"]}

# Models with a pinned reasoning effort ("none" disables thinking)
MODEL_REASONING = {m["model_id"]: m["reasoning_effort"] for m in _QUALITY_MODELS if m.get("reasoning_effort")}

BASE_CONFIG = {
    "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "local",
    "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL": "BAAI/bge-small-en-v1.5",
    "HINDSIGHT_API_RERANKER_PROVIDER": "rrf",
    "HINDSIGHT_API_ENABLE_OBSERVATIONS": "false",
    "HINDSIGHT_API_EXTRACT_CAUSAL_LINKS": "false",
    "HINDSIGHT_API_LLM_TIMEOUT": "45",
    # A reasoning model writes a thinking block before every extraction, so a
    # chunk that answers in 30s with thinking off can take minutes with it on.
    # The ceiling exists to catch a hung request, not to bound a slow one.
    "HINDSIGHT_API_RETAIN_LLM_TIMEOUT": "600",
    # 0 disables the per-task wall clock. A run we started knowingly should
    # finish rather than die at an arbitrary hour boundary.
    "HINDSIGHT_API_RETAIN_WALL_TIMEOUT": "0",
    # 32 concurrent extractions against one server queue behind each other, so
    # each request's latency grows with the queue and slow models time out on
    # waiting rather than on working. Extraction is per-chunk independent, so
    # this changes duration, not results.
    "HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT": "8",
    # BEAM sessions are ~43k tokens. The default 10k-token sub-batch budget
    # serializes a session into sequential extraction rounds; one big
    # sub-batch lets the chunks extract concurrently.
    # Identical for every model, so comparisons hold.
    "HINDSIGHT_API_RETAIN_BATCH_TOKENS": "50000",
    "HINDSIGHT_API_DB_POOL_MIN_SIZE": "20",
    "HINDSIGHT_API_RERANKER_MAX_CANDIDATES": "30",
    "HINDSIGHT_API_DB_COMMAND_TIMEOUT": "15",
}


def make_config(hindsight_provider: str, hindsight_model: str, api_key: str, run_ts: int, model_id: str = "") -> dict:
    config = {
        **BASE_CONFIG,
        "HINDSIGHT_API_LLM_PROVIDER": hindsight_provider,
        "HINDSIGHT_API_LLM_MODEL": hindsight_model,
        "HINDSIGHT_API_LLM_API_KEY": api_key,
        "HINDSIGHT_API_DATABASE_URL": f"pg0://quality-bench-{run_ts}",
    }
    if model_id in MODEL_BASE_URLS:
        config["HINDSIGHT_API_LLM_BASE_URL"] = MODEL_BASE_URLS[model_id]
    if model_id in MODEL_REASONING:
        config["HINDSIGHT_API_LLM_REASONING_EFFORT"] = MODEL_REASONING[model_id]
    return config


def _write_error(provider_id: str, model_id: str, message: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{provider_id}-{model_id}.json"
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing["model_id"] = model_id
    existing["provider_id"] = provider_id
    error_block = {
        "accuracy": 0.0, "correct": 0, "total": 0,
        "model_id": model_id, "provider_id": provider_id,
        "error": message,
    }
    # A completed benchmark result (marked by `dataset`) survives a later
    # failed run; the failure lands in quality_error instead.
    if (existing.get("quality") or {}).get("dataset"):
        existing["quality_error"] = error_block
    else:
        existing["quality"] = error_block
    path.write_text(json.dumps(existing, indent=2))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the quality benchmark for configured models")
    parser.add_argument("filter", nargs="?", default=None,
                        help="Only run models whose model_id matches (exact id or substring)")
    parser.add_argument("--max-conversations", type=int, default=None,
                        help="Limit to the first N dataset conversations (smoke tests)")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit to the first N questions per conversation (smoke tests)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write the result file (smoke tests)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models_to_run = MODELS
    if args.filter:
        from hindsight_benchmark.models_config import select_ids
        sel = select_ids([m[1] for m in MODELS], args.filter)
        models_to_run = [m for m in MODELS if m[1] in sel]
        print(f"Filtered to models matching '{args.filter}': {[m[1] for m in models_to_run]}")

    benchmark = QualityBenchmark(
        vertex_project=VERTEX_JUDGE_PROJECT or None,
        gemini_api_key=GEMINI_API_KEY,
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
        config = make_config(h_provider, h_model, api_key, run_ts, model_id=model_id)

        print(f"  Starting Hindsight daemon (profile={profile})...")
        if not mgr.ensure_running(config, profile):
            print(f"  ERROR: Daemon did not start in time")
            _write_error(provider_id, model_id, "Daemon failed to start")
            continue

        api_url = mgr.get_url(profile)
        print(f"  Daemon ready at {api_url}. Running benchmark...")
        try:
            result = benchmark.run(
                model_id=model_id, provider_id=provider_id, api_url=api_url,
                max_questions_per_conversation=args.max_questions,
                max_conversations=args.max_conversations,
                save=not args.no_save,
            )
            print(f"\n  Result: accuracy={result.get('accuracy')}%, correct={result.get('correct')}/{result.get('total')}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = f"{type(e).__name__}: {e}"
            print(f"  ERROR running benchmark: {msg}")
            _write_error(provider_id, model_id, msg)
        finally:
            print(f"  Stopping daemon...")
            mgr.stop(profile)
            time.sleep(3)

    print(f"\n{'='*60}")
    print("All benchmarks complete!")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
