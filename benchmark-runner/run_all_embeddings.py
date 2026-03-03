#!/usr/bin/env python3
"""
Embeddings benchmark orchestrator — Docker-based, one persistent volume per embedding model.

Each embedding model gets its own Docker volume because the Hindsight DB schema bakes in
vector dimensions — models with different dimensions cannot share a bank.

Each model also gets its own ground truth annotation file (datasets/locomo_embeddings_gt_{id}.json)
because the retain LLM is non-deterministic: fact texts differ per ingest, so a cross-bank GT
would produce false "not found" results on exact string matching.

Architecture per embedding model:
  Phase 1 (ingest + annotate, once — skipped if metadata + GT already exist):
    - Start a Hindsight container with the target embedding model + RRF reranker
    - Ingest conv-43 into a persistent bank
    - Annotate ground truth: recall(budget="high") + LLM relevance labelling per question
    - Save data_dir + bank_id to datasets/embedding_banks.json
    - Save GT to datasets/locomo_embeddings_gt_{embedding_id}.json

  Phase 2 (benchmark):
    - Start container with same embedding model + fixed MiniLM-L6 reranker
    - Run recall(budget="mid") on all annotated questions
    - Compute MRR and Recall@K metrics against per-model GT
    - Save results to results/leaderboard/embeddings/{embedding_id}.json

Run all embedding models:
  PYTHONUNBUFFERED=1 uv run python3 -u run_all_embeddings.py > /tmp/embeddings_run.log 2>&1 &

Run a single embedding model (substring match on embedding_id):
  uv run python3 -u run_all_embeddings.py bge-small-en-v1.5
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import requests
from testcontainers.core.container import DockerContainer

from hindsight_benchmark.embeddings import (
    EmbeddingsBenchmark, gt_path_for_model,
    FIXED_RERANKER_PROVIDER, FIXED_RERANKER_MODEL,
)

BENCHMARK_RUNNER_DIR = Path(__file__).parent
RESULTS_DIR = BENCHMARK_RUNNER_DIR.parent / "results" / "leaderboard" / "embeddings"

HINDSIGHT_IMAGE = "ghcr.io/vectorize-io/hindsight:latest"
HINDSIGHT_PORT = 8888

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")

RETAIN_PROVIDER = "gemini"
RETAIN_MODEL = "gemini-2.5-flash"
RETAIN_API_KEY = GEMINI_API_KEY

# Embedding models to benchmark: (embedding_id, provider, model)
# All models must produce exactly 384-dimensional vectors (Hindsight DB constraint).
EMBEDDING_MODELS = [
    # Local models (no API key required, run on CPU in Docker)
    ("bge-small-en-v1.5",  "local", "BAAI/bge-small-en-v1.5"),
    ("all-minilm-l6-v2",   "local", "sentence-transformers/all-MiniLM-L6-v2"),
]


def _base_env(embedding_provider: str, embedding_model: str) -> dict:
    """Build environment variables for a Hindsight container with the given embedding model."""
    env = {
        "HOME": "/app/data",
        "HINDSIGHT_ENABLE_CP": "false",
        "HINDSIGHT_API_HOST": "0.0.0.0",
        "HINDSIGHT_API_PORT": str(HINDSIGHT_PORT),
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER": embedding_provider,
        "HINDSIGHT_API_RERANKER_PROVIDER": "rrf",
        "HINDSIGHT_API_ENABLE_OBSERVATIONS": "false",
        "HINDSIGHT_API_EXTRACT_CAUSAL_LINKS": "false",
        "HINDSIGHT_API_LLM_TIMEOUT": "45",
        "HINDSIGHT_API_DB_POOL_MIN_SIZE": "20",
        "HINDSIGHT_API_RERANKER_MAX_CANDIDATES": "300",
        "HINDSIGHT_API_DB_COMMAND_TIMEOUT": "15",
        "HINDSIGHT_API_RETAIN_LLM_PROVIDER": RETAIN_PROVIDER,
        "HINDSIGHT_API_RETAIN_LLM_MODEL": RETAIN_MODEL,
        "HINDSIGHT_API_RETAIN_LLM_API_KEY": RETAIN_API_KEY,
        "HINDSIGHT_API_RETAIN_LLM_TIMEOUT": "120",
        "HINDSIGHT_API_LLM_PROVIDER": RETAIN_PROVIDER,
        "HINDSIGHT_API_LLM_MODEL": RETAIN_MODEL,
        "HINDSIGHT_API_LLM_API_KEY": RETAIN_API_KEY,
    }
    if embedding_provider == "local" and embedding_model:
        env["HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL"] = embedding_model
    elif embedding_provider == "openai" and embedding_model:
        env["HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL"] = embedding_model
        env["HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY"] = OPENAI_API_KEY
    elif embedding_provider == "cohere" and embedding_model:
        env["HINDSIGHT_API_EMBEDDINGS_COHERE_MODEL"] = embedding_model
        env["HINDSIGHT_API_EMBEDDINGS_COHERE_API_KEY"] = COHERE_API_KEY
    return env


def _benchmark_env(embedding_provider: str, embedding_model: str) -> dict:
    """Build env for a benchmark run: same embedding model + fixed MiniLM-L6 reranker."""
    env = _base_env(embedding_provider, embedding_model)
    env["HINDSIGHT_API_RERANKER_PROVIDER"] = FIXED_RERANKER_PROVIDER
    env["HINDSIGHT_API_RERANKER_LOCAL_MODEL"] = FIXED_RERANKER_MODEL
    env["HINDSIGHT_API_LAZY_RERANKER"] = "true"
    return env


def _start_hindsight(data_dir: str, env: dict) -> tuple[DockerContainer, str]:
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    filter_model = sys.argv[1] if len(sys.argv) > 1 else None
    models_to_run = EMBEDDING_MODELS
    if filter_model:
        models_to_run = [m for m in EMBEDDING_MODELS if filter_model in m[0]]
        print(f"Filtered to models matching '{filter_model}': {[m[0] for m in models_to_run]}")

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is required (used for retain LLM and GT annotation)")
        sys.exit(1)

    benchmark = EmbeddingsBenchmark(gemini_api_key=GEMINI_API_KEY)
    run_ts = int(time.time())

    total = len(models_to_run)
    for i, (embedding_id, provider, model) in enumerate(models_to_run, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {embedding_id}")
        print(f"  Provider: {provider}, Model: {model}")
        print(f"  Fixed reranker: {FIXED_RERANKER_MODEL}")
        print(f"{'='*60}")

        if provider == "openai" and not OPENAI_API_KEY:
            print("  SKIP: OPENAI_API_KEY not set")
            continue
        if provider == "cohere" and not COHERE_API_KEY:
            print("  SKIP: COHERE_API_KEY not set")
            continue

        # Check if we already have an ingested bank AND annotated GT for this model
        saved_data_dir, saved_bank_id = benchmark.get_bank_for_model(embedding_id)
        gt_path = gt_path_for_model(embedding_id)

        if saved_data_dir and saved_bank_id and gt_path.exists():
            data_dir = saved_data_dir
            bank_id = saved_bank_id
            print(f"  Reusing existing bank: {bank_id} (data_dir: {data_dir})")
            print(f"  GT already exists at {gt_path}")
        else:
            # Fresh ingest + annotate for this embedding model
            data_dir = tempfile.mkdtemp(prefix=f"hindsight-emb-{embedding_id[:20]}-")
            print(f"\n  Phase 1: Ingesting + annotating, volume: {data_dir}")

            ingest_env = _base_env(provider, model)
            ingest_container = None
            bank_id = None
            try:
                ingest_container, ingest_url = _start_hindsight(data_dir, ingest_env)
                bank_id = benchmark.ingest_bank(ingest_url, embedding_id, run_ts)
                benchmark.save_bank_for_model(embedding_id, data_dir, bank_id)
                print(f"  Bank {bank_id} saved. Starting GT annotation...")
                benchmark.create_annotations(ingest_url, bank_id, embedding_id)
                print(f"  Annotation complete.")
            except Exception as e:
                print(f"  ERROR during ingest/annotate: {e}")
                import traceback; traceback.print_exc()
                if ingest_container:
                    ingest_container.stop()
                continue
            finally:
                if ingest_container:
                    print("  Stopping ingest container...")
                    ingest_container.stop()

        # Phase 2: run benchmark with fixed MiniLM-L6 reranker
        print(f"\n  Phase 2: Running benchmark (bank={bank_id})")
        bench_container = None
        try:
            bench_env = _benchmark_env(provider, model)
            bench_container, bench_url = _start_hindsight(data_dir, bench_env)
            print(f"  Container ready at {bench_url}. Running embeddings benchmark...")
            result = benchmark.run_with_bank(
                embedding_id=embedding_id,
                provider=provider,
                model=model,
                api_url=bench_url,
                bank_id=bank_id,
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
            import traceback; traceback.print_exc()
        finally:
            if bench_container is not None:
                print("  Stopping benchmark container...")
                bench_container.stop()

    print(f"\n{'='*60}")
    print("All embedding benchmarks complete!")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
