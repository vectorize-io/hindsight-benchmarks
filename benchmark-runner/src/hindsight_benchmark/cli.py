"""
CLI entry point for the Hindsight benchmark.

Usage:
  hindsight-benchmark                          # run all registry models
  hindsight-benchmark --model qwen2.5-1.5b-instruct
  hindsight-benchmark --mlx-model /path/to/model --name "My Model" --model-id my-model
  hindsight-benchmark --url http://localhost:8080 --name "Remote Model" --model-id remote
"""

import argparse
import os
import statistics
import sys
from pathlib import Path

from .benchmark import run_mlx, run_url, run_gemini, save_run, load_latest_run, DATASETS_DIR

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    # Try multiple locations for .env file
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / ".env",  # locallm/.env
        Path.cwd() / ".env",  # current directory
        Path.cwd().parent / ".env",  # parent of current directory
    ]
    for env_path in possible_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            break
except ImportError:
    pass


def _print_summary(run):
    ok = [t for t in run.tests if t.success and t.valid_json]
    wall_s = run.wall_s
    print(f"\n  Success     : {len(ok)}/{len(run.tests)}")
    print(f"  Concurrency : {run.concurrency}")
    print(f"  Wall time   : {wall_s:.2f}s")
    if ok and wall_s > 0:
        print(f"  Throughput  : {len(ok) / wall_s:.2f} req/s")
        print(f"  Avg latency : {statistics.mean(t.latency_s for t in ok):.2f}s")
        total_completion = sum(t.completion_tokens for t in ok)
        total_prompt = sum(t.prompt_tokens for t in ok)
        total_facts = sum(t.num_facts for t in ok)
        if total_completion:
            print(f"  Gen tok/s   : {total_completion / wall_s:.1f}  ({total_completion} completion tokens)")
        if total_prompt and total_completion:
            print(f"  Total tok/s : {(total_prompt + total_completion) / wall_s:.1f}  ({total_prompt + total_completion} total tokens)")
            ratio = total_completion / total_prompt
            print(f"  Out/In ratio: {ratio:.3f}  (avg {total_completion/len(ok):.0f} out / {total_prompt/len(ok):.0f} in)")
        avg_facts = statistics.mean(t.num_facts for t in ok)
        print(f"  Avg facts   : {avg_facts:.1f}")
        if total_facts and total_completion:
            tokens_per_fact = total_completion / total_facts
            print(f"  Toks/fact   : {tokens_per_fact:.1f}  (efficiency metric)")
    for t in run.tests:
        retry_str = f"  [{t.retries} retr{'y' if t.retries == 1 else 'ies'}]" if t.retries else ""
        status = f"{t.latency_s:.2f}s  {t.num_facts} facts{retry_str}" if t.success else f"FAIL  {t.error[:80]}"
        print(f"  Test {t.test_index}: {status}")


def main():
    p = argparse.ArgumentParser(description="Hindsight fact extraction benchmark")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--run-all", action="store_true",
                       help="Run all models defined in benchmark_models.json")
    group.add_argument("--model", metavar="MODEL_ID",
                       help="Run a specific llama0 registry model")
    group.add_argument("--mlx-model", metavar="PATH",
                       help="Path to an MLX model directory (uses mlx_lm.server)")
    group.add_argument("--url", metavar="URL",
                       help="URL of an already-running OpenAI-compatible server (e.g. http://localhost:8080)")
    group.add_argument("--groq-model", metavar="MODEL",
                       help="Groq model name (e.g. llama3-groq-70b-8192-tool-use-preview)")
    group.add_argument("--gemini-model", metavar="MODEL",
                       help="Gemini model name (e.g. gemini-2.0-flash-exp)")
    p.add_argument("--name", metavar="NAME",
                   help="Display name for the model (required with --mlx-model, --url, --groq-model, or --gemini-model)")
    p.add_argument("--model-id", metavar="ID",
                   help="Result file slug (defaults to --name lowercased)")
    p.add_argument("--port", type=int, default=None,
                   help="Server port (default: 9500 for llama0, 9600 for mlx)")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Number of parallel requests to send simultaneously (default: 1)")
    p.add_argument("--api-key", metavar="KEY", default="",
                   help="API key for Authorization: Bearer header (required for --groq-model, optional for --url)")
    p.add_argument("--api-model", metavar="MODEL", default="",
                   help="Model name to send in API request (for --url, e.g. 'ministral-3:14b-cloud' for Ollama)")
    p.add_argument("--provider-id", metavar="PROVIDER", default="",
                   help="Provider identifier (e.g., 'openai', 'groq', 'ollama-cloud', 'local'). Auto-detected if not specified.")
    available_datasets = sorted(p.stem for p in DATASETS_DIR.glob("*.json"))
    p.add_argument("--dataset", required=True, choices=available_datasets,
                   help=f"Dataset to benchmark against. Available: {', '.join(available_datasets)}")
    args = p.parse_args()

    # Handle --run-all
    if args.run_all:
        import json
        config_path = Path(__file__).parent.parent.parent / "benchmark_models.json"
        if not config_path.exists():
            print(f"Error: {config_path} not found")
            sys.exit(1)

        with open(config_path) as f:
            config = json.load(f)

        print(f"\nRunning {len(config['models'])} models from {config_path}")
        print(f"Dataset: {args.dataset}, Concurrency: {args.concurrency}\n")

        for model_config in config['models']:
            provider_id = model_config["provider_id"]
            model_id = model_config["model_id"]
            model_name = model_config["model_name"]
            method = model_config["method"]

            try:
                if method == "gemini":
                    api_key = os.getenv(model_config["api_key_env"])
                    if not api_key:
                        print(f"Skipping {model_name}: {model_config['api_key_env']} not set")
                        continue
                    gemini_model = model_config["gemini_model"]
                    run = run_gemini(gemini_model, model_id=model_id, model_name=model_name,
                                   provider_id=provider_id, concurrency=args.concurrency,
                                   dataset=args.dataset, api_key=api_key)
                    save_run(provider_id, model_id, run)
                    _print_summary(run)

                elif method == "groq":
                    api_key = os.getenv(model_config["api_key_env"])
                    if not api_key:
                        print(f"Skipping {model_name}: {model_config['api_key_env']} not set")
                        continue
                    groq_model = model_config["groq_model"]
                    groq_url = "https://api.groq.com/openai"
                    run = run_url(groq_url, model_id=model_id, model_name=model_name,
                                provider_id=provider_id, concurrency=args.concurrency,
                                dataset=args.dataset, api_key=api_key, model=groq_model)
                    save_run(provider_id, model_id, run)
                    _print_summary(run)

                elif method == "url":
                    # API key is optional for local Ollama
                    api_key = ""
                    if "api_key_env" in model_config:
                        api_key = os.getenv(model_config["api_key_env"]) or ""
                        if not api_key:
                            print(f"Skipping {model_name}: {model_config['api_key_env']} not set")
                            continue
                    url = model_config["url"]
                    api_model = model_config["api_model"]
                    run = run_url(url, model_id=model_id, model_name=model_name,
                                provider_id=provider_id, concurrency=args.concurrency,
                                dataset=args.dataset, api_key=api_key, model=api_model)
                    save_run(provider_id, model_id, run)
                    _print_summary(run)

            except Exception as e:
                print(f"Error running {model_name}: {e}")
                continue

        print(f"\nAll benchmarks complete. Results saved to: {Path(__file__).parent.parent.parent / 'results'}")
        return

    if args.gemini_model:
        api_key = args.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            p.error("--api-key or GEMINI_API_KEY environment variable is required with --gemini-model")
        name = args.name or args.gemini_model
        model_id = args.model_id or args.gemini_model.lower().replace(" ", "-").replace("/", "-")
        provider_id = args.provider_id or "gemini"
        run = run_gemini(args.gemini_model, model_id=model_id, model_name=name, provider_id=provider_id, concurrency=args.concurrency, dataset=args.dataset, api_key=api_key)
        save_run(provider_id, model_id, run)
        _print_summary(run)
        return

    if args.groq_model:
        api_key = args.api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            p.error("--api-key or GROQ_API_KEY environment variable is required with --groq-model (get yours at https://console.groq.com)")
        name = args.name or args.groq_model
        model_id = args.model_id or args.groq_model.lower().replace(" ", "-").replace("/", "-")
        provider_id = args.provider_id or "groq"
        # Groq endpoint (without /v1 since _call_inference adds /v1/chat/completions)
        groq_url = "https://api.groq.com/openai"
        run = run_url(groq_url, model_id=model_id, model_name=name, provider_id=provider_id, concurrency=args.concurrency, dataset=args.dataset, api_key=api_key, model=args.groq_model)
        save_run(provider_id, model_id, run)
        _print_summary(run)
        return

    if args.url:
        if not args.name:
            p.error("--name is required with --url")
        model_id = args.model_id or args.name.lower().replace(" ", "-")
        # Auto-detect provider from URL if not specified
        if args.provider_id:
            provider_id = args.provider_id
        elif "api.openai.com" in args.url:
            provider_id = "openai"
        elif "api.ollama.ai" in args.url or "ollama.com" in args.url:
            provider_id = "ollama-cloud"
        elif "localhost" in args.url or "127.0.0.1" in args.url:
            provider_id = "local-ollama"
        else:
            provider_id = "remote"

        # Get API key from --api-key or environment variable
        api_key = args.api_key
        if not api_key and provider_id == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                p.error("--api-key or OPENAI_API_KEY environment variable is required for OpenAI URLs")
        elif not api_key and provider_id == "ollama-cloud":
            api_key = os.getenv("OLLAMA_API_KEY")
            if not api_key:
                p.error("--api-key or OLLAMA_API_KEY environment variable is required for Ollama Cloud URLs")

        run = run_url(args.url, model_id=model_id, model_name=args.name, provider_id=provider_id, concurrency=args.concurrency, dataset=args.dataset, api_key=api_key, model=args.api_model)
        save_run(provider_id, model_id, run)
        _print_summary(run)
        return

    if args.mlx_model:
        if not args.name:
            p.error("--name is required with --mlx-model")
        model_path = Path(args.mlx_model)
        model_id = args.model_id or args.name.lower().replace(" ", "-")
        provider_id = args.provider_id or "local-ollama"
        port = args.port or 9600
        run = run_mlx(model_path, model_id=model_id, model_name=args.name, provider_id=provider_id, port=port, concurrency=args.concurrency, dataset=args.dataset)
        save_run(provider_id, model_id, run)
        _print_summary(run)
        return

    # No individual model argument - use --run-all instead
    if args.model:
        p.error("--model is no longer supported. Use --url with local Ollama URL (http://localhost:11434) and --api-model to specify the model, or use --run-all to run all configured models from benchmark_models.json")

    print("No benchmark specified. Use --run-all, --url, --groq-model, --gemini-model, or --mlx-model")
