"""
Unified benchmark runner for Hindsight.

Sub-commands:
  add          Register a model in benchmark_models.json and run its benchmarks
  extraction   Fact-extraction benchmark (tests JSON schema output quality/speed)
  quality      LoComo quality benchmark (tests answer accuracy via Hindsight memory)
  reflect      LoComo reflect benchmark (tests reflect() endpoint accuracy + latency)
  all          Run all benchmarks sequentially

Examples:
  rag-benchmark add gemini gemini-3.5-flash
  rag-benchmark add gemini gemini-3.5-flash --benchmarks retain,quality
  rag-benchmark add groq "openai/gpt-oss-20b" --input-price 0.1 --output-price 0.5
  rag-benchmark add openai gpt-4o-mini --no-run
  rag-benchmark extraction --run-all --dataset locomo_3k_50
  rag-benchmark extraction --url http://localhost:11434 --name "My Model" --dataset simple
  rag-benchmark quality
  rag-benchmark quality --model gpt-4o-mini
  rag-benchmark reflect
  rag-benchmark reflect --model gpt-4.1-nano
  rag-benchmark all --dataset locomo_3k_50
"""

import sys

# Retain (fact-extraction) benchmark always runs against this dataset.
RETAIN_DATASET = "locomo_3k"


def _run_run_all(script_name: str, model_filter: str):
    """Import and run a run_all_*.py orchestrator, filtered to one model_id."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        script_name, pathlib.Path(__file__).parent.parent.parent / f"{script_name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore
    sys.argv = ["benchmark", model_filter]
    spec.loader.exec_module(mod)  # type: ignore
    mod.main()


def _add_and_run():
    """`rag-benchmark add <provider> <model>` — register a model and benchmark it."""
    import argparse
    from .models_config import add_model, PROVIDER_DEFAULTS, ALL_BENCHMARKS

    p = argparse.ArgumentParser(prog="rag-benchmark add",
                                description="Add a model to benchmark_models.json and run its benchmarks")
    p.add_argument("provider", choices=list(PROVIDER_DEFAULTS),
                   help="Provider: openai | gemini | groq | ollama-cloud | local-ollama")
    p.add_argument("model", help="Provider API model name, e.g. 'gemini-3.5-flash', 'openai/gpt-oss-20b'")
    p.add_argument("--name", help="Display name (default: the model name)")
    p.add_argument("--model-id", help="Result-file slug (default: sanitized model name)")
    p.add_argument("--input-price", type=float, help="Input price per 1M tokens")
    p.add_argument("--output-price", type=float, help="Output price per 1M tokens")
    p.add_argument("--size-gb", type=float, help="Model size in GB (local models)")
    p.add_argument("--url", help="Custom base URL (for --provider local-ollama or a custom host)")
    p.add_argument("--notes", help="Caveat shown under the model name on the leaderboard")
    p.add_argument("--benchmarks", default="retain,quality,reflect",
                   help="Comma-separated benchmarks to enable+run (default: all three)")
    p.add_argument("--no-run", action="store_true",
                   help="Only add to config; do not run any benchmark")
    args = p.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    bad = [b for b in benchmarks if b not in ALL_BENCHMARKS]
    if bad:
        p.error(f"Unknown benchmark(s) {bad}. Choose from {list(ALL_BENCHMARKS)}")

    entry = add_model(
        args.provider, args.model, name=args.name, model_id=args.model_id,
        input_price=args.input_price, output_price=args.output_price,
        size_gb=args.size_gb, url=args.url, notes=args.notes, benchmarks=benchmarks,
    )
    slug = entry["model_id"]
    print(f"✓ Added {entry['provider_id']}/{slug} to benchmark_models.json "
          f"(benchmarks: {', '.join(benchmarks)})")

    if args.no_run:
        print("--no-run set; skipping benchmark execution.")
        return

    # retain → fact-extraction CLI (uniform across providers via --run-all --only)
    if "retain" in benchmarks:
        print(f"\n=== Running retain (extraction) for {slug} ===")
        from .cli import main as extraction_main
        sys.argv = ["benchmark", "--run-all", "--only", slug,
                    "--dataset", RETAIN_DATASET, "--concurrency", "4"]
        extraction_main()

    if "quality" in benchmarks:
        print(f"\n=== Running quality for {slug} ===")
        _run_run_all("run_all_quality", slug)

    if "reflect" in benchmarks:
        print(f"\n=== Running reflect for {slug} ===")
        _run_run_all("run_all_reflect", slug)

    print(f"\n✓ Done. Results in results/leaderboard/llm/{entry['provider_id']}-{slug}.json")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    sub = sys.argv[1]
    # Remove the subcommand so downstream parsers see a clean argv
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if sub == "add":
        _add_and_run()
        return

    if sub == "extraction":
        from .cli import main as extraction_main
        extraction_main()

    elif sub == "quality":
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "run_all_quality",
            pathlib.Path(__file__).parent.parent.parent / "run_all_quality.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
        mod.main()

    elif sub == "reflect":
        import importlib.util, pathlib
        spec = importlib.util.spec_from_file_location(
            "run_all_reflect",
            pathlib.Path(__file__).parent.parent.parent / "run_all_reflect.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
        mod.main()

    elif sub == "all":
        import importlib.util, pathlib

        # 1. Extraction first
        from .cli import main as extraction_main
        extraction_main()

        # 2. Quality
        spec = importlib.util.spec_from_file_location(
            "run_all_quality",
            pathlib.Path(__file__).parent.parent.parent / "run_all_quality.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
        mod.main()

        # 3. Reflect
        spec = importlib.util.spec_from_file_location(
            "run_all_reflect",
            pathlib.Path(__file__).parent.parent.parent / "run_all_reflect.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
        mod.main()

    else:
        print(f"Unknown sub-command: {sub!r}")
        print("Available sub-commands: add, extraction, quality, reflect, all")
        sys.exit(1)
