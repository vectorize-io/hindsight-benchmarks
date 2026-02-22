"""
Unified benchmark runner for Hindsight.

Sub-commands:
  extraction   Fact-extraction benchmark (tests JSON schema output quality/speed)
  quality      LoComo quality benchmark (tests answer accuracy via Hindsight memory)
  reflect      LoComo reflect benchmark (tests reflect() endpoint accuracy + latency)
  all          Run all benchmarks sequentially

Examples:
  rag-benchmark extraction --run-all --dataset locomo_3k_50
  rag-benchmark extraction --url http://localhost:11434 --name "My Model" --dataset simple
  rag-benchmark quality
  rag-benchmark quality --model gpt-4o-mini
  rag-benchmark reflect
  rag-benchmark reflect --model gpt-4.1-nano
  rag-benchmark all --dataset locomo_3k_50
"""

import sys


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    sub = sys.argv[1]
    # Remove the subcommand so downstream parsers see a clean argv
    sys.argv = [sys.argv[0]] + sys.argv[2:]

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
        print("Available sub-commands: extraction, quality, reflect, all")
        sys.exit(1)
