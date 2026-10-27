"""Run Hindsight's system evals against one model, at a pinned Hindsight commit.

  hindsight-evals --model qwen/qwen3.6-35b-a3b                       # OpenRouter
  hindsight-evals --provider internal --model gemma-4-26b-a4b   # base URL from INTERNAL_BASE_URL
  hindsight-evals --model qwen/qwen3.6-35b-a3b --operations retain,reflect
  hindsight-evals --model ... --hindsight-repo ~/dev/hindsight   # fetch the pinned commit from a local clone

evals.json says which commit, which judge, and which eval files make up each
operation. The commit is checked out under .hindsight/<sha>; the evals start
their own hindsight-api from that checkout, so the code under test is exactly
the pinned commit. The result lands in results/leaderboard/evals/<model>.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

RUNNER_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = RUNNER_DIR / "src"
CONFIG = RUNNER_DIR / "evals.json"
WORK_DIR = RUNNER_DIR / ".hindsight"
RESULTS_DIR = RUNNER_DIR.parent / "results" / "leaderboard" / "evals"


def checkout(repo: str, commit: str) -> Path:
    """The pinned commit in its own directory, fetched once and reused."""
    dest = WORK_DIR / commit
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode == 0 and head.stdout.strip() == commit:
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    for args in (["init", "-q"], ["fetch", "-q", "--depth", "1", repo, commit], ["checkout", "-q", "FETCH_HEAD"]):
        subprocess.run(["git", "-C", str(dest), *args], check=True)
    return dest


def run_pytest(hindsight: Path, files: list[str], model: str, provider: dict, cfg: dict, junit: Path, tokens: Path,
               extra_env: dict[str, str] | None = None) -> None:
    key = os.environ.get(provider["api_key_env"])
    if not key:
        sys.exit(f"{provider['api_key_env']} is not set")
    # Only what we set reaches the evals: a stray HINDSIGHT_EVAL_SET_* would change what is measured.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HINDSIGHT_EVAL_", "HINDSIGHT_API_")) and k != "PYTEST_ADDOPTS"}
    env.update(
        HINDSIGHT_EVAL_LLM_PROVIDER=provider["hindsight_provider"],
        HINDSIGHT_EVAL_LLM_BASE_URL=provider["base_url"],
        HINDSIGHT_EVAL_LLM_MODEL=model,
        HINDSIGHT_EVAL_LLM_API_KEY=key,
        HINDSIGHT_EVAL_JUDGE_MODEL=cfg["judge_model"],
        HS_BENCH_TOKENS_OUT=str(tokens),
        PYTHONPATH=str(SRC_DIR),
        **(extra_env or {}),
    )
    evals = hindsight / "hindsight-system-evals"
    cmd = ["uv", "run", "--project", str(evals), "pytest", *files,
           "-p", "hindsight_benchmark.evals_plugin", "--junitxml", str(junit),
           # *_benchmark_* tests measure a rate or a cost; they are not graded cases.
           "-k", "not _benchmark_"]
    if cfg.get("full"):
        cmd.append("--full")
    # A non-zero exit means some evals failed. That is the score, not an error.
    subprocess.run(cmd, cwd=evals, env=env)


def _operation(name: str, by_stem: dict[str, str]) -> str | None:
    """Which operation a junit classname or pytest nodeid belongs to, by its eval file."""
    parts = name.replace("/", ".").replace("::", ".").split(".")
    return next((by_stem[p] for p in parts if p in by_stem), None)


def tally(junits: list[Path], token_files: list[Path], operations: dict[str, list[str]]) -> dict:
    by_stem = {Path(f).stem: op for op, files in operations.items() for f in files}
    ops = {op: {"passed": 0, "failed": 0, "errors": 0,
                "tokens": {"calls": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}}
           for op in operations}
    for junit in junits:
        if not junit.exists():
            continue
        for case in ET.parse(junit).iter("testcase"):
            op = _operation(case.get("classname", ""), by_stem)
            if op is None or case.find("skipped") is not None:
                continue
            outcome = "errors" if case.find("error") is not None else "failed" if case.find("failure") is not None else "passed"
            ops[op][outcome] += 1
    for path in token_files:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if op := _operation(row.pop("nodeid"), by_stem):
                for k, v in row.items():
                    ops[op]["tokens"][k] = ops[op]["tokens"].get(k, 0) + v
    for o in ops.values():
        o["total"] = o["passed"] + o["failed"] + o["errors"]
        # Errors count against the model: a run that crashes must not score better than one that fails.
        o["score"] = round(100 * o["passed"] / o["total"], 1) if o["total"] else None
    return ops


def parse_collected(lines: list[str]) -> dict[str, dict[str, list[str]]]:
    """Operation -> category -> case ids, from `pytest --collect-only` tree output.

    Case tests are id'd `<category>/<case id>`. Tests without one (`*_benchmark_*`:
    a rate or a cost, not a graded case) are listed under the "benchmarks" key.
    """
    out: dict[str, dict[str, list[str]]] = {}
    op = None
    for line in lines:
        line = line.strip()
        if m := re.match(r"<Module test_(\w+)\.py>", line):
            op = m.group(1)
        elif op and (m := re.match(r"<(?:Coroutine|Function) (\w+)(?:\[([^/\]]+)/([^\]]+)\])?>", line)):
            name, category, case = m.groups()
            if category:
                out.setdefault(op, {}).setdefault(category, []).append(case)
            elif "_benchmark_" in name:
                out.setdefault(op, {}).setdefault("benchmarks", []).append(name)
    return out


def parse_readme_table(readme: str) -> dict[tuple[str, str], str]:
    """(operation, category) -> "what it checks", from the evals README table."""
    rows = re.findall(r"^\| (\w+) \| (\w+) \| \d+ \| (.+?) \|$", readme, flags=re.M)
    return {(op, cat): about for op, cat, about in rows}


def export_structure(hindsight: Path, commit: str, out: Path, operations: dict[str, list[str]]) -> None:
    """Write the UI's eval structure: every operation, category and case at the pinned commit."""
    evals = hindsight / "hindsight-system-evals"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTEST_ADDOPTS", "VIRTUAL_ENV")}
    done = subprocess.run(["uv", "run", "pytest", *[f for fs in operations.values() for f in fs], "--collect-only", "-q", "--full"],
                          cwd=evals, env=env, capture_output=True, text=True, check=True)
    collected = parse_collected(done.stdout.splitlines())
    about = parse_readme_table((evals / "README.md").read_text())
    structure = {
        "hindsight_commit": commit,
        "operations": {
            op: {
                "categories": [
                    {"id": cat, "about": about.get((op, cat), ""), "cases": cases}
                    for cat, cases in cats.items() if cat != "benchmarks"
                ],
                "benchmarks": cats.get("benchmarks", []),
            }
            for op, cats in collected.items()
        },
    }
    out.write_text(json.dumps(structure, indent=2) + "\n")
    for op, o in structure["operations"].items():
        print(f"  {op:14} {sum(len(c['cases']) for c in o['categories']):3} cases in {len(o['categories'])} categories")
    print(f"wrote {out}")


def list_pricing(provider: dict, model: str) -> dict:
    """$ per token from the provider's own /models list."""
    req = urllib.request.Request(f"{provider['base_url']}/models",
                                 headers={"Authorization": f"Bearer {os.environ.get(provider['api_key_env'], '')}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        models = json.load(r)["data"]
    p = next((m.get("pricing") for m in models if m["id"] == model and m.get("pricing")), None)
    if p is None:
        return {}
    return parse_pricing(p)


def parse_pricing(p: dict) -> dict:
    if "input_per_million" in p:  # per-million shape
        inp = float(p["input_per_million"]) / 1e6
        return {"input": inp, "cached_input": inp, "output": float(p.get("output_per_million") or 0) / 1e6}
    # OpenRouter: strings in $ per token
    return {"input": float(p.get("prompt") or 0), "output": float(p.get("completion") or 0),
            "cached_input": float(p.get("input_cache_read") or p.get("prompt") or 0)}


def cost_usd(tokens: dict, pricing: dict) -> float | None:
    if not pricing:
        return None
    uncached = tokens["input_tokens"] - tokens["cached_tokens"]
    return round(uncached * pricing["input"] + tokens["cached_tokens"] * pricing["cached_input"]
                 + tokens["output_tokens"] * pricing["output"], 4)


def summarize(ops: dict) -> dict:
    scores = [o["score"] for o in ops.values() if o["score"] is not None]
    costs = [o["cost_usd"] for o in ops.values() if o["cost_usd"] is not None]
    return {"score": round(sum(scores) / len(scores), 1) if scores else None,
            "cost_usd": round(sum(costs), 4) if costs else None}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="model id at the provider, e.g. qwen/qwen3.6-35b-a3b")
    p.add_argument("--export-structure", metavar="PATH",
                   help="write every operation, category and case at the pinned commit as JSON (for the UI), then exit")
    p.add_argument("--provider", default="openrouter", help="a provider from evals.json (default: openrouter)")
    p.add_argument("--operations", help="comma-separated subset of evals.json operations")
    p.add_argument("--hindsight-repo", help="where to fetch the pinned commit from (default: evals.json)")
    p.add_argument("--repeats", type=int, help="override evals.json repeats")
    p.add_argument("--decision", metavar="PROVIDER:MODEL",
                   help="turn on a decision model, e.g. openrouter:cloudflare/clef (default: none)")
    args = p.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(RUNNER_DIR / ".env")
    except ImportError:
        pass

    # Base URLs may name env vars (${INTERNAL_BASE_URL}), so they stay out of the repo.
    cfg = json.loads(os.path.expandvars(CONFIG.read_text()))
    provider = cfg["providers"][args.provider]
    operations = cfg["operations"]
    if args.operations:
        operations = {op: operations[op] for op in args.operations.split(",")}
    repeats = args.repeats or cfg.get("repeats", 1)
    commit = cfg["hindsight_commit"]
    hindsight = checkout(os.path.expanduser(args.hindsight_repo or cfg["hindsight_repo"]), commit)
    if args.export_structure:
        export_structure(hindsight, commit, Path(args.export_structure), operations)
        return
    if not args.model:
        p.error("--model is required")

    extra_env, decision = {}, None
    if args.decision:
        d_name, d_model = args.decision.split(":", 1)
        decision = cfg["providers"][d_name]
        # Both providers serve System One, the API Hindsight's `typesafe` provider speaks.
        extra_env = {
            "HINDSIGHT_EVAL_SET_DECISION_MODEL_PROVIDER": "typesafe",
            "HINDSIGHT_EVAL_SET_DECISION_MODEL_BASE_URL": decision["decision_base_url"],
            "HINDSIGHT_EVAL_SET_DECISION_MODEL_MODEL": d_model,
            "HINDSIGHT_EVAL_SET_DECISION_MODEL_API_KEY": os.environ.get(decision["api_key_env"], ""),
        }
    slug = f"{args.provider}-{args.model}" + (f"+{args.decision}" if args.decision else "")
    slug = slug.replace("/", "-").replace(":", "-")
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = WORK_DIR / "runs" / f"{slug}-{stamp}"
    run_dir.mkdir(parents=True)
    # Same install as Hindsight's perf-test workflow: the eval server embeds and reranks locally.
    subprocess.run(["uv", "sync", "--frozen", "--all-extras", "--index-strategy", "unsafe-best-match"],
                   cwd=hindsight / "hindsight-api-slim", check=True)
    files = [f for fs in operations.values() for f in fs]
    junits, token_files = [], []
    for i in range(repeats):
        junits.append(run_dir / f"junit-{i}.xml")
        token_files.append(run_dir / f"tokens-{i}.jsonl")
        run_pytest(hindsight, files, args.model, provider, cfg, junits[-1], token_files[-1], extra_env)

    ops = tally(junits, token_files, operations)
    pricing = list_pricing(provider, args.model)
    d_pricing = list_pricing(decision, d_model) if decision else None
    for o in ops.values():
        o["cost_usd"] = cost_usd(o["tokens"], pricing)
        if d_pricing and o["cost_usd"] is not None:
            t = o["tokens"]
            o["cost_usd"] = round(o["cost_usd"] + t.get("decision_input_tokens", 0) * d_pricing["input"], 4)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{slug}.json"
    for o in ops.values():
        o["run_dir"] = str(run_dir.relative_to(RUNNER_DIR))
        o["timestamp"] = stamp
    # A run of some operations updates those and keeps the others, if they ran at the same commit.
    prev = json.loads(out.read_text()) if out.exists() else {}
    if prev.get("hindsight_commit") == commit:
        ops = {**prev.get("operations", {}), **ops}
    result = summarize(ops)
    result = {
        "model": args.model,
        "provider": args.provider,
        "decision_model": args.decision,
        "hindsight_commit": commit,
        "judge": f"codex/{cfg['judge_model']}",
        "full": bool(cfg.get("full")),
        "repeats": repeats,
        "timestamp": stamp,
        "pricing_per_token": pricing,
        **result,
        "operations": ops,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("model", "score", "cost_usd")}))
    for op, o in ops.items():
        print(f"  {op:14} {o['passed']}/{o['total']}  score={o['score']}  cost=${o['cost_usd']}  errors={o['errors']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
