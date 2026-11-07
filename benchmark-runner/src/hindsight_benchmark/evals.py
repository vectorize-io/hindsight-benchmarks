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
import time
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
               extra_env: dict[str, str] | None = None, cases: str | None = None) -> None:
    # A subscription provider (codex) logs in on its own and has no key.
    key = os.environ.get(provider["api_key_env"]) if "api_key_env" in provider else "unused"
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
        HS_BENCH_TRACES_DIR=str(tokens.parent / "traces"),
        PYTHONPATH=str(SRC_DIR),
        **(extra_env or {}),
    )
    evals = hindsight / "hindsight-system-evals"
    cmd = ["uv", "run", "--project", str(evals), "pytest", *files,
           "-p", "hindsight_benchmark.evals_plugin", "--junitxml", str(junit),
           # *_benchmark_* tests measure a rate or a cost; they are not graded cases.
           "-k", f"not _benchmark_ and ({cases})" if cases else "not _benchmark_"]
    if cfg.get("full"):
        cmd.append("--full")
    # A non-zero exit means some evals failed. That is the score, not an error.
    subprocess.run(cmd, cwd=evals, env=env)


QUOTA_ERROR = "ProviderRateLimitResetError"


def quota_hits(junit: Path) -> tuple[datetime.datetime | None, list[str]] | None:
    """The cases in a junit file that hit a subscription quota, and when the provider said it resets."""
    if not junit.exists():
        return None
    names, reset = [], None
    tree = ET.parse(junit)
    # A quota hit inside the server's background work fails the case as a wrong answer, so look in its
    # trace too: the calls the plugin saved during this pass (a later pass overwrites them).
    suite = next(tree.iter("testsuite"))
    start = datetime.datetime.fromisoformat(suite.get("timestamp")).timestamp() if suite.get("timestamp") else None
    end = start + float(suite.get("time") or 0) + 600 if start else None
    for case in tree.iter("testcase"):
        text = "".join((c.get("message") or "") + (c.text or "") for c in case)
        trace = junit.parent / "traces" / f"{re.sub(r'[^\w.-]+', '_', case.get('name'))}.json"
        if start and trace.exists() and start <= trace.stat().st_mtime <= end:
            text += trace.read_text()
        if QUOTA_ERROR in text:
            names.append(case.get("name"))
            if m := re.search(r"retry at (\d{4}-\d\d-\d\dT[\d:]+(?:[+-]\d\d:\d\d)?)", text):
                reset = max(reset, datetime.datetime.fromisoformat(m[1])) if reset else datetime.datetime.fromisoformat(m[1])
    return (reset, names) if names else None


def splice(junit: Path, tokens: Path, retry_junit: Path, retry_tokens: Path, names: set[str]) -> None:
    """Put the rerun cases' outcomes and token counts in place of the ones that hit the quota."""
    tree = ET.parse(junit)
    retried = {c.get("name"): c for c in ET.parse(retry_junit).iter("testcase") if c.get("name") in names}
    for suite in tree.iter("testsuite"):
        for i, case in enumerate(list(suite)):
            if case.tag == "testcase" and case.get("name") in retried:
                suite.remove(case)
                suite.insert(i, retried[case.get("name")])
    tree.write(junit, encoding="utf-8", xml_declaration=True)
    rows = [r for r in tokens.read_text().splitlines() if json.loads(r)["nodeid"].rsplit("::", 1)[-1] not in retried]
    rows += [r for r in (retry_tokens.read_text().splitlines() if retry_tokens.exists() else [])
             if json.loads(r)["nodeid"].rsplit("::", 1)[-1] in retried]
    tokens.write_text("".join(r + "\n" for r in rows))
    retry_junit.unlink()
    retry_tokens.unlink(missing_ok=True)


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


# Only these reach the public site, under these names. A result for any other model stays local.
# "params" is the size from the model card: total and, for mixture-of-experts models, active per token.
SITE_MODELS = {
    "gemma-4-26b-a4b": {"model": "gemma-4-26b-a4b", "provider": "Google", "icon": "/gemini.png", "openWeights": True, "params": "25.2B total, 3.8B active"},
    "gemini-3.1-flash-lite": {"model": "gemini-3.1-flash-lite", "provider": "Google", "icon": "/gemini.png"},
    "gpt-6.1-terra": {"model": "gpt-6.1-terra", "provider": "OpenAI", "icon": "/openai.png"},
    "gpt-6-luna": {"model": "gpt-6-luna", "provider": "OpenAI", "icon": "/openai.png"},
    "gpt-5.6-terra": {"model": "gpt-5.6-terra", "provider": "OpenAI", "icon": "/openai.png"},
    "gpt-5.6-luna": {"model": "gpt-5.6-luna", "provider": "OpenAI", "icon": "/openai.png"},
    # The same models through OpenRouter, while the Codex subscription is out of usage.
    "openai/gpt-6.1-terra": {"model": "gpt-6.1-terra", "provider": "OpenAI", "icon": "/openai.png"},
    "openai/gpt-6-luna": {"model": "gpt-6-luna", "provider": "OpenAI", "icon": "/openai.png"},
    "openai/gpt-5.6-terra": {"model": "gpt-5.6-terra", "provider": "OpenAI", "icon": "/openai.png"},
    "openai/gpt-5.6-luna": {"model": "gpt-5.6-luna", "provider": "OpenAI", "icon": "/openai.png"},
    "gemini-3.8-flash": {"model": "gemini-3.8-flash", "provider": "Google", "icon": "/gemini.png"},
    "anthropic/claude-haiku-5.5": {"model": "claude-haiku-5.5", "provider": "Anthropic", "icon": "/logos/anthropic.webp"},
    "claude-haiku-4.5": {"model": "claude-haiku-4.5", "provider": "Anthropic", "icon": "/logos/anthropic.webp"},
    "deepseek/deepseek-v4.1-flash": {"model": "deepseek-v4.1-flash", "provider": "DeepSeek", "icon": "/logos/deepseek.webp", "openWeights": True, "params": "552B total, 8B active on input, 16B on output"},
    "qwen/qwen3.8-flash": {"model": "qwen3.8-flash", "provider": "Qwen", "icon": "/logos/qwen.webp", "openWeights": True, "params": "125B total, 6B active"},
    "z-ai/glm-5.3-flash": {"model": "glm-5.3-flash", "provider": "Z.ai", "icon": "/logos/zai.webp", "openWeights": True, "params": "320B total, 18B active"},
    "minimax/minimax-m3.1-flash": {"model": "minimax-m3.1-flash", "provider": "MiniMax", "icon": "/logos/minimax.webp", "openWeights": True, "params": "428B total, 23B active"},
    "nvidia/nemotron-3.5-super-vl": {"model": "nemotron-3.5-super-vl", "provider": "NVIDIA", "icon": "/logos/nvidia.webp", "openWeights": True, "params": "120B total, 12B active"},
    "qwen/qwen3.8-27b": {"model": "qwen3.8-27b", "provider": "Qwen", "icon": "/logos/qwen.webp", "openWeights": True, "params": "27B"},
    "qwen/qwen3.6-35b-a3b": {"model": "qwen3.6-35b-a3b", "provider": "Qwen", "icon": "/logos/qwen.webp", "openWeights": True, "params": "35B total, 3B active"},
}
# Decision models the site shows. "onLlmBoards" shows the run as a row on the LLM boards.
SITE_DECISION_MODELS: dict[str, dict] = {
    "hs-decide-1-flash": {"model": "hs-decide-1-flash", "provider": "Hindsight", "onLlmBoards": True, "params": "0.8B"},
}
# More, as the same JSON, for a local preview of ones not public yet.
SITE_DECISION_MODELS_ENV = "HS_BENCH_SITE_DECISION_MODELS"
# Decision models priced off any list: $ per 1M input tokens, output free. More through the env, as the same JSON.
DECISION_PRICES = {"hs-decide-1-flash": 0.02}
DECISION_PRICES_ENV = "HS_BENCH_DECISION_PRICES"


def decision_prices() -> dict[str, float]:
    return {**DECISION_PRICES, **json.loads(os.environ.get(DECISION_PRICES_ENV, "{}"))}


def case_results(run_dir: Path, op: str, by_stem: dict[str, str]) -> dict[str, dict]:
    """Category -> passed, total and failed case ids for one operation, over every pass in a run.

    Every pass counts, as in the operation's own passed/total, so the categories add up to the same score.
    """
    out: dict[str, dict] = {}
    cases = [c for j in sorted(run_dir.glob("junit-*.xml")) for c in ET.parse(j).iter("testcase")]
    for case in cases:
        m = re.search(r"\[([^/\]]+)/(.+)\]$", case.get("name", ""))
        if not m or case.find("skipped") is not None or _operation(case.get("classname", ""), by_stem) != op:
            continue
        c = out.setdefault(m[1], {"passed": 0, "total": 0, "failed": []})
        c["total"] += 1
        if case.find("failure") is None and case.find("error") is None:
            c["passed"] += 1
        elif m[2] not in c["failed"]:
            c["failed"].append(m[2])
    return out


def export_site(out: Path, operations: dict[str, list[str]], commit: str) -> None:
    """Write the public site's results: SITE_MODELS rows only, without decision model or with a public one.

    Only runs at the pinned commit: rows from different commits ran different cases.
    """
    by_stem = {Path(f).stem: op for op, files in operations.items() for f in files}
    deciders = {**SITE_DECISION_MODELS, **json.loads(os.environ.get(SITE_DECISION_MODELS_ENV, "{}"))}
    providers = json.loads(os.path.expandvars(CONFIG.read_text()))["providers"]
    prices = decision_prices()
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("hindsight_commit") != commit:
            continue
        decision = (r.get("decision_model") or ":").split(":", 1)[1]
        if r["model"] not in SITE_MODELS or (decision and decision not in deciders):
            continue
        ops = {}
        for op, o in r["operations"].items():
            if op == "recall":
                continue  # the site has no recall board
            t = o["tokens"]
            ops[op] = {
                "passed": o["passed"], "total": o["total"], "cost_usd": o["cost_usd"],
                # Output as billed: the visible answer plus the reasoning.
                "calls": t["calls"], "input_tokens": t["input_tokens"],
                "output_tokens": t["output_tokens"] + t.get("thoughts_tokens", 0),
                "decision_calls": t.get("decision_calls", 0),
                "decision_input_tokens": t.get("decision_input_tokens", 0),
                # What the decision model cost: the total less the LLM's share.
                "decision_cost_usd": round(o["cost_usd"] - (cost_usd(t, r["pricing_per_token"]) or 0), 4)
                if o["cost_usd"] is not None else None,
                "categories": case_results(RUNNER_DIR / o["run_dir"], op, by_stem),
            }
        # Where the price came from, for the site to show: the OpenRouter list entry every route is priced at.
        prov = providers.get(r["provider"], {})
        or_id = prov.get("openrouter_models", {}).get(r["model"]) or prov.get("openrouter_prefix", "") + r["model"]
        per_m = lambda v: round(v * 1e6, 4)  # noqa: E731
        source = {
            "openrouter_id": or_id,
            "price_per_1m": {k: per_m(v) for k, v in r["pricing_per_token"].items()},
            "decision_price_per_1m_input": prices.get(decision) if decision else None,
            "params": SITE_MODELS[r["model"]].get("params"),
            "decision_params": deciders[decision].get("params") if decision else None,
            "ran": f"{r['timestamp'][:4]}-{r['timestamp'][4:6]}-{r['timestamp'][6:8]}",
            "judge": r["judge"],
            "repeats": r["repeats"],
        }
        rows.append({**SITE_MODELS[r["model"]], "decision": deciders.get(decision), "source": source, "operations": ops})
    out.write_text(json.dumps({"rows": rows}, indent=1) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


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
            for op, cats in collected.items() if op != "recall"
        },
    }
    out.write_text(json.dumps(structure, indent=2) + "\n")
    for op, o in structure["operations"].items():
        print(f"  {op:14} {sum(len(c['cases']) for c in o['categories']):3} cases in {len(o['categories'])} categories")
    print(f"wrote {out}")


def list_pricing(provider: dict, model: str) -> dict:
    """$ per token from the provider's own /models list, or OpenRouter's for a provider without one."""
    # Direct and subscription routes show OpenRouter's list price, so models compare fairly.
    if or_id := provider.get("openrouter_models", {}).get(model):
        # A model served on our own server: OpenRouter's list price for it, like every other model.
        provider, model = {"base_url": "https://openrouter.ai/api/v1"}, or_id
    elif prefix := provider.get("openrouter_prefix"):
        provider, model = {"base_url": "https://openrouter.ai/api/v1"}, prefix + model
    if not provider["base_url"]:
        return {}
    req = urllib.request.Request(f"{provider['base_url']}/models",
                                 headers={"Authorization": f"Bearer {os.environ.get(provider.get('api_key_env', ''), '')}"})
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
                 + (tokens["output_tokens"] + tokens.get("thoughts_tokens", 0)) * pricing["output"], 4)


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
                   help="turn on a decision model, e.g. internal:hs-decide-1-flash (default: none)")
    p.add_argument("--export-site", metavar="PATH",
                   help="write the public site's results from results/leaderboard/evals, then exit")
    p.add_argument("--cases", metavar="EXPR",
                   help="pytest -k expression to rerun some cases, e.g. 'hq-owner-latest or du-launch-risk'; "
                        "prints the outcome and writes no result file")
    p.add_argument("--resume", metavar="RUN_DIR",
                   help="finish a run that was cut short: reuse its run dir and skip the passes that wrote their junit file")
    p.add_argument("--embedder", metavar="PROVIDER:MODEL",
                   help="swap the embedder, e.g. internal:hs-embed-1 (default: Hindsight's own, bge-small-en-v1.5)")
    p.add_argument("--reranker", metavar="PROVIDER:MODEL",
                   help="swap the reranker, e.g. internal:hs-rerank-1 (default: Hindsight's own, ms-marco-MiniLM-L-6-v2)")
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
    if args.export_site:
        export_site(Path(args.export_site), operations, cfg["hindsight_commit"])
        return
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
    # Server settings one model needs to be served at all, from evals.json `model_settings`.
    extra_env |= {f"HINDSIGHT_EVAL_SET_{k}": v for k, v in cfg.get("model_settings", {}).get(args.model, {}).items()}
    # Recall's swaps: the embedder over the provider's OpenAI-style /embeddings, the reranker over its Cohere-style /rerank.
    if args.embedder:
        e_name, e_model = args.embedder.split(":", 1)
        e = cfg["providers"][e_name]
        extra_env |= {
            "HINDSIGHT_EVAL_SET_EMBEDDINGS_PROVIDER": "openai",
            "HINDSIGHT_EVAL_SET_EMBEDDINGS_OPENAI_BASE_URL": e["base_url"],
            "HINDSIGHT_EVAL_SET_EMBEDDINGS_OPENAI_MODEL": e_model,
            "HINDSIGHT_EVAL_SET_EMBEDDINGS_OPENAI_API_KEY": os.environ.get(e["api_key_env"], ""),
        }
    if args.reranker:
        r_name, r_model = args.reranker.split(":", 1)
        r = cfg["providers"][r_name]
        extra_env |= {
            "HINDSIGHT_EVAL_SET_RERANKER_PROVIDER": "cohere",
            "HINDSIGHT_EVAL_SET_RERANKER_COHERE_BASE_URL": r["rerank_url"],
            "HINDSIGHT_EVAL_SET_RERANKER_COHERE_MODEL": r_model,
            "HINDSIGHT_EVAL_SET_RERANKER_COHERE_API_KEY": os.environ.get(r["api_key_env"], ""),
        }
    slug = (f"{args.provider}-{args.model}" + (f"+{args.decision}" if args.decision else "")
            + (f"+embed-{args.embedder}" if args.embedder else "") + (f"+rerank-{args.reranker}" if args.reranker else ""))
    slug = slug.replace("/", "-").replace(":", "-")
    if args.resume:
        run_dir = Path(args.resume).resolve()
        stamp = run_dir.name.rsplit("-", 1)[1]
    else:
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir = WORK_DIR / "runs" / f"{slug}-{stamp}"
        run_dir.mkdir(parents=True)
    # Same install as Hindsight's perf-test workflow: the eval server embeds and reranks locally.
    # --inexact keeps packages added by hand (torchvision, which the recall boards' Nemotron embedder needs).
    subprocess.run(["uv", "sync", "--frozen", "--inexact", "--all-extras", "--index-strategy", "unsafe-best-match"],
                   cwd=hindsight / "hindsight-api-slim", check=True)
    files = [f for fs in operations.values() for f in fs]
    junits, token_files = [], []
    for i in range(repeats):
        junits.append(run_dir / f"junit-{i}.xml")
        token_files.append(run_dir / f"tokens-{i}.jsonl")
        if not (args.resume and junits[-1].exists()):
            # The plugin appends token counts, so a cut-short pass's partial file would count twice.
            token_files[-1].unlink(missing_ok=True)
            run_pytest(hindsight, files, args.model, provider, cfg, junits[-1], token_files[-1], extra_env, args.cases)
        # A subscription quota hit mid-pass scored the limit, not the model. A pass can use up a whole quota
        # window, so rerunning it would hit the limit again: after the reset, rerun only the cases that hit it.
        while hits := quota_hits(junits[-1]):
            wait = max(60.0, (hits[0] - datetime.datetime.now(datetime.UTC)).total_seconds() + 60) if hits[0] else 1800.0
            names = hits[1]
            print(f"pass {i + 1}: {len(names)} cases hit a quota limit; rerunning them in {wait / 60:.0f} min",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
            retry_junit, retry_tokens = junits[-1].with_suffix(".retry.xml"), token_files[-1].with_suffix(".retry.jsonl")
            retry_tokens.unlink(missing_ok=True)
            ids = " or ".join(re.sub(r"^.*\[(?:.*/)?|\]$", "", n) for n in names)
            run_pytest(hindsight, files, args.model, provider, cfg, retry_junit, retry_tokens, extra_env, ids)
            if not retry_junit.exists():
                continue  # the rerun died before writing its results: wait and try again
            splice(junits[-1], token_files[-1], retry_junit, retry_tokens, set(names))

    ops = tally(junits, token_files, operations)
    if args.cases:
        # A few cases are a look at those cases, not a score: keep them out of the results.
        for op, o in ops.items():
            if o["total"]:
                print(f"  {op:14} {o['passed']}/{o['total']}")
        print(f"run and traces in {run_dir}")
        return
    pricing = list_pricing(provider, args.model)
    d_pricing = list_pricing(decision, d_model) if decision else None
    # A decision model priced off any list: DECISION_PRICES.
    if decision and (price := decision_prices().get(d_model)) is not None:
        d_pricing = {"input": price / 1e6, "cached_input": price / 1e6, "output": 0.0}
    if decision and not d_pricing:
        print(f"warning: no list price for decision model {d_model}; its calls are left out of the cost", file=sys.stderr)
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
        "embedder": args.embedder,
        "reranker": args.reranker,
        "hindsight_commit": commit,
        # A GPT judge runs through codex (evals_plugin.py); any other is the evals' own Gemini judge.
        "judge": f"{'codex' if cfg['judge_model'].startswith('gpt-') else 'gemini'}/{cfg['judge_model']}",
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
