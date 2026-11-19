"""Pull every Agent Memory Benchmark result the Benchmarks page shows, at one commit.

  python pull_amb_benchmarks.py <path to an agent-memory-benchmark clone> [ref]   (default: origin/main)
  python pull_amb_benchmarks.py <clone> worktree   reads the clone's files on disk instead, for runs not pushed yet

Writes visualizer/lib/benchmarks.json: per dataset, every run in outputs/ (Hindsight's and the
other systems AMB ran) with its score per split, and the numbers other systems published
(AMB's external_results.json). Datasets that publish one number over several splits also get
the mean as the ``average`` split, only for a run that has every split. ViDoRe runs carry the
own-language score too (each set scored on questions in its own language, Mixedbread's setting).
Reads straight from git at the commit, like pull_amb_coding.py.
"""

import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from pull_amb_coding import git, read

OUT = Path(__file__).parent.parent / "visualizer" / "lib" / "benchmarks.json"
#: Shown elsewhere (sdebench: the coding leaderboard) or not on the page.
SKIP = {"sdebench"}
# ViDoRe tasks' own language (from Mixedbread's per-query files).
VIDORE_NATIVE = {"hr": "english", "computer_science": "english", "pharmaceuticals": "english", "finance_en": "english",
                 "industrial": "english", "finance_fr": "french", "energy": "french", "physics": "french"}
RUN_FIELDS = ("memory_provider", "mode", "answer_llm", "judge_llm", "description")
#: What a run cost, where AMB records it (SupportBench: the answer's LLM tokens per question, and ingestion).
COST_FIELDS = ("avg_query_usd", "ingest_usd")


def main(repo: str, ref: str = "origin/main") -> None:
    if ref == "worktree":
        commit = "worktree@" + git(repo, "rev-parse", "HEAD").decode().strip()[:7]
        root = Path(repo)

        def load(path: str) -> dict:
            f = root / path if (root / path).exists() else root / f"{path}.gz"
            return json.loads(gzip.decompress(f.read_bytes()) if f.suffix == ".gz" else f.read_bytes())
        files = [str(f.relative_to(root)) for f in (root / "outputs").rglob("*.json*")]
    else:
        commit = git(repo, "rev-parse", ref).decode().strip()

        def load(path: str) -> dict:
            return read(repo, commit, path)
        files = git(repo, "ls-tree", "-r", "--name-only", commit, "outputs/").decode().split()
    catalog = load("catalog.json")["datasets"]
    external = load("external_results.json")
    out = {}
    for key, ds in catalog.items():
        if key in SKIP:
            continue
        runs: dict[str, dict] = defaultdict(lambda: {"splits": {}})
        for f in files:
            parts = f.split("/")  # outputs/<dataset>/<run>/<mode>/<split>.json[.gz]
            if len(parts) != 5 or parts[1] != key or parts[3].startswith("_"):
                continue
            split = parts[4].split(".json")[0]
            r = load(f.removesuffix(".gz"))
            s = {"score": r["accuracy"], "queries": r["total_queries"], "path": f}
            s |= {k: r[k] for k in COST_FIELDS if r.get(k) is not None}  # for the score-against-cost charts
            if key.startswith("vidore"):
                meta = [q["meta"] for q in r["results"]]
                s["own_language"] = statistics.mean(
                    float(q["score"]) for q, m in zip(r["results"], meta) if m["group"] == VIDORE_NATIVE[split])
                if len({m["group"] for m in meta}) == 1:
                    s["score"] = None  # own-language queries only: no all-languages score
            run = runs[parts[2]]
            run.update({k: r.get(k) for k in RUN_FIELDS})
            run["splits"][split] = s
        if not runs:
            continue
        avg = ds.get("average_split")
        for run in runs.values():
            splits = [run["splits"].get(s) for s in ds["splits"]]
            if avg and all(splits):  # the leaderboard number is the mean over every split
                run["splits"][avg] = {
                    "score": None if any(s["score"] is None for s in splits) else statistics.mean(s["score"] for s in splits),
                    **({"own_language": statistics.mean(s["own_language"] for s in splits)} if key.startswith("vidore") else {}),
                }
        out[key] = {
            "description": ds["description"], "score_type": ds.get("score_type", "accuracy"), "note": ds.get("scoring_note"),
            "runs": {name: runs[name] for name in sorted(runs)},
            "published": {
                split: [{k: p.get(k) for k in ("memory", "accuracy", "comment", "source_url", "source_label")} for p in rows]
                for split, rows in external.get(key, {}).items()
            },
        }
    OUT.write_text(json.dumps({"amb_commit": commit, "datasets": out}, indent=2) + "\n")
    print(f"wrote {OUT}: {len(out)} datasets at {commit[:7]}")
    for key, d in out.items():
        for name, r in d["runs"].items():
            print(f"  {key:22} {name:26} " + " ".join(
                f"{s}={(v.get('own_language') if v.get('score') is None else v['score']) or 0:.3f}" for s, v in r["splits"].items()))


if __name__ == "__main__":
    main(*sys.argv[1:])
