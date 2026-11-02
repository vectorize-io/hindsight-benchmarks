"""Pull the use-case results from agent-memory-benchmark at one commit: coding agents (SDE-Bench) and
conversations (LoCoMo, with its broken questions removed).

  python pull_amb_coding.py <path to an agent-memory-benchmark clone> [commit]   (default: origin/main)

Writes visualizer/lib/coding-agents.json (one row per agent and memory setup, averaged over its repeats)
and visualizer/lib/conversations.json (one row per Hindsight setup).
Reads the files straight from git at the commit, so the clone's working tree does not matter.
"""

import gzip
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).parent.parent / "visualizer" / "lib" / "coding-agents.json"
CONV_OUT = OUT.with_name("conversations.json")
# LoCoMo runs by name, and the models each run's Hindsight server used (AMB records them only in prose).
CONV_RUNS = {"hindsight-gemma4": ["Hindsight", "gemma-4-26b-a4b"],
             "hindsight-gemma4-hs-decide": ["Hindsight", "gemma-4-26b-a4b", "hs-decide-1-flash"]}
# Arms by run name with the agent and repeat cut off (hindsight-gemma-decide-codex-2 -> hindsight-gemma-decide).
# Other run names (full61-*) are experiments and are skipped.
SETUPS = {"vanilla": "No memory", "hindsight": "Hindsight coding-agent plugin",
          "hindsight-gemma": "Hindsight coding-agent plugin", "hindsight-gemma-decide": "Hindsight coding-agent plugin"}
# What the memory arm runs on: Hindsight, then the models its server used (LLM, then the decision model
# when one is on). Read from the run description where AMB records it, set here otherwise.
MEMORY_MODELS = {"vanilla": [], "hindsight": ["Hindsight", "gemini-2.5-flash-lite"],
                 "hindsight-gemma": ["Hindsight", "gemma-4-26b-a4b"],
                 "hindsight-gemma-decide": ["Hindsight", "gemma-4-26b-a4b", "hs-decide-1-flash"]}


def git(repo: str, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", repo, *args], stderr=subprocess.DEVNULL)


def read(repo: str, commit: str, path: str) -> dict:
    try:
        return json.loads(git(repo, "show", f"{commit}:{path}"))
    except subprocess.CalledProcessError:
        return json.loads(gzip.decompress(git(repo, "show", f"{commit}:{path}.gz")))


def main(repo: str, ref: str = "origin/main") -> None:
    commit = git(repo, "rev-parse", ref).decode().strip()
    runs = defaultdict(list)
    for d in git(repo, "ls-tree", "--name-only", commit, "outputs/sdebench/").decode().split():
        arm = re.sub(r"-(claude|codex|opencode)-\d+$", "", d.rsplit("/", 1)[-1])
        if arm in SETUPS:
            r = read(repo, commit, f"{d}/coding/boltons.json")
            runs[(r["answer_llm"], arm)].append([x["meta"] for x in r["results"]])
    rows = []
    for (llm, memory), reps in sorted(runs.items()):
        def mean(f, reps=reps):
            return statistics.mean(statistics.mean(f(t) for t in rep) for rep in reps)
        agent, model = llm.split(":", 1)
        rows.append({
            "agent": agent, "model": model, "memory": memory, "setup": SETUPS[memory], "memory_models": MEMORY_MODELS[memory],
            "repeats": len(reps), "tasks": len(reps[0]),
            "solved": round(100 * mean(lambda t: t["solved"]), 1),
            # interventions counts corrections: 0 means the first attempt passed the tests
            "first_try": round(100 * mean(lambda t: t["interventions"] == 0), 1),
            "corrections": round(mean(lambda t: t["interventions"]), 2),
            "agent_cost_usd": round(mean(lambda t: t["cost_usd"] or 0), 3),
            # The Hindsight server's own LLM and decision-model bill, when the run measured it (newer runs only).
            "memory_cost_usd": round(mean(lambda t: t["memory_cost_usd"]), 3)
            if all(t.get("memory_cost_usd") is not None for rep in reps for t in rep) else None,
            "wall_s": round(mean(lambda t: t["wall_s"]), 1),
        })
        # What a bug costs in all: the agent plus Hindsight's memory, where the run measured it.
        rows[-1]["cost_usd"] = round(rows[-1]["agent_cost_usd"] + (rows[-1]["memory_cost_usd"] or 0), 3)
    OUT.write_text(json.dumps({"amb_commit": commit, "dataset": "sdebench/boltons", "rows": rows}, indent=2) + "\n")
    print(f"wrote {OUT}: {len(rows)} rows at {commit[:7]}")

    conv = []
    for name, models in CONV_RUNS.items():
        r = read(repo, commit, f"outputs/locomo-fixed/{name}/agent/locomo10.json")
        by_type = defaultdict(list)
        for q in r["results"]:
            by_type[q["meta"]["category"]].append(q["correct"])
        conv.append({
            "run": name, "memory_models": models, "questions": r["total_queries"],
            "accuracy": round(100 * r["accuracy"], 1),
            "by_type": {t: round(100 * statistics.mean(v), 1) for t, v in sorted(by_type.items())},
            "answer_s": round(r["avg_retrieve_time_ms"] / 1000, 1), "judge": r["judge_llm"].split(":", 1)[-1],
        })
    CONV_OUT.write_text(json.dumps({"amb_commit": commit, "dataset": "locomo-fixed/locomo10", "rows": conv}, indent=2) + "\n")
    print(f"wrote {CONV_OUT}: {len(conv)} rows")


if __name__ == "__main__":
    main(*sys.argv[1:])
