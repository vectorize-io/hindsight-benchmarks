"""Fill the Hindsight Models pages from our models' Hugging Face cards: no numbers of our own.

  python pull_hf_cards.py      (reads the token from ~/.cache/huggingface/token, or HF_TOKEN)

Writes visualizer/lib/models.json. Scores come from each card's `model-index` metadata, the standard
place for evaluation results on Hugging Face. Each `model-index` entry is one table row: the first is
our model, any further entry is a model it is compared with, named by its Hugging Face id
(e.g. `BAAI/bge-small-en-v1.5`). Columns are the results' datasets and metrics, in the card's order.
Until a card has its `model-index`, `hf_cards/<model>.json` stands in: the same metadata, as it will be
published. A model with neither gives a page with its facts and no table.

Most columns are scores, shown 0 to 100. Speed and size columns (metric types in PLAIN) are shown as
measured, and lower wins where LOWER says so.
"""

import json
import os
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent.parent / "visualizer" / "lib" / "models.json"
# The model's facts that are specs, not results; the parameter count is read from its weights.
FAMILIES = [
    {"id": "decision", "title": "Decision model", "repo": "hindsight-ai/hs-decide-1-flash",
     "short": "Fast yes/no, pick-one and score answers with a probability for every option.",
     "facts": [["Parameters", "0.8B"], ["Context", "65,536 tokens"], ["Languages", "8"], ["Serves on", "vLLM"]]},
    {"id": "embeddings", "title": "Embedding model", "repo": "hindsight-ai/hs-embed-1",
     "short": "Turns memories and queries into vectors for recall.",
     "facts": [["Dimensions", "384 (also 256, 128, 64)"], ["Context", "8,192 tokens"], ["Languages", "8"], ["Runs on", "CPU or GPU, stock TEI"]]},
    {"id": "reranker", "title": "Reranker", "repo": "hindsight-ai/hs-rerank-1",
     "short": "Orders what recall finds so the right memory comes first.",
     "facts": [["Context", "8,192 tokens"], ["Languages", "8"], ["Runs on", "CPU or GPU, stock TEI"]]},
]
# Hugging Face org -> the provider name the site shows (and picks a logo by).
ORGS = {"hindsight-ai": "Hindsight", "BAAI": "BAAI", "cross-encoder": "SBERT", "jhu-clsp": "JHU CLSP",
        "sentence-transformers": "SBERT", "nvidia": "NVIDIA", "intfloat": "Microsoft",
        "ibm-granite": "IBM", "Snowflake": "Snowflake", "Alibaba-NLP": "Alibaba", "mixedbread-ai": "Mixedbread",
        "StartLux": "StartLux", "vLLM Semantic Router": "vLLM Semantic Router", "Mapika": "Mapika"}
DRAFTS = Path(__file__).parent / "hf_cards"
PLAIN = {"latency_ms": "ms", "latency_s": "s", "throughput": "/s", "max_tokens": "tokens"}
LOWER = {"latency_ms", "latency_s"}


def token() -> str:
    path = Path.home() / ".cache/huggingface/token"
    return os.environ.get("HF_TOKEN") or (path.read_text().strip() if path.exists() else "")


def api(repo: str) -> dict:
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}", headers={"Authorization": f"Bearer {token()}"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def size(params: int | None) -> str | None:
    if not params:
        return None
    return f"{params / 1e9:.1f}B" if params >= 1e9 else f"{round(params / 1e6)}M"


def family(f: dict, info: dict, sizes=lambda repo: None) -> dict:
    entries = (info.get("cardData") or {}).get("model-index") or []
    draft = DRAFTS / f"{f['repo'].split('/')[1]}.json"
    if not entries and draft.exists():
        entries = json.loads(draft.read_text())["model-index"]
    benchmarks, rows = {}, []
    for i, e in enumerate(entries):
        name = e.get("name") or f["repo"]
        org, _, model = name.rpartition("/")
        ours = i == 0
        scores = {}
        for r in e.get("results", []):
            d = r.get("dataset", {})
            for m in r.get("metrics", []):
                bid = f"{d.get('type') or d.get('name')}:{d.get('config') or ''}:{m['type']}"
                src = r.get("source") or {}
                plain = m["type"] in PLAIN
                benchmarks.setdefault(bid, {"id": bid, "name": d.get("name") or d.get("type"), "metric": PLAIN.get(m["type"], ""),
                                            "href": src.get("url"), "about": m.get("name") or src.get("name") or "",
                                            "section": (r.get("task") or {}).get("name") or "",
                                            **({"plain": True} if plain else {}), **({"lower": True} if m["type"] in LOWER else {})})
                # Cards write a fraction (0.715) or a percent (71.5): the site shows scores 0 to 100.
                v = float(m["value"])
                scores[bid] = v if plain else round(100 * v if v <= 1 else v, 1)
        rows.append({"model": model if not ours else f["repo"].split("/")[1], "provider": "Hindsight" if ours else ORGS.get(org, org or "—"),
                     **({"tag": "hindsight"} if ours else {}),
                     **({"size": sz} if (sz := sizes(name if org else f["repo"])) else {}), "scores": scores})
    params = size((info.get("safetensors") or {}).get("total"))
    return {"id": f["id"], "title": f["title"], "model": f["repo"].split("/")[1], "short": f["short"],
            "card": f"https://huggingface.co/{f['repo']}",
            "facts": ([["Parameters", params]] if params and all(k != "Parameters" for k, _ in f["facts"]) else []) + f["facts"],
            "benchmarks": list(benchmarks.values()), "rows": rows}


def check() -> None:
    """A card with our model and one compared model; and a card with no results yet."""
    card = {"safetensors": {"total": 111728256}, "cardData": {"model-index": [
        {"name": "hindsight-ai/hs-embed-1", "results": [{"dataset": {"name": "NanoBEIR", "type": "nanobeir"}, "metrics": [{"type": "ndcg_at_10", "value": 0.613}]}]},
        {"name": "BAAI/bge-small-en-v1.5", "results": [{"dataset": {"name": "NanoBEIR", "type": "nanobeir"}, "metrics": [{"type": "ndcg_at_10", "value": 62.6}]}]},
    ]}}
    f = family(FAMILIES[1], card)
    assert [b["id"] for b in f["benchmarks"]] == ["nanobeir::ndcg_at_10"] and not f["benchmarks"][0].get("plain")
    assert [(r["model"], r["provider"], r.get("tag"), r["scores"]["nanobeir::ndcg_at_10"]) for r in f["rows"]] == [
        ("hs-embed-1", "Hindsight", "hindsight", 61.3), ("bge-small-en-v1.5", "BAAI", None, 62.6)]
    assert f["facts"][0] == ["Parameters", "112M"]
    empty = family({**FAMILIES[0], "repo": "hindsight-ai/no-draft"}, {"cardData": {}})
    assert empty["rows"] == [] and empty["benchmarks"] == [] and empty["facts"][0] == ["Parameters", "0.8B"]


def main() -> None:
    check()
    def sizes(repo: str) -> str | None:
        try:
            return size((api(repo).get("safetensors") or {}).get("total"))
        except Exception:
            return None
    fams = [family(f, api(f["repo"]), sizes) for f in FAMILIES]
    OUT.write_text(json.dumps({"families": fams}, indent=2) + "\n")
    for f in fams:
        print(f"{f['id']}: {len(f['rows'])} rows, {len(f['benchmarks'])} columns")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
