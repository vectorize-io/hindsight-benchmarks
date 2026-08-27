"""Single source of truth for benchmark models.

All runners (retain/extraction, quality, reflect) read the model list from
``benchmark_models.json`` via :func:`load_models`. Adding a model is therefore
just appending one JSON entry (see :func:`add_model`) — no code edits, no
duplicated MODELS lists.
"""
import json
import os
from pathlib import Path

# benchmark_models.json lives at the benchmark-runner root (../../.. from this file)
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "benchmark_models.json"

ALL_BENCHMARKS = ("retain", "quality", "reflect")

# Per-provider defaults used when adding a new model. The "method"/"model_field"
# pair tells us where the API model name is stored and how the runner calls it.
PROVIDER_DEFAULTS = {
    "openai": {
        "provider_name": "OpenAI", "provider_icon": "/openai.png",
        "pricing_type": "pay-per-use", "method": "url", "model_field": "api_model",
        "url": "https://api.openai.com", "api_key_env": "OPENAI_API_KEY",
    },
    "gemini": {
        "provider_name": "Google", "provider_icon": "/gemini.png",
        "pricing_type": "pay-per-use", "method": "gemini", "model_field": "gemini_model",
        "url": None, "api_key_env": "GEMINI_API_KEY",
    },
    "groq": {
        "provider_name": "Groq", "provider_icon": "/groq.png",
        "pricing_type": "pay-per-use", "method": "groq", "model_field": "groq_model",
        "url": None, "api_key_env": "GROQ_API_KEY",
    },
    "ollama-cloud": {
        "provider_name": "Ollama Cloud", "provider_icon": "/ollama.png",
        "pricing_type": "pay-per-use", "method": "url", "model_field": "api_model",
        "url": "https://ollama.com", "api_key_env": "OLLAMA_API_KEY",
    },
    "openrouter": {
        # OpenAI-compatible gateway. base_url resolves to https://openrouter.ai/api/v1.
        "provider_name": "OpenRouter", "provider_icon": "/openrouter.png",
        "pricing_type": "pay-per-use", "method": "url", "model_field": "api_model",
        "url": "https://openrouter.ai/api", "api_key_env": "OPENROUTER_API_KEY",
    },
    "local-ollama": {
        "provider_name": "Ollama (Local)", "provider_icon": "/ollama.png",
        "pricing_type": "local", "method": "url", "model_field": "api_model",
        "url": "http://localhost:11434", "api_key_env": None,
    },
    "gcp": {
        # Self-hosted on Vertex AI via run_gcp_model.py; the result file's
        # `deployment` block records the exact hardware. The runners never
        # call these entries directly — the url is the run-time token proxy.
        "provider_name": "GCP (self-hosted)", "provider_icon": "/gcp.png",
        "pricing_type": "local", "method": "url", "model_field": "api_model",
        "url": "http://127.0.0.1:8811", "api_key_env": None,
    },
}


def _normalize(m: dict) -> dict:
    """Map a raw JSON entry to the fields the runners need.

    Derives the Hindsight LLM provider/model/base_url from ``method`` so the
    runners never have to special-case providers.
    """
    method = m["method"]
    if method == "gemini":
        hindsight_provider, hindsight_model, base_url = "gemini", m["gemini_model"], None
    elif method == "groq":
        hindsight_provider, hindsight_model, base_url = "groq", m["groq_model"], None
    elif method == "url":
        # OpenAI-compatible endpoint. Only non-OpenAI hosts need an explicit base_url.
        hindsight_provider = "openai"
        hindsight_model = m["api_model"]
        url = (m.get("url") or "").rstrip("/")
        base_url = None if m["provider_id"] == "openai" else f"{url}/v1"
    else:
        raise ValueError(f"Unknown method {method!r} for model {m.get('model_id')}")

    api_key_env = m.get("api_key_env")
    return {
        "provider_id": m["provider_id"],
        "model_id": m["model_id"],
        "model_name": m.get("model_name", m["model_id"]),
        "hindsight_provider": hindsight_provider,
        "hindsight_model": hindsight_model,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "api_key": os.environ.get(api_key_env, "") if api_key_env else "",
        "benchmarks": m.get("benchmarks", list(ALL_BENCHMARKS)),
        # "none" disables thinking on reasoning models; unset sends no
        # reasoning parameter at all (each model runs at its own default).
        "reasoning_effort": m.get("reasoning_effort"),
    }


def select_ids(all_ids, spec: str) -> set:
    """Resolve a comma-separated filter spec to a set of matching model_ids.

    A token matches by EXACT id first; it only falls back to substring matching
    when the token is not itself an exact model_id of some model. This keeps the
    convenience of partial filters (e.g. "gemini-3.") while preventing a precise
    id like "gpt-oss-20b" from also matching another provider's
    "openai-gpt-oss-20b".
    """
    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    idset = set(all_ids)
    out = set()
    for mid in all_ids:
        for t in tokens:
            if t == mid or (t not in idset and t in mid):
                out.add(mid)
                break
    return out


def load_models(benchmark: str | None = None) -> list[dict]:
    """Return normalized model dicts, optionally filtered to one benchmark.

    ``benchmark`` is one of "retain"/"quality"/"reflect". When given, only
    models whose ``benchmarks`` array includes it are returned.
    """
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    models = [_normalize(m) for m in config["models"]]
    if benchmark:
        models = [m for m in models if benchmark in m["benchmarks"]]
    return models


def add_model(
    provider: str,
    model: str,
    *,
    name: str | None = None,
    model_id: str | None = None,
    input_price: float | None = None,
    output_price: float | None = None,
    size_gb: float | None = None,
    url: str | None = None,
    notes: str | None = None,
    benchmarks: list[str] | None = None,
) -> dict:
    """Append a new model entry to benchmark_models.json and return it.

    ``model`` is the provider's API model name (e.g. "gemini-3.5-flash",
    "openai/gpt-oss-20b", "gemma4:31b"). ``model_id`` (the result-file slug)
    defaults to a sanitized form of ``model``.
    """
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unknown provider {provider!r}. Choose from {list(PROVIDER_DEFAULTS)}")
    d = PROVIDER_DEFAULTS[provider]

    slug = model_id or model.lower().replace("/", "-").replace(":", "-").replace(" ", "-")
    benchmarks = benchmarks or list(ALL_BENCHMARKS)

    entry: dict = {
        "provider_id": provider,
        "provider_name": d["provider_name"],
        "provider_icon": d["provider_icon"],
        "pricing_type": d["pricing_type"],
        "model_id": slug,
        "model_name": name or model,
        "method": d["method"],
    }
    # The API model name field is provider-specific.
    entry[d["model_field"]] = model
    if d["url"] or url:
        entry["url"] = url or d["url"]
    if d["api_key_env"]:
        entry["api_key_env"] = d["api_key_env"]
    if input_price is not None:
        entry["input_price_per_1m"] = input_price
    if output_price is not None:
        entry["output_price_per_1m"] = output_price
    if size_gb is not None:
        entry["size_gb"] = size_gb
    if notes:
        entry["notes"] = notes
    entry["benchmarks"] = benchmarks

    text = CONFIG_PATH.read_text()
    config = json.loads(text)
    existing = {(m["provider_id"], m["model_id"]) for m in config["models"]}
    if (provider, slug) in existing:
        raise ValueError(f"Model {provider}/{slug} already exists in {CONFIG_PATH.name}")

    # Append textually (not a full json.dump rewrite) to preserve existing
    # formatting — keeps git diffs to just the new entry.
    entry_text = "\n".join("    " + line for line in json.dumps(entry, indent=2).splitlines())
    marker = "\n  ]"  # closing bracket of the "models" array
    idx = text.rfind(marker)
    if idx == -1:
        raise ValueError(f"Could not locate models array in {CONFIG_PATH.name}")
    new_text = text[:idx].rstrip() + ",\n" + entry_text + text[idx:]
    CONFIG_PATH.write_text(new_text)
    return entry
