#!/usr/bin/env python3
"""
Self-hosted reranker evaluation — quality across models, modes, and candidate counts.

Companion to run_all_reranker.py, which benchmarks hosted rerankers on CPU. This
one only tests models we can self-host, serves every one of them through TEI on
the same GPU so the comparison is hardware-fair, and adds the two dimensions the
original harness does not cover:

  * candidate count — Hindsight's default is 300; cutting it is the cheapest
    available speedup, and this measures what it costs in ranking quality
  * extraction mode — "fact" stores LLM-extracted facts (~170 chars), "chunk"
    stores raw 3000-char chunks as memory units, which is the long-content
    regime where reranking is slowest and where a 512-token model truncates

Both retain and ground-truth annotation run gemini-3.7-flash through Vertex,
authenticated by ADC via a local token proxy, so no API key is needed.

Ground truth is per mode and is annotated once, then reused. Note that GT fact
text is matched exactly against recall results, so the GT is only valid for the
bank it was annotated against; changing the retain model invalidates it.

Run on a GPU host with Docker. TEI is pinned to 2 CPUs to match the dev
reranker container.

  python run_reranker_eval.py                 # everything
  python run_reranker_eval.py --models bge-reranker-base --modes chunk
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests
from testcontainers.core.container import DockerContainer

sys.path.insert(0, str(Path(__file__).parent / "src"))

import hindsight_benchmark.reranker as reranker_mod
from hindsight_benchmark.gcp import start_token_proxy, vertex_openai_upstream
from hindsight_benchmark.reranker import RerankerBenchmark
from hindsight_benchmark.locomo import parse_locomo_date

BENCHMARK_RUNNER_DIR = Path(__file__).parent
DATASETS_DIR = BENCHMARK_RUNNER_DIR / "datasets"
RESULTS_DIR = BENCHMARK_RUNNER_DIR.parent / "results" / "leaderboard" / "reranker-eval"
STATE_PATH = DATASETS_DIR / "reranker_eval_banks.json"

HINDSIGHT_IMAGE = "ghcr.io/vectorize-io/hindsight:latest"
HINDSIGHT_PORT = 8888
TEI_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:89-1.8.3"
TEI_HOST_PORT = 8081
TEI_CONTAINER_NAME = "tei-eval"

VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "model-benchmark-506614")
JUDGE_MODEL = "google/gemini-3.7-flash"
# Overridable so a bank build and a checkpoint bench can run concurrently on
# one host without fighting over the proxy port.
PROXY_PORT = int(os.environ.get("RRB_PROXY_PORT", "8899"))
# Docker's default bridge gateway. Containers reach host-published ports here;
# a loopback address inside a container points at the container itself.
DOCKER_HOST_GATEWAY = os.environ.get("DOCKER_HOST_GATEWAY", "172.17.0.1")

# Self-hostable candidates only. Hosted APIs are out of scope: this work has to
# serve on-prem deployments.
MODELS: list[tuple[str, str]] = [
    ("bge-reranker-base", "BAAI/bge-reranker-base"),
    ("gte-multilingual-reranker-base", "Alibaba-NLP/gte-multilingual-reranker-base"),
    ("bge-reranker-v2-m3", "BAAI/bge-reranker-v2-m3"),
    # English-only. Kept as a speed/quality reference point, not a viable
    # replacement, because multilingual support is a product requirement.
    ("ms-marco-minilm-l6", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    # Also English-only, and also not a candidate. It is here as an ARCHITECTURE
    # PROBE: a 22x768 ModernBERT against a 12x768 custom-arch model isolates how
    # much of gte-multilingual's slowness is the architecture versus TEI's lack
    # of optimisation for trust_remote_code models. That answers whether mmBERT
    # (the multilingual ModernBERT) is a sane student.
    ("gte-reranker-modernbert-base", "Alibaba-NLP/gte-reranker-modernbert-base"),
]

# A locally-trained checkpoint can be added at runtime with --extra-model
# "id=/path/to/checkpoint" so it is scored against the very same bank and
# ground truth as everything above. Ground truth matches fact text exactly, so
# a checkpoint measured against a different bank is not comparable.
INCUMBENT = "bge-reranker-base"

MODES: dict[str, dict] = {
    # min_annotated_frac floors are per mode because their healthy baselines
    # differ by construction: on the original conv-43 bank, fact annotated
    # 165/178 questions (0.927) while chunk annotated 70/178 (0.393) — chunk
    # units are ~3000-char windows and many answers simply have no matching
    # chunk. The floors sit well below healthy and well above the outage
    # signature (near 0.0 when recall or the judge LLM dies mid-run).
    "fact": {
        "extraction_mode": "concise",
        "description": "LLM-extracted facts (~170 chars)",
        "min_annotated_frac": 0.7,
    },
    "chunk": {
        "extraction_mode": "chunks",
        "description": "raw 3000-char chunks as memory units",
        "min_annotated_frac": 0.2,
    },
}

# 300 is Hindsight's default. The lower counts are the cheap-speedup question.
CANDIDATE_COUNTS = [300, 100, 50]

TARGET_CONVERSATION = "conv-43"


def conv_token(conversation: str) -> str:
    return conversation.replace("-", "")


def gt_path(mode: str, conversation: str = TARGET_CONVERSATION, suite: str = "") -> Path:
    if suite:
        return DATASETS_DIR / f"locomo_reranker_gt_{suite}_{mode}_{conv_token(conversation)}.json"
    return DATASETS_DIR / f"locomo_reranker_gt_{mode}.json"


def state_path(suite: str, mode: str) -> Path:
    # Suite state is per mode so the fact and chunk builds can run as two
    # concurrent processes without read-modify-write races on one file.
    if suite:
        return DATASETS_DIR / f"reranker_eval_banks_{suite}_{mode}.json"
    return STATE_PATH


def base_env(mode: str, llm_base_url: str) -> dict:
    """Container env shared by ingest and every measured run of a mode."""
    return {
        # pg0 stores under $HOME; without this the data does not survive a
        # container restart and the shared bank is lost between rerankers.
        "HOME": "/app/data",
        "HINDSIGHT_ENABLE_CP": "false",
        "HINDSIGHT_API_HOST": "0.0.0.0",
        "HINDSIGHT_API_PORT": str(HINDSIGHT_PORT),
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "local",
        "HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL": "BAAI/bge-small-en-v1.5",
        "HINDSIGHT_API_RERANKER_PROVIDER": "rrf",
        "HINDSIGHT_API_ENABLE_OBSERVATIONS": "false",
        "HINDSIGHT_API_EXTRACT_CAUSAL_LINKS": "false",
        "HINDSIGHT_API_RETAIN_EXTRACTION_MODE": MODES[mode]["extraction_mode"],
        "HINDSIGHT_API_LLM_TIMEOUT": "120",
        "HINDSIGHT_API_RETAIN_LLM_TIMEOUT": "600",
        "HINDSIGHT_API_DB_POOL_MIN_SIZE": "20",
        "HINDSIGHT_API_DB_COMMAND_TIMEOUT": "15",
        "HINDSIGHT_API_RERANKER_MAX_CANDIDATES": "300",
        # Retain and fallback LLM both go to Vertex through the token proxy.
        "HINDSIGHT_API_RETAIN_LLM_PROVIDER": "openai",
        "HINDSIGHT_API_RETAIN_LLM_MODEL": JUDGE_MODEL,
        "HINDSIGHT_API_RETAIN_LLM_API_KEY": "vertex-proxy",
        "HINDSIGHT_API_RETAIN_LLM_BASE_URL": llm_base_url,
        "HINDSIGHT_API_LLM_PROVIDER": "openai",
        "HINDSIGHT_API_LLM_MODEL": JUDGE_MODEL,
        "HINDSIGHT_API_LLM_API_KEY": "vertex-proxy",
        "HINDSIGHT_API_LLM_BASE_URL": llm_base_url,
    }


def reranker_env(mode: str, llm_base_url: str, tei_url: str, max_candidates: int) -> dict:
    env = base_env(mode, llm_base_url)
    env.update(
        {
            "HINDSIGHT_API_RERANKER_PROVIDER": "tei",
            "HINDSIGHT_API_RERANKER_TEI_URL": tei_url,
            "HINDSIGHT_API_RERANKER_MAX_CANDIDATES": str(max_candidates),
            "HINDSIGHT_API_ENABLE_RERANKING": "true",
        }
    )
    for knob in ("TEI_BATCH_SIZE", "TEI_MAX_CONCURRENT", "TEI_HTTP_TIMEOUT"):
        val = os.environ.get(f"RERANK_{knob}")
        if val:
            env[f"HINDSIGHT_API_RERANKER_{knob}"] = val
    return env


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


def start_hindsight(data_dir: str, env: dict) -> tuple[DockerContainer, str]:
    container = DockerContainer(HINDSIGHT_IMAGE)
    container.with_volume_mapping(data_dir, "/app/data", "rw")
    container.with_exposed_ports(HINDSIGHT_PORT)
    for key, value in env.items():
        container.with_env(key, value)
    container.start()
    api_url = f"http://localhost:{container.get_exposed_port(HINDSIGHT_PORT)}"
    deadline = time.time() + 420
    while time.time() < deadline:
        try:
            if requests.get(f"{api_url}/health", timeout=5).status_code == 200:
                return container, api_url
        except Exception:
            pass
        time.sleep(3)
    # testcontainers removes the container on stop, taking its logs with it, so
    # capture them here or the failure is undiagnosable after the fact.
    try:
        logs = container.get_logs()
        tail = b"\n".join(logs[0].splitlines()[-25:]).decode(errors="replace")
        print(f"  Hindsight container logs before giving up:\n{tail}", flush=True)
    except Exception:
        pass
    container.stop()
    raise TimeoutError(f"Hindsight never became healthy at {api_url}")


def start_tei(hf_model: str) -> str:
    """Start TEI on the GPU with the dev reranker's serving config.

    A LOCAL checkpoint path has to be mounted into the container and referred to
    by its in-container path. Passing a host path straight through as
    --model-id makes TEI treat it as a HuggingFace repo id and try to download
    `https://huggingface.co//home/andrew/...`, which 404s.
    """
    from pathlib import Path as _Path

    local = _Path(hf_model)
    mount: list[str] = []
    model_arg = hf_model
    if local.exists() and local.is_dir():
        mount = ["-v", f"{local.resolve()}:/model"]
        model_arg = "/model"
        print(f"  local checkpoint: mounting {local} -> /model", flush=True)

    subprocess.run(["docker", "rm", "-f", TEI_CONTAINER_NAME], capture_output=True)
    subprocess.run(
        [
            "docker", "run", "-d", "--name", TEI_CONTAINER_NAME,
            "--gpus", "all",
            *mount,
            # dev's tei-rerank container is limited to 2 CPUs; matching it keeps
            # tokenization throughput comparable to production.
            "--cpus=2",
            "-p", f"{TEI_HOST_PORT}:8081",
            "-v", "/data:/data",
            TEI_IMAGE,
            "--model-id", model_arg,
            "--hostname", "0.0.0.0", "--port", "8081",
            "--auto-truncate",
            "--max-client-batch-size", "128",
            "--max-batch-tokens", "32768",
            "--max-concurrent-requests", "512",
            "--tokenization-workers", "2",
            "--payload-limit", "2000000",
        ],
        check=True, capture_output=True,
    )
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            info = requests.get(f"http://localhost:{TEI_HOST_PORT}/info", timeout=5)
            if info.status_code == 200:
                details = info.json()
                print(
                    f"  TEI ready: {details.get('model_id')} "
                    f"max_input={details.get('max_input_length')} dtype={details.get('model_dtype')}",
                    flush=True,
                )
                return f"http://{DOCKER_HOST_GATEWAY}:{TEI_HOST_PORT}"
        except Exception:
            pass
        time.sleep(5)
    logs = subprocess.run(["docker", "logs", "--tail", "30", TEI_CONTAINER_NAME], capture_output=True, text=True)
    raise TimeoutError(f"TEI never became ready for {hf_model}\n{logs.stdout}\n{logs.stderr}")


def stop_tei() -> None:
    subprocess.run(["docker", "rm", "-f", TEI_CONTAINER_NAME], capture_output=True)


# ---------------------------------------------------------------------------
# Ingest and ground truth, once per mode
# ---------------------------------------------------------------------------


def load_state(path: Path = STATE_PATH) -> dict:
    if path.exists():
        with path.open() as fh:
            return json.load(fh)
    return {}


def save_state(state: dict, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(state, fh, indent=2)


def ingest_bank(api_url: str, mode: str, conversation: str = TARGET_CONVERSATION) -> str:
    from hindsight_client import Hindsight

    with (DATASETS_DIR / "locomo_quality.json").open() as fh:
        dataset = json.load(fh)
    item = next((i for i in dataset if i["sample_id"] == conversation), None)
    if item is None:
        raise ValueError(f"{conversation} not found in dataset")

    conv = item["conversation"]
    speaker_a, speaker_b = conv["speaker_a"], conv["speaker_b"]
    session_keys = sorted(k for k in conv if k.startswith("session_") and not k.endswith("_date_time"))

    client = Hindsight(base_url=api_url)
    bank_id = f"rrbeval_{mode}_{conv_token(conversation)}_{int(time.time())}"
    client.create_bank(bank_id=bank_id)
    print(f"  Ingesting {conversation} into {bank_id} (mode={mode})...", flush=True)

    for session_key in session_keys:
        if not isinstance(conv.get(session_key), list):
            continue
        client.retain(
            bank_id=bank_id,
            content=json.dumps(conv[session_key]),
            context=f"Conversation between {speaker_a} and {speaker_b} ({session_key})",
            timestamp=parse_locomo_date(conv[f"{session_key}_date_time"]),
            document_id=f"{conv_token(conversation)}_{session_key}_{bank_id}",
        )
        print(f"    ingested {session_key}", flush=True)
    return bank_id


def prepare_bank(
    mode: str,
    conversation: str,
    benchmark: RerankerBenchmark,
    llm_base_url: str,
    state: dict,
    spath: Path,
    suite: str,
    results_dir: Path,
    require_prepared: bool = False,
) -> tuple[str, str]:
    """Return (data_dir, bank_id) for one (mode, conversation), ingesting and
    annotating if needed. State is keyed by mode in the legacy single-bank
    file and by conversation in per-mode suite files."""
    key = conversation if suite else mode
    gpath = gt_path(mode, conversation, suite)
    entry = state.get(key)
    if entry and Path(entry["data_dir"]).exists() and gpath.exists():
        print(f"\n[{mode}/{conversation}] reusing bank {entry['bank_id']} and ground truth {gpath.name}")
        return entry["data_dir"], entry["bank_id"]

    if require_prepared:
        # A scoring run must never fall through into a paid ingest+annotation:
        # on a rebuilt box with a partial restore this silently rebuilds banks
        # with DIFFERENT annotations and voids cross-model comparability.
        raise SystemExit(
            f"--require-prepared: bank for {mode}/{conversation} is missing or torn "
            f"(state entry: {bool(entry)}, data_dir exists: "
            f"{bool(entry) and Path(entry['data_dir']).exists()}, gt exists: {gpath.exists()}). "
            f"Restore the suite snapshot or run the build explicitly."
        )

    # Ground truth matches fact text EXACTLY against recall results, so a fresh
    # ingest produces a different bank and different annotations. Scoring a new
    # model against that while older results sit in the results dir would
    # compare numbers from two different ground truths and silently void the
    # table.
    pattern = f"*__{mode}_{conv_token(conversation)}__*.json" if suite else f"*__{mode}__*.json"
    existing = sorted(results_dir.glob(pattern))
    if existing:
        raise SystemExit(
            f"Refusing to re-ingest and re-annotate '{mode}/{conversation}': {len(existing)} "
            f"result file(s) already exist from a previous bank (e.g. {existing[0].name}), but "
            f"the bank state is missing or stale. Ground truth is bank-specific, so a new "
            f"annotation would not be comparable.\n"
            f"Either run on the host that holds the original bank, or delete "
            f"{results_dir} and re-run every model against one fresh bank."
        )

    import tempfile

    prefix = f"hindsight-rerank-eval-{suite + '-' if suite else ''}{mode}-{conv_token(conversation)}-"
    data_dir = tempfile.mkdtemp(prefix=prefix)
    # mkdtemp is 0700 and owned by this user. On Linux the container's uid is
    # not remapped, so the Hindsight process cannot write its pg0 data and the
    # container never reaches healthy.
    os.chmod(data_dir, 0o777)
    print(f"\n{'='*70}\n[{mode}/{conversation}] ingest + annotate ({MODES[mode]['description']})\n{'='*70}", flush=True)

    # The floor is env-overridable so a conversation that legitimately sits
    # below it can be admitted after inspection without a code edit:
    #   RRB_MIN_ANNOTATED_FRAC_FACT / RRB_MIN_ANNOTATED_FRAC_CHUNK
    min_frac = float(os.environ.get(
        f"RRB_MIN_ANNOTATED_FRAC_{mode.upper()}", MODES[mode]["min_annotated_frac"]
    ))
    try:
        container = None
        try:
            container, api_url = start_hindsight(data_dir, base_env(mode, llm_base_url))
            bank_id = ingest_bank(api_url, mode, conversation)
            print(f"  Annotating ground truth with {JUDGE_MODEL}...", flush=True)
            benchmark.create_annotations(
                api_url, bank_id, gpath, conversation=conversation,
                min_annotated_frac=min_frac,
                dataset_path=DATASETS_DIR / "locomo_quality.json",
            )
            with gpath.open() as fh:
                gt = json.load(fh)
            gt["data_dir"] = data_dir
            gt["bank_id"] = bank_id
            gt["mode"] = mode
            gt["conversation"] = conversation
            gt["annotation_model"] = JUDGE_MODEL
            with gpath.open("w") as fh:
                json.dump(gt, fh, indent=2)
        finally:
            if container is not None:
                container.stop()
    except BaseException:
        # This attempt's bank is unusable (no state entry, maybe no gt).
        # Remove its ~1GB data dir or repeated resumes fill /tmp with
        # orphans no state file points at and no snapshot ever collects.
        shutil.rmtree(data_dir, ignore_errors=True)
        raise

    state[key] = {"data_dir": data_dir, "bank_id": bank_id, "conversation": conversation}
    save_state(state, spath)
    return data_dir, bank_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default=None, help="Comma-separated model ids (default: all)")
    parser.add_argument("--modes", default="fact,chunk", help="Comma-separated modes")
    parser.add_argument(
        "--candidates",
        default=None,
        help="Comma-separated candidate counts. Default: 300 for every model, plus "
        f"{CANDIDATE_COUNTS[1:]} for {INCUMBENT} only.",
    )
    parser.add_argument("--redo", action="store_true", help="Re-run rows whose result file already exists")
    parser.add_argument(
        "--extra-model",
        action="append",
        default=[],
        help="Add a model as id=path_or_hf_id (repeatable). Use for a locally trained "
        "checkpoint so it is scored against the same bank and ground truth.",
    )
    parser.add_argument(
        "--conversations",
        default=TARGET_CONVERSATION,
        help="Comma-separated LoCoMo sample_ids to evaluate over (one bank each). "
        f"Default: {TARGET_CONVERSATION}, the original single-conversation bank.",
    )
    parser.add_argument(
        "--suite",
        default="",
        help="Name for a separate bank suite (e.g. 'xl'). Required whenever "
        "--conversations differs from the default: it routes state, ground truth, "
        "and results to suite-specific paths so the original bank and its result "
        "history are never touched.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Ingest and annotate the requested banks, then exit without scoring.",
    )
    parser.add_argument(
        "--require-prepared",
        action="store_true",
        help="Refuse to ingest or annotate anything: every requested bank must "
        "already exist (restored or previously built). Use for scoring runs so a "
        "torn restore can never silently trigger a paid re-annotation.",
    )
    args = parser.parse_args()

    conversations = [c.strip() for c in args.conversations.split(",") if c.strip()]
    if args.suite and not re.fullmatch(r"[a-z0-9]+", args.suite):
        print(f"--suite must be lowercase alphanumeric, got {args.suite!r}", file=sys.stderr)
        return 1
    if conversations != [TARGET_CONVERSATION] and not args.suite:
        print(
            "--conversations other than the default requires --suite: the legacy "
            "state/ground-truth/results paths belong to the original bank.",
            file=sys.stderr,
        )
        return 1

    for spec in args.extra_model:
        if "=" not in spec:
            print(f"--extra-model must be id=path, got {spec!r}", file=sys.stderr)
            return 1
        mid, path = spec.split("=", 1)
        MODELS.append((mid.strip(), path.strip()))
        print(f"Added extra model {mid.strip()} -> {path.strip()}")

    results_dir = RESULTS_DIR.parent / f"reranker-eval-{args.suite}" if args.suite else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    # Keep these results out of results/leaderboard/reranker/, which holds the
    # committed hosted-API runs annotated with a different model.
    reranker_mod.RERANKER_RESULTS_DIR = results_dir

    models = MODELS
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = [m for m in MODELS if m[0] in wanted]
        if not models:
            print(f"No models matched {wanted}; known: {[m[0] for m in MODELS]}", file=sys.stderr)
            return 1
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in MODES:
            print(f"Unknown mode {mode}; known: {sorted(MODES)}", file=sys.stderr)
            return 1

    print(f"Starting Vertex token proxy for {VERTEX_PROJECT} ...", flush=True)
    start_token_proxy(vertex_openai_upstream(VERTEX_PROJECT), PROXY_PORT, bind_host="0.0.0.0")
    # The benchmark process talks to the proxy over loopback; the Hindsight
    # container has to come in over the docker bridge gateway.
    host_llm_url = f"http://127.0.0.1:{PROXY_PORT}/v1"
    container_llm_url = f"http://{DOCKER_HOST_GATEWAY}:{PROXY_PORT}/v1"

    benchmark = RerankerBenchmark()
    # Point annotation at Vertex instead of the Gemini API key path.
    from openai import OpenAI

    benchmark.llm_client = OpenAI(api_key="vertex-proxy", base_url=host_llm_url, timeout=600.0)
    benchmark.annotation_model = JUDGE_MODEL

    prepared: dict[tuple[str, str], tuple[str, str]] = {}
    prepare_failures: list[tuple[str, str, str]] = []
    for mode in modes:
        spath = state_path(args.suite, mode)
        state = load_state(spath)
        for conv in conversations:
            # One transient container or annotation failure must not kill the
            # remaining conversations on an unattended build. SystemExit from
            # --require-prepared is BaseException and still aborts everything,
            # which is the point of that flag.
            try:
                prepared[(mode, conv)] = prepare_bank(
                    mode, conv, benchmark, container_llm_url, state, spath,
                    args.suite, results_dir, require_prepared=args.require_prepared,
                )
            except Exception as exc:
                print(f"PREPARE_FAILED {mode}/{conv}: {type(exc).__name__}: {exc}", flush=True)
                prepare_failures.append((mode, conv, str(exc)))

    if prepare_failures:
        print(f"\n{len(prepare_failures)} bank(s) FAILED to prepare:", flush=True)
        for mode, conv, msg in prepare_failures:
            print(f"  {mode}/{conv}: {msg[:140]}")

    if args.prepare_only:
        print(f"\nPrepared {len(prepared)} bank(s); --prepare-only, skipping scoring.")
        return 1 if prepare_failures else 0

    # Build the run plan. Only the incumbent gets the candidate sweep; the
    # comparison models are measured at the production default.
    plan: list[tuple[str, str, str, str, int]] = []
    for model_id, hf_model in models:
        for mode in modes:
            counts = (
                [int(c) for c in args.candidates.split(",")]
                if args.candidates
                else (CANDIDATE_COUNTS if model_id == INCUMBENT else [CANDIDATE_COUNTS[0]])
            )
            for count in counts:
                for conv in conversations:
                    if (mode, conv) in prepared:
                        plan.append((model_id, hf_model, mode, conv, count))

    print(f"\n{len(plan)} quality runs planned\n", flush=True)

    completed, failed = 0, 0
    current_model: str | None = None
    tei_url = ""
    try:
        for model_id, hf_model, mode, conv, count in plan:
            run_id = (
                f"{model_id}__{mode}_{conv_token(conv)}__c{count}"
                if args.suite
                else f"{model_id}__{mode}__c{count}"
            )
            out_path = results_dir / f"{run_id}.json"
            if out_path.exists() and not args.redo:
                print(f"SKIP {run_id} (result exists)", flush=True)
                continue

            if model_id != current_model:
                stop_tei()
                if hf_model.startswith(("http://", "https://")):
                    print(f"\n{'='*70}\nExternal endpoint {hf_model}\n{'='*70}", flush=True)
                    tei_url = hf_model
                else:
                    print(f"\n{'='*70}\nServing {hf_model}\n{'='*70}", flush=True)
                    tei_url = start_tei(hf_model)
                current_model = model_id

            data_dir, bank_id = prepared[(mode, conv)]
            print(f"\n--- {run_id} ---", flush=True)
            container = None
            try:
                env = reranker_env(mode, container_llm_url, tei_url, count)
                container, api_url = start_hindsight(data_dir, env)
                result = benchmark.run_with_bank(
                    reranker_id=run_id,
                    provider="tei",
                    model=hf_model,
                    api_url=api_url,
                    bank_id=bank_id,
                    gt_path=gt_path(mode, conv, args.suite),
                    sample_id=conv,
                )
                # Annotate the row with the dimensions the base harness does not know about.
                with out_path.open() as fh:
                    saved = json.load(fh)
                saved.update(
                    {
                        "model_id": model_id,
                        "hf_model": hf_model,
                        "mode": mode,
                        "conversation": conv,
                        "suite": args.suite,
                        "extraction_mode": MODES[mode]["extraction_mode"],
                        "max_candidates": count,
                        "serving": ("external:" + hf_model
                                    if hf_model.startswith(("http://", "https://"))
                                    else "tei-1.8.3-l4-fp16"),
                        "annotation_model": JUDGE_MODEL,
                    }
                )
                with out_path.open("w") as fh:
                    json.dump(saved, fh, indent=2)
                print(f"  MRR={result['mrr']} R@1={result['recall_at_1']} R@5={result['recall_at_5']}", flush=True)
                completed += 1
            except Exception as exc:
                print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
                failed += 1
            finally:
                if container is not None:
                    container.stop()
    finally:
        stop_tei()

    if args.suite:
        # One aggregate row per (model, mode, count): micro-average over every
        # question in every conversation bank, recomputed from the persisted
        # per-question ranks rather than re-averaging rounded averages.
        for model_id, _hf in models:
            for mode in modes:
                counts_seen = sorted({c for m, _h, mo, _cv, c in plan if m == model_id and mo == mode})
                for count in counts_seen:
                    rows = []
                    for conv in conversations:
                        p = results_dir / f"{model_id}__{mode}_{conv_token(conv)}__c{count}.json"
                        if p.exists():
                            with p.open() as fh:
                                rows.append(json.load(fh))
                    if len(rows) != len(conversations):
                        # Never let a partial run overwrite a complete ALL file
                        # under the same name: the headline number would move
                        # while the filename still claims full coverage.
                        print(
                            f"AGG SKIPPED {model_id}__{mode}__c{count}: only {len(rows)}/"
                            f"{len(conversations)} conversation rows present; not writing ALL.",
                            flush=True,
                        )
                        continue
                    ranks = [q["rank"] for r in rows for q in r.get("per_question", [])]
                    if not ranks:
                        print(f"AGG SKIPPED {model_id}__{mode}__c{count}: no per-question ranks.", flush=True)
                        continue
                    # A narrower manual run must not clobber a wider ALL file
                    # either: the CLI conversation list is the run's scope, not
                    # proof of full coverage.
                    all_path = results_dir / f"{model_id}__{mode}_ALL__c{count}.json"
                    if all_path.exists():
                        with all_path.open() as fh:
                            prior = json.load(fh)
                        if len(prior.get("conversations", [])) > len(conversations):
                            print(
                                f"AGG SKIPPED {all_path.name}: existing file covers "
                                f"{len(prior['conversations'])} conversations, this run only "
                                f"{len(conversations)}.",
                                flush=True,
                            )
                            continue
                    n = len(ranks)
                    agg = {
                        "reranker_id": f"{model_id}__{mode}_ALL__c{count}",
                        "model_id": model_id,
                        "mode": mode,
                        "max_candidates": count,
                        "suite": args.suite,
                        "conversations": [r["sample_id"] for r in rows],
                        "total_questions": n,
                        "mrr": round(sum(1.0 / r for r in ranks if r) / n, 4),
                        "recall_at_1": round(sum(1 for r in ranks if r == 1) / n, 4),
                        "recall_at_3": round(sum(1 for r in ranks if r and r <= 3) / n, 4),
                        "recall_at_5": round(sum(1 for r in ranks if r and r <= 5) / n, 4),
                        "per_conversation": [
                            {
                                "sample_id": r["sample_id"],
                                "total_questions": r["total_questions"],
                                "mrr": r["mrr"],
                                "recall_at_1": r["recall_at_1"],
                                "recall_at_5": r["recall_at_5"],
                            }
                            for r in rows
                        ],
                        # Full rank vector so a paired test between two models
                        # reads two ALL files, not eighteen per-conversation ones.
                        "per_question": [
                            {"sample_id": r["sample_id"], **q}
                            for r in rows
                            for q in r.get("per_question", [])
                        ],
                    }
                    with (results_dir / f"{agg['reranker_id']}.json").open("w") as fh:
                        json.dump(agg, fh, indent=2)
                    print(
                        f"AGG {agg['reranker_id']}: MRR={agg['mrr']} R@1={agg['recall_at_1']} "
                        f"R@5={agg['recall_at_5']} over {n} questions / {len(rows)} banks",
                        flush=True,
                    )

    print(
        f"\n{'='*70}\nDone. {completed} completed, {failed} failed, "
        f"{len(prepare_failures)} bank(s) unprepared. Results in {results_dir}"
    )
    return 0 if failed == 0 and not prepare_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
