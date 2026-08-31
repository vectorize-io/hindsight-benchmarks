#!/usr/bin/env python3
"""
Deploy a model from Vertex AI Model Garden, run the benchmarks against it,
and tear the deployment down.

    uv run python run_gcp_model.py --model google/gemma-4@gemma-4-31b-it \
        --model-id gemma-4-31b --served-model-name gemma-4-31b-it

The endpoint serves an OpenAI-compatible chat-completions route via vLLM.
Vertex wants an hourly-expiring OAuth bearer token, so a local proxy injects a
fresh token per request; the fast benchmark and the Hindsight daemon both point
at the proxy and never see GCP credentials.

Deployment metadata (machine type, GPU, region) is written into the result
file so every speed number carries the hardware it was measured on.

Teardown deletes ONLY resources this script created (endpoint display names
are prefixed "hbench-"). Pass --keep to leave the deployment running.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BENCHMARK_RUNNER_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCHMARK_RUNNER_DIR / "src"))

from hindsight_benchmark.benchmark import run_url, save_run  # noqa: E402
from hindsight_benchmark.gcp import start_token_proxy  # noqa: E402

PROXY_PORT = 8811
ENDPOINT_PREFIX = "hbench-"
PROVIDER_ID = "gcp"


# ── gcloud helpers ────────────────────────────────────────────────────────────

def _gcloud(args: list[str], project: str, timeout: int = 7200) -> str:
    cmd = ["gcloud", *args, f"--project={project}", f"--billing-project={project}"]
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        # Long-running deploys emit thousands of progress dots that would
        # flood the kept stderr and hide the actual error and operation id.
        stderr = "\n".join(
            line for line in result.stderr.splitlines()
            if line.strip(". ") != ""
        )
        raise RuntimeError(f"gcloud failed:\n{stderr[-4000:]}")
    return result.stdout


def deploy(model: str, project: str, region: str, machine_type: str | None,
           accelerator_type: str | None, accelerator_count: int | None,
           hf_token: str | None, container_args: str | None = None,
           predict_route: str | None = None, health_route: str | None = None,
           container_image_uri: str | None = None,
           container_port: int | None = None) -> str:
    """Deploy and return the endpoint resource name."""
    display_name = f"{ENDPOINT_PREFIX}{model.split('/')[-1].split('@')[-1]}-{int(time.time())}"
    args = [
        "ai", "model-garden", "models", "deploy",
        f"--model={model}",
        f"--region={region}",
        f"--endpoint-display-name={display_name}",
        "--accept-eula",
    ]
    if machine_type:
        args.append(f"--machine-type={machine_type}")
    if accelerator_type:
        args.append(f"--accelerator-type={accelerator_type}")
    if accelerator_count:
        args.append(f"--accelerator-count={accelerator_count}")
    if hf_token:
        args.append(f"--hugging-face-access-token={hf_token}")
    if container_args:
        # Full replacement of the serving command. Needed for generic
        # HuggingFace deploys, whose fallback config serves a 2048-token
        # context that no benchmark request fits into. Vertex ignores
        # --container-args unless the image is pinned too.
        args.append(f"--container-args={container_args}")
    if container_image_uri:
        args.append(f"--container-image-uri={container_image_uri}")
    if predict_route:
        args.append(f"--container-predict-route={predict_route}")
    if health_route:
        args.append(f"--container-health-route={health_route}")
    if container_port:
        # Must match the --port the serving command listens on, otherwise
        # Vertex health-checks a port nothing is bound to.
        args.append(f"--container-ports={container_port}")

    _gcloud(args, project)

    # Find the endpoint we just created by its display name.
    out = _gcloud([
        "ai", "endpoints", "list", f"--region={region}",
        f"--filter=displayName={display_name}", "--format=json",
    ], project)
    endpoints = json.loads(out)
    if not endpoints:
        raise RuntimeError(f"deploy finished but endpoint {display_name} not found")
    return endpoints[0]["name"]  # projects/N/locations/R/endpoints/ID


def describe_deployment(endpoint_name: str, project: str, region: str) -> dict:
    out = _gcloud([
        "ai", "endpoints", "describe", endpoint_name.split("/")[-1],
        f"--region={region}", "--format=json",
    ], project)
    ep = json.loads(out)
    deployed = (ep.get("deployedModels") or [{}])[0]
    machine = (deployed.get("dedicatedResources") or {}).get("machineSpec", {})

    # The serving container (image + launch args, e.g. context length and
    # memory fraction) shapes speed as much as the GPU does; record it so the
    # row is reproducible. Lives on the Model resource, not the endpoint.
    container = {}
    if deployed.get("model"):
        try:
            mout = _gcloud([
                "ai", "models", "describe", deployed["model"].split("/")[-1],
                f"--region={region}", "--format=json",
            ], project)
            spec = json.loads(mout).get("containerSpec", {})
            container = {
                "image": spec.get("imageUri", ""),
                "args": spec.get("args", []),
            }
        except Exception as e:
            print(f"Warning: could not read container spec ({e})", flush=True)

    return {
        "endpoint_name": ep["name"],
        "dedicated_domain": ep.get("dedicatedEndpointDns", ""),
        "machine_type": machine.get("machineType"),
        "accelerator_type": machine.get("acceleratorType"),
        "accelerator_count": machine.get("acceleratorCount"),
        "region": region,
        "serving": "vertex-model-garden",
        "container": container,
    }


TEARDOWN_GRACE_S = 600
KEEP_FLAG = "/tmp/hbench_keep"
PROCEED_FLAG = "/tmp/hbench_teardown_now"


def _teardown_grace(endpoint_id: str):
    """Hold before deleting so a human can intervene. Touch /tmp/hbench_keep
    to abort the teardown (endpoint stays up, KEEPS BILLING) or
    /tmp/hbench_teardown_now to proceed immediately. With no decision the
    teardown proceeds after the grace period, so nothing is ever orphaned."""
    deadline = time.time() + TEARDOWN_GRACE_S
    print(f"Teardown grace for {endpoint_id}: holding up to "
          f"{TEARDOWN_GRACE_S // 60} min. touch {KEEP_FLAG} to keep, "
          f"{PROCEED_FLAG} to proceed now.", flush=True)
    while time.time() < deadline:
        if os.path.exists(KEEP_FLAG):
            os.remove(KEEP_FLAG)
            print(f"Teardown ABORTED for {endpoint_id} via {KEEP_FLAG}; "
                  "endpoint left running (COSTS MONEY)", flush=True)
            return False
        if os.path.exists(PROCEED_FLAG):
            os.remove(PROCEED_FLAG)
            print(f"Teardown proceeding for {endpoint_id} via {PROCEED_FLAG}",
                  flush=True)
            return True
        time.sleep(10)
    print(f"Teardown grace expired for {endpoint_id}; proceeding", flush=True)
    return True


def teardown(endpoint_name: str, project: str, region: str):
    """Undeploy and delete the endpoint this script created. Refuses anything
    not carrying the hbench- prefix, so pre-existing resources are untouchable
    from here."""
    endpoint_id = endpoint_name.split("/")[-1]
    if not _teardown_grace(endpoint_id):
        return
    out = _gcloud([
        "ai", "endpoints", "describe", endpoint_id,
        f"--region={region}", "--format=json",
    ], project)
    ep = json.loads(out)
    if not ep.get("displayName", "").startswith(ENDPOINT_PREFIX):
        raise RuntimeError(
            f"Refusing to delete endpoint {endpoint_id}: display name "
            f"{ep.get('displayName')!r} was not created by this script"
        )
    model_names = []
    for dm in ep.get("deployedModels") or []:
        model_names.append(dm.get("model"))
        _gcloud([
            "ai", "endpoints", "undeploy-model", endpoint_id,
            f"--deployed-model-id={dm['id']}", f"--region={region}", "--quiet",
        ], project)
    _gcloud(["ai", "endpoints", "delete", endpoint_id,
             f"--region={region}", "--quiet"], project)
    # Delete the uploaded Model resources backing this endpoint (also created
    # by the deploy call; nothing else references them).
    for mn in model_names:
        if mn:
            _gcloud(["ai", "models", "delete", mn.split("/")[-1],
                     f"--region={region}", "--quiet"], project)
    print(f"Teardown complete for {endpoint_id}", flush=True)


# ── benchmark orchestration ───────────────────────────────────────────────────

def _served_context_from_logs(endpoint_id: str, project: str) -> int | None:
    """Read the context window the engine actually booted with.

    Vertex's dedicated endpoints route only the predict path, so /v1/models is
    unreachable for most deployments. vLLM states the real number on startup
    ("Maximum concurrency for N tokens per request"), which is the served
    value whether it came from --max-model-len or the model's own config.
    """
    out = subprocess.run(
        ["gcloud", "logging", "read",
         f'resource.labels.endpoint_id="{endpoint_id}" "Maximum concurrency for"',
         f"--project={project}", "--limit=1", "--freshness=6h",
         "--format=value(jsonPayload.message)"],
        capture_output=True, text=True, timeout=180,
    ).stdout
    m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request", out)
    return int(m.group(1).replace(",", "")) if m else None


def _wait_engine_ready(proxy_url: str, served_model_name: str,
                       timeout_s: int = 900, interval_s: int = 20):
    """Block until the vLLM engine answers a real completion request.

    Vertex marks the deployment ready off /health, which vLLM's API server
    answers 200 while the engine is still loading weights and compiling.
    Requests in that window get 503, so probe with a 1-token completion
    until one succeeds.
    """
    import urllib.error
    import urllib.request
    body = json.dumps({
        "model": served_model_name,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }).encode()
    deadline = time.time() + timeout_s
    attempt = 0
    while True:
        attempt += 1
        try:
            req = urllib.request.Request(
                f"{proxy_url}/v1/chat/completions", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp.read()
            print(f"Engine ready (warmup attempt {attempt})", flush=True)
            return
        except urllib.error.HTTPError as e:
            # 503: engine still loading. 404: dedicated-endpoint route not
            # propagated yet. Both clear on their own.
            if e.code not in (503, 404) or time.time() > deadline:
                raise
            print(f"Engine warming up ({e.code}, attempt {attempt}); "
                  f"retrying in {interval_s}s", flush=True)
        except OSError as e:
            if time.time() > deadline:
                raise
            print(f"Engine not reachable yet ({e}, attempt {attempt}); "
                  f"retrying in {interval_s}s", flush=True)
        time.sleep(interval_s)


def _merge_deployment_metadata(model_id: str, deployment: dict):
    path = BENCHMARK_RUNNER_DIR.parent / "results" / "leaderboard" / "llm" / f"{PROVIDER_ID}-{model_id}.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data["deployment"] = deployment
    path.write_text(json.dumps(data, indent=2))
    print(f"Deployment metadata written to {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="Model Garden id (publisher/model@version) or HuggingFace id")
    p.add_argument("--model-id", required=True, help="Leaderboard slug for result files")
    p.add_argument("--served-model-name", required=True,
                   help="Model name the vLLM server expects in the request body")
    p.add_argument("--name", default=None, help="Display name (defaults to --model-id)")
    p.add_argument("--project", default="model-benchmark-506614")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--machine-type", default=None, help="Override Model Garden's default")
    p.add_argument("--accelerator-type", default=None)
    p.add_argument("--accelerator-count", type=int, default=None)
    p.add_argument("--hf-token", default=None, help="HuggingFace token for gated models")
    p.add_argument("--endpoint", default=None,
                   help="Reuse an existing hbench- endpoint instead of deploying")
    p.add_argument("--dataset", default="locomo_3k_50")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--reasoning-effort", default="",
                   help="'none' disables thinking; other values pin the effort. "
                        "Applied to both the fast benchmark and the quality daemon.")
    p.add_argument("--retain-concurrency", type=int, default=None,
                   help="Extraction requests the daemon keeps in flight during "
                        "retain. Measure it with concurrency_sweep.py against the "
                        "live endpoint; recorded in the row's deployment block.")
    p.add_argument("--extra-body", default="",
                   help="JSON dict merged into every LLM request body, for models "
                        "whose thinking control is a body field, e.g. "
                        '\'{"chat_template_kwargs": {"enable_thinking": false}}\' '
                        "for Gemma/Qwen on SGLang/vLLM.")
    p.add_argument("--skip-quality", action="store_true", help="Fast benchmark only")
    p.add_argument("--container-args", default=None,
                   help="Comma-separated full serving command override "
                        "(gcloud --container-args). Use for HF-path deploys.")
    p.add_argument("--container-image-uri", default=None,
                   help="Serving image URI; required for --container-args to take "
                        "effect on HuggingFace-path deploys.")
    p.add_argument("--predict-verb", action="store_true",
                   help="Route through endpoints:predict instead of the "
                        "OpenAI-style /chat/completions passthrough, for "
                        "endpoints that do not expose it.")
    p.add_argument("--container-port", type=int, default=None,
                   help="Port the serving command binds; passed to Vertex so "
                        "health checks and predictions reach it.")
    p.add_argument("--predict-route", default=None,
                   help="Container predict route (needed when the serving "
                        "command uses vLLM's standard OpenAI entrypoint)")
    p.add_argument("--health-route", default=None)
    p.add_argument("--proxy-port", type=int, default=PROXY_PORT,
                   help="Local port for the endpoint token proxy; the judge proxy uses port+1. "
                        "Give concurrent runs distinct ports.")
    p.add_argument("--keep", action="store_true", help="Skip teardown at the end")
    args = p.parse_args()

    extra_body = json.loads(args.extra_body) if args.extra_body else None

    endpoint_name = args.endpoint
    if not endpoint_name:
        print(f"Deploying {args.model} (this can take 15-30 minutes)...", flush=True)
        endpoint_name = deploy(
            args.model, args.project, args.region,
            args.machine_type, args.accelerator_type, args.accelerator_count,
            args.hf_token, args.container_args,
            args.predict_route, args.health_route,
            args.container_image_uri,
            args.container_port,
        )

    try:
        deployment = describe_deployment(endpoint_name, args.project, args.region)
        if args.reasoning_effort:
            deployment["reasoning_effort"] = args.reasoning_effort
        if extra_body:
            deployment["llm_extra_body"] = extra_body
        print(f"Deployment: {json.dumps(deployment, indent=2)}", flush=True)
        domain = deployment["dedicated_domain"]
        if not domain:
            raise RuntimeError("Endpoint has no dedicated domain; deploy with a dedicated endpoint")

        # vLLM's OpenAI route on a dedicated Vertex endpoint. The proxy serves
        # /v1/... locally and maps it onto the endpoint's chat completions path.
        endpoint_id = endpoint_name.split("/")[-1]
        project_number = endpoint_name.split("/")[1]
        upstream = (
            f"https://{domain}/v1/projects/{project_number}/locations/"
            f"{args.region}/endpoints/{endpoint_id}"
        )
        if args.predict_verb:
            # Endpoints without the OpenAI-style passthrough still accept the
            # same body on :predict and return the container's reply verbatim.
            upstream += ":predict"
        start_token_proxy(upstream, args.proxy_port)
        proxy_url = f"http://127.0.0.1:{args.proxy_port}"

        _wait_engine_ready(proxy_url, args.served_model_name)

        # The serving stack reports the effective context window on its models
        # endpoint; record it so the row states what was actually served.
        try:
            import urllib.request
            with urllib.request.urlopen(f"{proxy_url}/v1/models", timeout=60) as resp:
                models_info = json.loads(resp.read()).get("data", [])
            if models_info and models_info[0].get("max_model_len"):
                deployment["max_model_len"] = models_info[0]["max_model_len"]
                deployment["max_model_len_source"] = "serving /v1/models endpoint"
        except Exception as e:
            print(f"/v1/models unavailable ({e}); reading served context "
                  "from the engine's startup log", flush=True)
        if not deployment.get("max_model_len"):
            served = _served_context_from_logs(endpoint_name.split("/")[-1],
                                               args.project)
            if served:
                deployment["max_model_len"] = served
                deployment["max_model_len_source"] = (
                    "engine startup log (vLLM reported served context)")
                print(f"Served context length: {served:,} tokens", flush=True)
            else:
                deployment["max_model_len_source"] = "UNVERIFIED: not read from the server"
                print("WARNING: could not verify served context length",
                      flush=True)

        # Fast benchmark (JSON conformance + speed)
        run = run_url(
            proxy_url,
            model_id=args.model_id,
            model_name=args.name or args.model_id,
            provider_id=PROVIDER_ID,
            concurrency=args.concurrency,
            dataset=args.dataset,
            api_key="proxy",
            model=args.served_model_name,
            reasoning_effort=args.reasoning_effort,
            extra_body=extra_body,
        )
        save_run(PROVIDER_ID, args.model_id, run)

        # Quality benchmark through the embedded Hindsight daemon
        if not args.skip_quality:
            from hindsight_embed import DaemonEmbedManager
            from hindsight_benchmark.quality import QualityBenchmark
            from run_all_quality import BASE_CONFIG
            run_ts = int(time.time())
            config = {
                **BASE_CONFIG,
                "HINDSIGHT_API_LLM_PROVIDER": "openai",
                "HINDSIGHT_API_LLM_MODEL": args.served_model_name,
                "HINDSIGHT_API_LLM_BASE_URL": f"{proxy_url}/v1",
                "HINDSIGHT_API_LLM_API_KEY": "proxy",
                "HINDSIGHT_API_DATABASE_URL": f"pg0://gcp-bench-{run_ts}",
            }
            if args.retain_concurrency:
                config["HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT"] = str(args.retain_concurrency)
                deployment["retain_concurrency"] = args.retain_concurrency
            else:
                deployment["retain_concurrency"] = int(
                    BASE_CONFIG["HINDSIGHT_API_RETAIN_LLM_MAX_CONCURRENT"])
            if args.reasoning_effort:
                config["HINDSIGHT_API_LLM_REASONING_EFFORT"] = args.reasoning_effort
            if extra_body:
                config["HINDSIGHT_API_LLM_EXTRA_BODY"] = json.dumps(extra_body)
            benchmark = QualityBenchmark(vertex_project=args.project, vertex_judge_port=args.proxy_port + 1)
            mgr = DaemonEmbedManager()
            profile = f"gcpb-{run_ts}"
            if not mgr.ensure_running(config, profile):
                raise RuntimeError("Hindsight daemon failed to start")
            try:
                benchmark.run(model_id=args.model_id, provider_id=PROVIDER_ID,
                              api_url=mgr.get_url(profile))
            finally:
                mgr.stop(profile)

        _merge_deployment_metadata(args.model_id, deployment)

    finally:
        if args.keep:
            print(f"--keep set: endpoint {endpoint_name} left running (COSTS MONEY)", flush=True)
        else:
            print("Tearing down deployment...", flush=True)
            teardown(endpoint_name, args.project, args.region)


if __name__ == "__main__":
    main()
