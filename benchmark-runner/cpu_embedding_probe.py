"""Measure CPU inference on a fixed LoCoMo sample; shared-host timings are indicative."""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from embedding_experiment import MODELS
from rerank_eval import DATA, digest, read_json, versions, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding", choices=["bge-small", "granite", "leaf"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    fixture = read_json(DATA / "locomo10-top100.json.gz")
    ids = sorted(fixture["corpus"])
    doc_ids = [ids[i] for i in np.linspace(0, len(ids) - 1, 256, dtype=int)]
    queries = [
        fixture["queries"][i]
        for i in np.linspace(0, len(fixture["queries"]) - 1, 64, dtype=int)
    ]
    docs = [fixture["corpus"][d] for d in doc_ids]
    texts = [q["query"] for q in queries]
    config = MODELS[args.embedding]
    started = time.perf_counter()
    model = SentenceTransformer(
        config["model"],
        revision=config["revision"],
        device="cpu",
        local_files_only=True,
        trust_remote_code=False,
        model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
    )
    load_s = time.perf_counter() - started

    def encode(values, kind):
        return model.encode(
            values,
            prompt=config[kind + "_prompt"],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

    encode(docs[:4], "document")
    encode(texts[:1], "query")
    document_times = []
    for _ in range(3):
        started = time.perf_counter()
        vectors = encode(docs, "document")
        document_times.append(time.perf_counter() - started)
    query_times = []
    query_vectors = []
    for query in texts:
        started = time.perf_counter()
        query_vectors.append(encode([query], "query")[0])
        query_times.append(time.perf_counter() - started)
    norms = np.linalg.norm(np.concatenate([vectors, np.array(query_vectors)]), axis=1)
    if not np.allclose(norms, 1, atol=1e-5):
        raise ValueError("Non-unit embeddings in normalized CPU inference")
    cpu_info = Path("/proc/cpuinfo")
    cpu = platform.processor() or platform.machine()
    if cpu_info.exists():
        cpu = next(
            line.split(":", 1)[1].strip()
            for line in cpu_info.read_text().splitlines()
            if line.startswith("model name")
        )
    result = {
        "embedding": args.embedding,
        "config": config,
        "device": "cpu",
        "cpu": cpu,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "dtype": str(next(model.parameters()).dtype),
        "attention": "sdpa",
        "parameters": sum(p.numel() for p in model.parameters()),
        "dimension": vectors.shape[1],
        "max_seq_length": model.max_seq_length,
        "model_load_s": load_s,
        "document_count": len(docs),
        "document_batch_size": 32,
        "document_pass_seconds": document_times,
        "median_documents_per_second": len(docs) / float(np.median(document_times)),
        "query_count": len(texts),
        "single_query_latency_s": query_times,
        "single_query_p50_s": float(np.percentile(query_times, 50)),
        "single_query_p95_s": float(np.percentile(query_times, 95)),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "l2_norm_min": float(norms.min()),
        "l2_norm_max": float(norms.max()),
        "sample_sha256": digest({"doc_ids": doc_ids, "queries": queries, "docs": docs}),
        "baseline_fixture_sha256": fixture["fixture_sha256"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": versions(),
        "timing_conditions": "Fresh process, cached model weights, float32 CPU SDPA, 4 threads by default, warmup excluded. Shared host with other work; indicative timings, not an isolated speed benchmark. Peak RSS includes Python/Torch and model load.",
    }
    write_json(args.output, result)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
