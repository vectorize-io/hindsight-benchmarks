"""Pinned MemGovern inputs and resumable, label-independent embedding preparation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from embedding_experiment import MODELS, encode
from rerank_eval import digest, read_json, versions, write_json

REVISION = "f9d4294be4a24b8c16d6bfb59d56a6fec4bd95c0"
MODEL_NAMES = ("bge-small", "granite", "leaf", "voyage-1024")
DEFAULT_DATA = Path("/mnt/optane/hindsight-reranker-work/memgovern-data")
DEFAULT_WORK = Path("/mnt/optane/hindsight-reranker-work/memgovern-work")
_MODEL = None
_CONFIG = None


def read_items(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError(f"Empty or duplicate source IDs: {path}")
    if any(not isinstance(i, str) for i in ids):
        raise ValueError("Source IDs must be strings")
    if any(not isinstance(row["text"], str) or not row["text"].strip() for row in rows):
        raise ValueError(f"Empty source text: {path}")
    return sorted(rows, key=lambda row: row["id"])


def selected_queries(rows, repository, count=200, seed=20261003):
    if count > len(rows) or count < 1:
        raise ValueError("Query count must fit the full repository")
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{repository}\0{row['id']}".encode()
        ).digest(),
    )[:count]


def source_manifest(data):
    if (data / "revision.txt").read_text().strip() != REVISION:
        raise ValueError("Dataset revision mismatch")
    repositories = sorted((data / "Procedural/MemGovern").iterdir())
    manifest = {
        "dataset": "KaLM-Embedding/LMEB",
        "revision": REVISION,
        "repositories": {},
    }
    for repo in repositories:
        corpus = read_items(repo / "corpus.jsonl")
        queries = read_items(repo / "queries.jsonl")
        doc_ids = {r["id"] for r in corpus}
        query_ids = {r["id"] for r in queries}
        qrels = {}
        for line in (repo / "qrels.tsv").read_text().splitlines():
            query, doc, grade = line.split("\t")
            if query not in query_ids or doc not in doc_ids or int(grade) != 1:
                raise ValueError(f"Invalid relevance association: {repo.name}")
            if query in qrels:
                raise ValueError("Expected exactly one positive per query")
            qrels[query] = doc
        if set(qrels) != query_ids:
            raise ValueError("Unlabeled source queries")
        manifest["repositories"][repo.name] = {
            "documents": len(corpus),
            "queries": len(queries),
            "files": {
                name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv")
            },
        }
    if len(repositories) != 48:
        raise ValueError("Expected all 48 repository corpora")
    return manifest


def initialize_model(name, device, threads):
    global _MODEL, _CONFIG
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _CONFIG = MODELS[name]
    if name == "voyage-1024":
        from transformers import Qwen3Config
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        cls = get_class_from_dynamic_module(
            "modeling_qwen3_bidirectional.Qwen3BidirectionalModel",
            _CONFIG["model"],
            revision=_CONFIG["revision"],
            local_files_only=True,
        )
        cls.config_class = Qwen3Config
    _MODEL = SentenceTransformer(
        _CONFIG["model"],
        revision=_CONFIG["revision"],
        device=device,
        trust_remote_code=_CONFIG["trust_remote_code"],
        local_files_only=True,
        model_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
    )


def encode_chunk(task):
    offset, texts, kind, batch_size = task
    result = encode(_MODEL, texts, _CONFIG, kind, batch_size)
    return offset, np.asarray(result, dtype=np.float32)


def validate_vectors(vectors, count):
    if vectors.ndim != 2 or len(vectors) != count or vectors.dtype != np.float32:
        raise ValueError("Embedding shape/dtype mismatch")
    if not np.isfinite(vectors).all() or not np.allclose(
        np.linalg.norm(vectors, axis=1), 1, atol=2e-5
    ):
        raise ValueError("Nonfinite or unnormalized embeddings")


def prepare(args):
    manifest = source_manifest(args.data)
    source_hash = digest(manifest)
    manifest_path = args.work / "source.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Source changed since preparation")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)
    metadata = {
        "source_sha256": source_hash,
        "config": MODELS[args.embedding],
        "selection": {
            "count_per_repository": args.count,
            "seed": args.seed,
            "algorithm": "lowest SHA256(seed NUL repository NUL query_id)",
        },
        "dtype": "float32",
        "attention": "sdpa",
        "normalize": True,
        "device": args.device,
        "workers": args.workers,
        "threads_per_worker": args.threads,
        "batch_size": args.batch_size,
        "versions": versions(),
        "preparation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    fingerprint = digest(metadata)
    dest = args.work / args.embedding / "encodings"
    dest.mkdir(parents=True, exist_ok=True)
    meta_path = dest / "metadata.json"
    if meta_path.exists() and read_json(meta_path) != metadata:
        raise ValueError("Changed encoding configuration; refusing stale checkpoint")
    write_json(meta_path, metadata)
    executor = None
    if args.device == "cpu":
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_model,
            initargs=(args.embedding, "cpu", args.threads),
        )
    else:
        initialize_model(args.embedding, args.device, args.threads)
    try:
        for repo in sorted(manifest["repositories"]):
            base = args.data / "Procedural/MemGovern" / repo
            corpus = read_items(base / "corpus.jsonl")
            queries = selected_queries(
                read_items(base / "queries.jsonl"), repo, args.count, args.seed
            )
            for kind, items in (("document", corpus), ("query", queries)):
                path = dest / f"{repo}.{kind}.npz"
                ids = [r["id"] for r in items]
                if path.exists():
                    with np.load(path, allow_pickle=False) as checkpoint:
                        if (
                            checkpoint["fingerprint"].item() != fingerprint
                            or checkpoint["ids"].tolist() != ids
                        ):
                            raise ValueError("Stale embedding checkpoint")
                        validate_vectors(checkpoint["vectors"], len(ids))
                    continue
                start = time.perf_counter()
                tasks = [
                    (i, [r["text"] for r in items[i : i + 256]], kind, args.batch_size)
                    for i in range(0, len(items), 256)
                ]
                results = (
                    executor.map(encode_chunk, tasks)
                    if executor
                    else map(encode_chunk, tasks)
                )
                vectors = None
                for offset, block in results:
                    if vectors is None:
                        vectors = np.empty(
                            (len(items), block.shape[1]), dtype=np.float32
                        )
                    vectors[offset : offset + len(block)] = block
                validate_vectors(vectors, len(items))
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as f:
                    np.savez(
                        f,
                        fingerprint=fingerprint,
                        ids=np.array(ids),
                        vectors=vectors,
                        elapsed_s=time.perf_counter() - start,
                    )
                os.replace(temporary, path)
                print(
                    f"READY {args.embedding} {repo} {kind} {len(ids)} {time.perf_counter() - start:.1f}s",
                    flush=True,
                )
        print(
            f"COMPLETE {args.embedding} {len(manifest['repositories'])} repositories",
            flush=True,
        )
    finally:
        if executor:
            executor.shutdown(wait=True, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding", choices=MODEL_NAMES, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
