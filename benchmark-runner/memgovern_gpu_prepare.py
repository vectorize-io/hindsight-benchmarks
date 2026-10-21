"""Resume MemGovern embedding checkpoints on a second GPU with explicit provenance."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from memgovern_prepare import (
    DEFAULT_DATA,
    DEFAULT_WORK,
    MODEL_NAMES,
    encode_chunk,
    initialize_model,
    read_items,
    selected_queries,
    source_manifest,
    validate_vectors,
)
from rerank_eval import digest, read_json, versions

# Hardware/scheduling can change; source, tokenizer/model settings and arithmetic cannot.
SEMANTIC_FIELDS = (
    "source_sha256",
    "config",
    "selection",
    "dtype",
    "attention",
    "normalize",
    "batch_size",
    "versions",
)


def checkpoint_metadata(checkpoint, initial):
    metadata = (
        json.loads(checkpoint["execution_metadata"].item())
        if "execution_metadata" in checkpoint.files
        else initial
    )
    if "execution_metadata" in checkpoint.files and any(
        metadata[key] != initial[key] for key in SEMANTIC_FIELDS
    ):
        raise ValueError("Changed embedding semantics in checkpoint")
    if checkpoint["fingerprint"].item() != digest(metadata):
        raise ValueError("Checkpoint provenance mismatch")
    return metadata


def prepare(args):
    import torch

    source = source_manifest(args.data)
    if source != read_json(args.work / "source.json"):
        raise ValueError("Source changed before GPU continuation")
    for embedding in (args.embedding,):
        dest = args.work / embedding / "encodings"
        initial = read_json(dest / "metadata.json")
        if (
            initial["source_sha256"] != digest(source)
            or initial["versions"] != versions()
        ):
            raise ValueError("Source or inference environment changed")
        initialize_model(embedding, "cuda:0", args.threads)
        metadata = {
            **initial,
            "device": "cuda:0",
            "hardware": torch.cuda.get_device_name(0),
            "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "workers": 1,
            "threads_per_worker": args.threads,
            "preparation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "core_preparation_sha256": hashlib.sha256(
                Path(__file__).with_name("memgovern_prepare.py").read_bytes()
            ).hexdigest(),
            "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
        }
        for repo in sorted(source["repositories"]):
            base = args.data / "Procedural/MemGovern" / repo
            corpus = read_items(base / "corpus.jsonl")
            selection = initial["selection"]
            queries = selected_queries(
                read_items(base / "queries.jsonl"),
                repo,
                selection["count_per_repository"],
                selection["seed"],
            )
            for kind, items in (("document", corpus), ("query", queries)):
                ids = [r["id"] for r in items]
                path = dest / f"{repo}.{kind}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as checkpoint:
                        checkpoint_metadata(checkpoint, initial)
                        if checkpoint["ids"].tolist() != ids:
                            raise ValueError("Checkpoint ID mismatch")
                        validate_vectors(checkpoint["vectors"], len(ids))
                    continue
                start = time.perf_counter()
                blocks = []
                for offset in range(0, len(items), 256):
                    _, block = encode_chunk(
                        (
                            offset,
                            [r["text"] for r in items[offset : offset + 256]],
                            kind,
                            initial["batch_size"],
                        )
                    )
                    blocks.append(block)
                vectors = np.concatenate(blocks, axis=0)
                validate_vectors(vectors, len(ids))
                temp = path.with_suffix(".tmp")
                with temp.open("wb") as stream:
                    np.savez(
                        stream,
                        fingerprint=digest(metadata),
                        execution_metadata=json.dumps(metadata, sort_keys=True),
                        ids=np.array(ids),
                        vectors=vectors,
                        elapsed_s=time.perf_counter() - start,
                    )
                os.replace(temp, path)
                print(
                    f"READY {embedding} {repo} {kind} {len(ids)} {time.perf_counter() - start:.1f}s A4000",
                    flush=True,
                )
        print(f"COMPLETE {embedding} 48 repositories", flush=True)
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--embedding",
        choices=MODEL_NAMES[:3],
        required=True,
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
