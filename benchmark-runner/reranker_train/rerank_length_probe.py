#!/usr/bin/env python3
"""
Measure how long the documents Hindsight actually reranks really are.

Hindsight builds each scored document as "[Date: ...] {context}: {text}"
(engine/search/reranking.py). Nobody has measured that string's length. Our perf
profiles guess 45 tokens for an extracted fact and 380 for a 3000-char chunk,
and every latency number we report is anchored to those guesses. Reading the
real distribution off the wire settles it.

This is a pass-through proxy: Hindsight points at it instead of TEI, it forwards
every request unchanged, and it tokenizes the documents on the way past. It adds
one local hop to a benchmark that is already bound by LLM calls.

The tokenizer must be the one the model under test uses. Token counts are not
portable across tokenizers, so a bge count and an mmBERT count are different
numbers for the same string.

  python rerank_length_probe.py --upstream http://localhost:8081 \
      --port 8097 --out lengths.json --tokenizer BAAI/bge-reranker-base
"""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATE = {
    "doc_tokens": [],       # one entry per scored document
    "docs_per_request": [], # one entry per /rerank call
    "query_tokens": [],
    "requests": 0,
    "forward_errors": 0,
    "tokenizer": None,
    "upstream": None,
}
LOCK = threading.Lock()
TOKENIZER = None
OUT_PATH: Path | None = None


def percentile(xs: list[int], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summary() -> dict:
    with LOCK:
        docs = list(STATE["doc_tokens"])
        per_req = list(STATE["docs_per_request"])
        queries = list(STATE["query_tokens"])
        reqs, errs = STATE["requests"], STATE["forward_errors"]
        tok, up = STATE["tokenizer"], STATE["upstream"]

    out = {
        "tokenizer": tok,
        "upstream": up,
        "requests": reqs,
        "forward_errors": errs,
        "documents_scored": len(docs),
    }
    if docs:
        out["doc_tokens"] = {
            "min": min(docs), "max": max(docs),
            "mean": round(statistics.fmean(docs), 1),
            "median": round(percentile(docs, 0.5), 1),
            "p90": round(percentile(docs, 0.90), 1),
            "p99": round(percentile(docs, 0.99), 1),
            # The comparison that matters: production's bge-reranker-base stops
            # at 512 tokens, so anything above this line is silently truncated.
            "over_512": sum(1 for d in docs if d > 512),
            "over_512_pct": round(100.0 * sum(1 for d in docs if d > 512) / len(docs), 2),
            "histogram": _histogram(docs),
        }
    if per_req:
        out["docs_per_request"] = {
            "min": min(per_req), "max": max(per_req),
            "mean": round(statistics.fmean(per_req), 1),
        }
    if queries:
        out["query_tokens"] = {
            "min": min(queries), "max": max(queries),
            "mean": round(statistics.fmean(queries), 1),
        }
    return out


def _histogram(docs: list[int]) -> dict[str, int]:
    edges = [0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    hist: dict[str, int] = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        hist[f"{lo}-{hi}"] = sum(1 for d in docs if lo <= d < hi)
    hist[f">={edges[-1]}"] = sum(1 for d in docs if d >= edges[-1])
    return hist


def flush() -> None:
    if OUT_PATH is None:
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary(), indent=2))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # silence per-request logging
        pass

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", ""):
            self._send(200, b'{"status":"ok"}')
            return
        if self.path.rstrip("/") == "/stats":
            self._send(200, json.dumps(summary()).encode())
            return
        self._forward(b"", method="GET")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        # Record first, forward second. A tokenizer failure must never break the
        # benchmark it is observing, so measurement is wrapped and swallowed
        # while the forward is not.
        try:
            self._record(body)
        except Exception as exc:
            print(f"probe: record failed ({type(exc).__name__}: {exc})", file=sys.stderr, flush=True)
        self._forward(body, method="POST")

    def _record(self, body: bytes) -> None:
        if "rerank" not in self.path and "predict" not in self.path:
            return
        payload = json.loads(body)
        texts = payload.get("texts") or payload.get("documents") or []
        if not isinstance(texts, list):
            return
        lens = [len(TOKENIZER.encode(t).ids) for t in texts if isinstance(t, str)]
        query = payload.get("query")
        qlen = len(TOKENIZER.encode(query).ids) if isinstance(query, str) else None
        with LOCK:
            STATE["doc_tokens"].extend(lens)
            STATE["docs_per_request"].append(len(lens))
            if qlen is not None:
                STATE["query_tokens"].append(qlen)
            STATE["requests"] += 1
            n = STATE["requests"]
        # Periodic flush so a hard kill still leaves usable data on disk.
        if n % 25 == 0:
            flush()

    def _forward(self, body: bytes, method: str) -> None:
        url = STATE["upstream"].rstrip("/") + self.path
        req = urllib.request.Request(url, data=body if method == "POST" else None, method=method)
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "connection"):
                req.add_header(k, v)
        if method == "POST":
            req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
                self._send(resp.status, data, resp.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as exc:
            # Pass upstream errors through verbatim. Rewriting them would hide a
            # real TEI failure behind a proxy failure.
            self._send(exc.code, exc.read(), exc.headers.get("Content-Type", "application/json"))
        except Exception as exc:
            with LOCK:
                STATE["forward_errors"] += 1
            self._send(502, json.dumps({"error": f"probe upstream: {exc}"}).encode())

    def _send(self, status: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> int:
    global TOKENIZER, OUT_PATH

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", required=True, help="TEI base URL to forward to")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="HF id or local path of the model under test")
    args = ap.parse_args()

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    # Baked-in padding is a real trap: some rerankers ship a tokenizer that pads
    # every input to 512, which would make every measurement here read 512 and
    # look like a truncation crisis. Strip padding and truncation so the count is
    # the string's true length.
    hf = AutoTokenizer.from_pretrained(args.tokenizer)
    TOKENIZER = hf.backend_tokenizer if hasattr(hf, "backend_tokenizer") else Tokenizer.from_pretrained(args.tokenizer)
    TOKENIZER.no_padding()
    TOKENIZER.no_truncation()

    probe = len(TOKENIZER.encode("hello").ids)
    if probe > 8:
        raise SystemExit(f"tokenizer still padding: 'hello' encoded to {probe} tokens")

    OUT_PATH = Path(args.out)
    STATE["tokenizer"] = args.tokenizer
    STATE["upstream"] = args.upstream

    def _bye(*_):
        flush()
        print(f"probe: wrote {OUT_PATH}", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _bye)
    signal.signal(signal.SIGINT, _bye)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"probe listening on :{args.port} -> {args.upstream}", flush=True)
    try:
        server.serve_forever()
    finally:
        flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
