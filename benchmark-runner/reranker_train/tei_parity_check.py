#!/usr/bin/env python3
"""
Check a trained checkpoint scores the same through TEI as it does locally.

Two things this catches, both of which would otherwise show up only as a
mysteriously bad MRR:

1. SIGMOID SATURATION. TEI applies a sigmoid to single-label scores unless the
   request sets `raw_scores: true`, and Hindsight's client does not set it.
   MarginMSE constrains only score DIFFERENCES, so adding a constant to all of
   the student's logits leaves the loss unchanged — the absolute operating
   point is free to drift during training. Measured in f32: sigmoid ties at
   1.0 above logit ~+16.6, and reaches exactly 0.0 only below ~-89 (where exp
   overflows). Small scores like sigmoid(-11) = 1.6e-5 are perfectly ordered,
   so only the HIGH side is a realistic tie risk. A student whose logits
   drifted high would rank perfectly in every local raw-logit check and then
   return TIED scores through TEI. The teacher never hits this (its logits sit within about ±11),
   but nothing pins the student there.

2. Any other TEI translation bug — RoPE settings, pooling mode, tokenizer
   mismatch — which would change the ranking silently.

Run this after training and BEFORE trusting any benchmark number.

  python tei_parity_check.py --checkpoint ~/train/out_v1/final --url http://localhost:8081
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import httpx


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(xs: list[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 2:
        return 1.0
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    num = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    den = (sum((x - mean_a) ** 2 for x in ra) * sum((y - mean_b) ** 2 for y in rb)) ** 0.5
    return num / den if den else 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--url", default="http://localhost:8081")
    ap.add_argument("--beam", default=None, help="BEAM subset JSON for realistic text")
    ap.add_argument("--pairs", type=int, default=100)
    args = ap.parse_args()

    import torch
    from sentence_transformers import CrossEncoder

    rng = random.Random(7)
    texts: list[str] = []
    if args.beam:
        with open(args.beam) as fh:
            for sample in json.load(fh):
                for session in sample.get("sessions", []):
                    for m in session.get("messages", []):
                        c = (m.get("content") or "").strip()
                        if len(c) > 300:
                            texts.append(c[:3000])
    if len(texts) < 20:
        texts = [f"Document {i}: we discussed the migration and agreed to postpone it." for i in range(40)]

    queries = [
        "what did we decide about the migration",
        "when are we moving",
        "what did they say about the budget",
        "我们决定什么时候搬家",
        "预算方面他们怎么说",
    ]
    pairs = [(rng.choice(queries), rng.choice(texts)) for _ in range(args.pairs)]

    print(f"Scoring {len(pairs)} pairs locally (raw logits) ...", flush=True)
    model = CrossEncoder(args.checkpoint, max_length=2560,
                         device="cuda" if torch.cuda.is_available() else "cpu")
    local = [float(x) for x in model.predict(pairs, batch_size=16, show_progress_bar=False,
                                             activation_fn=torch.nn.Identity())]

    print(f"Scoring the same pairs through TEI at {args.url} ...", flush=True)
    served: list[float] = []
    with httpx.Client(timeout=120.0) as client:
        for q, t in pairs:
            r = client.post(args.url.rstrip("/") + "/rerank",
                            json={"query": q, "texts": [t], "return_text": False})
            r.raise_for_status()
            served.append(float(r.json()[0]["score"]))

    lo, hi = min(local), max(local)
    rho = spearman(local, served)
    # Count TRUE ties only. An earlier version flagged s <= 1e-4 (logit
    # <= -9.21) as saturated, which is the normal score for an irrelevant pair
    # — the teacher's own irrelevant example sits at -11.02 — so it failed
    # healthy models.
    n_tied_hi = sum(1 for s in served if s >= 1.0 - 1e-7)
    n_tied_lo = sum(1 for s in served if s == 0.0)
    drift_high = hi > 15.0

    print()
    print(f"  local raw logits : min {lo:.3f}  max {hi:.3f}")
    print(f"  TEI scores       : min {min(served):.6f}  max {max(served):.6f}")
    print(f"  rank correlation : {rho:.4f}")
    print(f"  tied at 1.0      : {n_tied_hi}/{len(served)}")
    print(f"  tied at 0.0      : {n_tied_lo}/{len(served)}")
    print(f"  max local logit  : {hi:.3f}  (f32 tie point ~+16.6)")

    ok = True
    if rho < 0.99:
        print("\nFAIL: ranking differs between local and TEI. Something in the TEI "
              "translation (RoPE, pooling, tokenizer) is wrong — the benchmark would be invalid.")
        ok = False
    if n_tied_hi + n_tied_lo > len(served) * 0.02:
        print("\nFAIL: sigmoid saturation — TEI is returning tied scores, so ranking "
              "collapses at serve time even though local raw-logit ranking looks fine. "
              "Subtract a constant from the classifier head bias at export.")
        ok = False
    elif drift_high:
        print(f"\nWARNING: max local logit {hi:.2f} is approaching the +16.6 f32 tie point. "
              f"MarginMSE does not constrain the absolute scale, so this can drift further "
              f"with more training. Watch it, or re-centre the head bias.")
    if ok:
        print("\nPASS: TEI ranking matches local ranking and no saturation.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
