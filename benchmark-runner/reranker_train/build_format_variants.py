#!/usr/bin/env python3
"""
Build three formatting variants of the stage-2 data, and rescore each with the
teacher.

WHY THIS EXISTS

Hindsight wraps every document before scoring it
(`engine/search/reranking.py`), and **both prefixes are conditional**:

    doc = text
    if context:        doc = f"{context}: {doc}"
    if occurred_start: doc = f"[Date: {readable} ({iso})] {doc}"

So production already emits four distinct shapes depending on which fields are
populated. Our stage-2 documents carry a different prefix again
(`[Oct 14, 2024 - 7:42 AM] Elena: ...`).

Two questions follow, and they are not the same question:

  1. Does training on the serving format make the model rank BETTER on Hindsight?
  2. Does it make the model WORSE at anything else, i.e. is it overfitting to a
     format that may change?

A two-arm test conflates them. Three arms separate them:

    bare       stage-2 documents unchanged
    exact      every document wrapped in Hindsight's exact format
    augmented  randomly wrapped, mirroring the four shapes production emits,
               with varied date styles and context labels

Scored on the benchmarks we already have: fact and chunk MRR measure Hindsight
excellence (they run through the real recall path, so the real format is already
applied), and BEIR NDCG@10 measures general ability on bare text.

The decision rule is deliberately asymmetric, because the goal is "excellent on
Hindsight, no general regression", not "equally good at both":

  * exact lifts MRR, BEIR flat        -> ship exact
  * exact lifts MRR, BEIR drops       -> ship augmented if it keeps the lift
  * nothing moves MRR                 -> ship bare, and gain format independence

RESCORING IS NOT OPTIONAL

Labels are the teacher's margin on the text the student will see. Wrapping the
documents changes that text, so a label computed on unwrapped text is a label for
a different input. Each variant is rescored with the same teacher.

  python build_format_variants.py --src data/stage2.jsonl --out-dir data_fmt \
      --teacher BAAI/bge-reranker-v2-m3
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Plausible values for the fields Hindsight injects. The point is the SHAPE of
# the prefix, not the accuracy of any particular date, so these are synthesised
# deterministically per row rather than mined from the document text.
CONTEXT_LABELS = [
    "conversation", "chat between user and assistant", "project discussion",
    "message thread", "meeting notes", "support conversation",
    "team channel", "direct messages",
]


def hindsight_format(text: str, when: datetime, context: str | None) -> str:
    """Reproduce engine/search/reranking.py exactly."""
    doc = text
    if context:
        doc = f"{context}: {doc}"
    doc = f"[Date: {when.strftime('%B %d, %Y')} ({when.strftime('%Y-%m-%d')})] {doc}"
    return doc


def augmented_format(text: str, when: datetime, context: str | None, rng: random.Random) -> str:
    """Mirror the four shapes production actually emits, plus style variation.

    Production applies each prefix only when its field is populated, so a model
    trained on one fixed shape has still never seen the other three. Varying the
    date rendering and the context wording on top makes the model robust to the
    format changing rather than to one spelling of it.
    """
    doc = text
    use_ctx = rng.random() < 0.7
    use_date = rng.random() < 0.8
    if use_ctx and context:
        sep = rng.choice([": ", " - ", " | "])
        doc = f"{context}{sep}{doc}"
    if use_date:
        style = rng.randrange(4)
        if style == 0:
            doc = f"[Date: {when.strftime('%B %d, %Y')} ({when.strftime('%Y-%m-%d')})] {doc}"
        elif style == 1:
            doc = f"[{when.strftime('%b %d, %Y')}] {doc}"
        elif style == 2:
            doc = f"[Date: {when.strftime('%Y-%m-%d')}] {doc}"
        else:
            doc = f"({when.strftime('%d %B %Y')}) {doc}"
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="stage2.jsonl to reformat")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--teacher", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.src).open()]
    print(f"{len(rows)} source rows", flush=True)

    out = Path(args.out_dir)
    base = datetime(2024, 1, 1)

    variants: dict[str, list[dict]] = {"bare": [], "exact": [], "augmented": []}
    rng = random.Random(args.seed)
    for i, r in enumerate(rows):
        # Deterministic per row so the three variants describe the same document
        # with the same metadata, and only the rendering differs.
        when = base + timedelta(days=(i * 7919) % 900, hours=(i * 13) % 24)
        ctx = CONTEXT_LABELS[i % len(CONTEXT_LABELS)]
        variants["bare"].append({**r})
        variants["exact"].append({**r,
            "positive": hindsight_format(r["positive"], when, ctx),
            "negative": hindsight_format(r["negative"], when, ctx)})
        variants["augmented"].append({**r,
            "positive": augmented_format(r["positive"], when, ctx, rng),
            "negative": augmented_format(r["negative"], when, ctx, rng)})

    from sentence_transformers import CrossEncoder
    import torch

    # activation_fn=Identity is load-bearing. sentence-transformers applies a
    # sigmoid by default for single-label models, which compresses the teacher's
    # margins roughly 35x and destroys the resolution MarginMSE trains on. This
    # exact mistake produced a materially worse model earlier in the project.
    teacher = CrossEncoder(args.teacher, max_length=args.max_length,
                           device="cuda" if torch.cuda.is_available() else "cpu",
                           activation_fn=torch.nn.Identity())

    for name, rs in variants.items():
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        if name != "bare":
            print(f"rescoring {name} with {args.teacher} ...", flush=True)
            pos = teacher.predict([(r["query"], r["positive"]) for r in rs],
                                  batch_size=args.batch, show_progress_bar=True)
            neg = teacher.predict([(r["query"], r["negative"]) for r in rs],
                                  batch_size=args.batch, show_progress_bar=True)
            for r, p, n in zip(rs, pos, neg):
                r["label"] = float(p) - float(n)
            margins = [r["label"] for r in rs]
            print(f"  margin mean {sum(margins)/len(margins):+.3f} "
                  f"min {min(margins):+.3f} max {max(margins):+.3f}", flush=True)
        with (d / "stage2.jsonl").open("w") as fh:
            for r in rs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {d/'stage2.jsonl'} ({len(rs)} rows)", flush=True)
        print(f"  sample: {rs[0]['positive'][:110]}", flush=True)

    print("\nNo stage1.jsonl is written: each arm resumes from a shared "
          "after_stage1 checkpoint, so only stage 2 differs and the comparison "
          "isolates formatting.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
