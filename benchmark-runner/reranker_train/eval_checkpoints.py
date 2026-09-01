#!/usr/bin/env python3
"""
Score saved checkpoints against the training holdout, after the fact.

The v2 run produced no eval entries: the holdout was built correctly (2,990 rows
over 956 unseen queries) and `eval_strategy=STEPS` with `eval_steps=2500` was
confirmed on the args object, but sentence-transformers 6.0.0 with transformers
5.16.1 logged none. v1's run, on older versions, produced six.

Rather than burn a rerun on a diagnostic, this reconstructs the curve from the
checkpoints the trainer already wrote. Fewer points, same measurement.

THE HOLDOUT MUST MATCH THE ONE TRAINING USED, exactly. It is split by query, not
by row, with a fixed seed. Reproducing it approximately would score partly on
rows the model trained on and report a flatteringly low loss. The split logic
here is a deliberate duplicate of `run_stage`'s, and the row and query counts it
prints must match what the training log reported.

  python eval_checkpoints.py --data ~/train/data_v1/stage1.jsonl \
      --checkpoints ~/train/out6/ckpt_stage1/checkpoint-5000 \
                    ~/train/out6/ckpt_stage1/checkpoint-10000 \
                    ~/train/out6/after_stage1 \
      --maxlen 512 --out ~/train/out6/eval_stage1.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({"query": r["query"], "positive": r["positive"],
                         "negative": r["negative"], "label": float(r["label"])})
    return rows


def holdout(rows: list[dict], eval_rows: int = 3000, seed: int = 1234) -> list[dict]:
    """Reproduce run_stage's group-split holdout exactly.

    Up to 6 rows share a query, so a row-level split would leak the same query
    into both halves and the loss would measure memorisation.
    """
    if not (eval_rows > 0 and len(rows) > eval_rows * 3):
        raise SystemExit(f"training would not have split a {len(rows)}-row dataset")
    queries = sorted({r["query"] for r in rows})
    rgen = random.Random(seed)
    rgen.shuffle(queries)
    n_eval_q = max(1, int(len(queries) * min(0.2, eval_rows / len(rows))))
    held = set(queries[:n_eval_q])
    ev = [r for r in rows if r["query"] in held]
    print(f"holdout: {len(ev)} rows over {len(held)} queries "
          f"(training log must report the same two numbers)", flush=True)
    return ev


def margin_mse(ckpt: Path, ev: list[dict], maxlen: int, batch: int) -> float:
    """Mean squared error between the student's margin and the teacher's.

    This is exactly the training objective, so the number is comparable to the
    loss curve and to v1's eval points.
    """
    from sentence_transformers.cross_encoder import CrossEncoder

    m = CrossEncoder(str(ckpt), num_labels=1, max_length=maxlen, device="cuda",
                     model_kwargs={"attn_implementation": "sdpa"})
    m.model.eval()
    tok, total, n = m.tokenizer, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(ev), batch):
            chunk = ev[i:i + batch]
            q = [r["query"] for r in chunk]
            lab = torch.tensor([r["label"] for r in chunk], device="cuda", dtype=torch.float32)
            scores = []
            for key in ("positive", "negative"):
                enc = tok(q, [r[key] for r in chunk], padding=True, truncation=True,
                          max_length=maxlen, return_tensors="pt").to("cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    scores.append(m.model(**enc).logits.view(-1).float())
            total += torch.nn.functional.mse_loss(
                scores[0] - scores[1], lab, reduction="sum").item()
            n += len(chunk)
    del m
    torch.cuda.empty_cache()
    return total / n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--eval-rows", type=int, default=3000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ev = holdout(load_rows(Path(args.data)), args.eval_rows)

    out: dict[str, float] = {}
    for c in args.checkpoints:
        p = Path(c)
        if not p.exists():
            print(f"  {p.name}: MISSING, skipped", flush=True)
            continue
        t0 = time.time()
        loss = margin_mse(p, ev, args.maxlen, args.batch)
        out[p.name] = round(loss, 4)
        print(f"  {p.name:24s} eval MarginMSE {loss:8.4f}   ({time.time() - t0:.0f}s)", flush=True)

    Path(args.out).write_text(json.dumps(
        {"data": args.data, "maxlen": args.maxlen, "eval_rows": len(ev), "losses": out}, indent=2))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
