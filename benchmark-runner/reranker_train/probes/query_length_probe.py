"""Does the teacher still discriminate when the query is 1-4 words?

Uses the REAL evaluation bank ground truth: 178 LoCoMo questions and the Hindsight-extracted
concise facts annotated as relevant to them. Negatives are the relevant facts of OTHER
questions in the same conversation, which is the production negative profile (same people,
same conversation, lexically overlapping) rather than the cross-topic negatives our current
training data uses.

For each question we score the positive and 24 within-bank negatives under three query forms:
  long    the original question, median 10 words
  medium  content words only, capped at 6
  short   content words only, capped at 3

and report the teacher margin (positive minus best negative) and the rank of the positive.
If the margin collapses on short queries, the teacher is a weaker supervisor exactly where
the reflect agent operates, and our labels inherit that.
"""

import json
import random
import re
import statistics

import torch
from sentence_transformers import CrossEncoder

GT = ("/Users/andrew/Documents/code/hindsight-benchmarks/reranker-artifacts/"
      "bank/locomo_reranker_gt_fact.json")
MODEL = "BAAI/bge-reranker-v2-m3"
N_NEG = 24

STOP = set("""a an the is are was were be been being do does did doing have has had of to in on
at for with about from by as and or but if then than that this these those what which who whom
whose when where why how did i you he she it we they me him her them my your his its our their
there here can could should would will shall may might must not no yes any some all more most
other another such only own same so too very s t just don now been get got go went make made
say said tell told mention mentioned regard regards regarding end up ended""".split())


def keywords(q: str, cap: int) -> str:
    toks = re.findall(r"[A-Za-z0-9'\-]+", q)
    kept = [t for t in toks if t.lower() not in STOP]
    if not kept:
        kept = toks
    return " ".join(kept[:cap])


def main() -> None:
    gt = json.load(open(GT))["annotations"]
    items = [(a["question"], a["relevant_facts"][0]) for a in gt if a.get("relevant_facts")]
    all_facts = sorted({f for a in gt for f in a["relevant_facts"]})
    print(f"{len(items)} questions, {len(all_facts)} distinct extracted facts in the pool")

    rng = random.Random(7)
    forms = {"long": lambda q: q, "medium": lambda q: keywords(q, 6),
             "short": lambda q: keywords(q, 3)}

    print(f"loading {MODEL} ...", flush=True)
    ce = CrossEncoder(MODEL, max_length=512, device="cpu", activation_fn=torch.nn.Identity())

    wl = {k: [] for k in forms}
    pairs, index = [], []
    for qi, (q, pos) in enumerate(items):
        negs = rng.sample([f for f in all_facts if f != pos], N_NEG)
        for fname, fn in forms.items():
            qq = fn(q)
            wl[fname].append(len(qq.split()))
            base = len(pairs)
            pairs.append((qq, pos))
            pairs += [(qq, n) for n in negs]
            index.append((qi, fname, base))

    print("query word counts: " + ", ".join(
        f"{k} median {statistics.median(v):.0f}" for k, v in wl.items()))
    print(f"scoring {len(pairs)} pairs (this takes a few minutes) ...", flush=True)
    sc = list(map(float, ce.predict(pairs, batch_size=32, show_progress_bar=False)))

    res = {k: {"margin": [], "rank1": 0, "n": 0, "mrr": []} for k in forms}
    for qi, fname, base in index:
        block = sc[base:base + 1 + N_NEG]
        p, negs = block[0], block[1:]
        res[fname]["margin"].append(p - max(negs))
        rank = 1 + sum(1 for n in negs if n > p)
        res[fname]["rank1"] += int(rank == 1)
        res[fname]["mrr"].append(1.0 / rank)
        res[fname]["n"] += 1

    print("\n=== teacher discrimination vs query length, real bank facts, "
          f"{N_NEG} within-bank negatives ===")
    print(f"{'form':7s} {'words':>5s}  {'margin(pos - best neg)':>24s}  {'R@1':>7s}  {'MRR':>6s}")
    out = {}
    for k in forms:
        m = res[k]["margin"]
        print(f"{k:7s} {statistics.median(wl[k]):5.0f}  "
              f"mean {statistics.mean(m):+7.3f} median {statistics.median(m):+7.3f}  "
              f"{100*res[k]['rank1']/res[k]['n']:6.1f}%  {statistics.mean(res[k]['mrr']):.4f}")
        out[k] = {"mean_margin": statistics.mean(m),
                  "median_margin": statistics.median(m),
                  "r_at_1": res[k]["rank1"] / res[k]["n"],
                  "mrr": statistics.mean(res[k]["mrr"]),
                  "median_words": statistics.median(wl[k])}

    print("\n  Reference: our stage-2 training margins have median 7.83 logits, against "
          "cross-topic negatives.")
    with open("query_length_probe.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("wrote query_length_probe.json")


if __name__ == "__main__":
    main()
