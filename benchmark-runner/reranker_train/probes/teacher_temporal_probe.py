"""Does bge-reranker-v2-m3 use the [Date: ...] prefix, and does it prefer newer over older?

Three constructions, all rendered in Hindsight's exact serving format
    [Date: {Month DD, YYYY} ({YYYY-MM-DD})] {text}
with `text` in the concise pipe genre  `{what} | When: {when} | Involving: {who}`.

A. KNOWLEDGE UPDATE. Two documents, same entity, contradictory state, different dates.
   Query asks for the current state. Correct answer is the NEWER document.
   Reports: fraction where score(new) > score(old), and the mean margin.

B. DATE ANCHORED. One document text, two renderings differing ONLY in the date prefix
   and the `When:` field. Query names one of the two dates. Correct answer is the
   matching render. Isolates date sensitivity from content.

C. DATE SWAP. Construction A with the two date prefixes exchanged, content untouched.
   A date-aware model flips its preference. A date-blind model does not move.
   Reports: mean |delta| between A and C on the same content.

Run:  probe/bin/python teacher_temporal_probe.py
"""

import itertools
import json
import statistics
from datetime import date

import torch
from sentence_transformers import CrossEncoder

MODEL = "BAAI/bge-reranker-v2-m3"

# (who, what_old, what_new, why, current_query, short_query)
SCENARIOS = [
    ("Priya", "lives in the Fremont apartment on Bryant Street",
     "lives in the Ballard apartment on 24th Ave", "after the lease ended",
     "where does Priya live now", "Priya address"),
    ("Marcus", "works as a backend engineer at Zenith Logistics",
     "works as a staff engineer at Corvid Systems", "after accepting the offer",
     "where does Marcus work currently", "Marcus employer"),
    ("Elena", "drives a 2016 Subaru Outback",
     "drives a 2024 Kia EV6", "after selling the Subaru",
     "what car does Elena drive now", "Elena car"),
    ("Tomas", "is training for a half marathon",
     "is training for a full marathon", "after finishing the half in April",
     "what race is Tomas training for now", "Tomas race"),
    ("Anika", "uses Postgres with pgvector for the search index",
     "uses Qdrant for the search index", "after the migration",
     "what vector store does Anika use now", "Anika vector store"),
    ("Devon", "has a standing Tuesday therapy appointment",
     "has a standing Thursday therapy appointment", "after the schedule change",
     "when is Devon's therapy appointment now", "Devon therapy day"),
    ("Rosa", "is allergic to penicillin only",
     "is allergic to penicillin and sulfa drugs", "after the reaction in clinic",
     "what is Rosa allergic to currently", "Rosa allergies"),
    ("Kenji", "manages the payments team of four engineers",
     "manages the platform team of eleven engineers", "after the reorg",
     "what team does Kenji manage now", "Kenji team"),
    ("Farida", "pays 1,450 dollars a month in rent",
     "pays 1,780 dollars a month in rent", "after the renewal",
     "what is Farida's current rent", "Farida rent"),
    ("Owen", "keeps the deploy pipeline on Jenkins",
     "keeps the deploy pipeline on GitHub Actions", "after retiring the Jenkins box",
     "what CI does Owen use now", "Owen CI"),
    ("Lucia", "is reading a biography of Ada Lovelace",
     "is reading a novel by Tove Jansson", "after finishing the biography",
     "what is Lucia reading now", "Lucia reading"),
    ("Bram", "stores session state in Redis",
     "stores session state in the Postgres table directly", "after the Redis outage",
     "where does Bram store session state now", "Bram session state"),
    ("Nadia", "has a cat named Pepper",
     "has a cat named Pepper and a dog named Juno", "after adopting the dog",
     "what pets does Nadia have now", "Nadia pets"),
    ("Ilya", "runs the nightly job at 02:00 UTC",
     "runs the nightly job at 05:30 UTC", "after the timezone complaint",
     "when does Ilya's nightly job run now", "Ilya job schedule"),
    ("Sofia", "commutes by bus from Ravenna",
     "commutes by bike from Wallingford", "after moving",
     "how does Sofia commute now", "Sofia commute"),
    ("Hector", "uses a 27-inch monitor at the desk",
     "uses two 24-inch monitors at the desk", "after the equipment refresh",
     "what monitors does Hector use now", "Hector monitors"),
    ("Yuki", "is on the 15 milligram dose",
     "is on the 30 milligram dose", "after the titration",
     "what dose is Yuki on now", "Yuki dose"),
    ("Colin", "handles billing questions for the EU region",
     "handles billing questions for all regions", "after the coverage change",
     "what region does Colin cover now", "Colin region"),
    ("Amara", "flies out of Midway for work trips",
     "flies out of O'Hare for work trips", "after the airline switch",
     "which airport does Amara fly from now", "Amara airport"),
    ("Peter", "keeps the API contract at version 2",
     "keeps the API contract at version 4", "after the breaking change",
     "what API version is Peter on now", "Peter API version"),
]

OLD_DATES = [date(2023, 3, 10), date(2023, 7, 22), date(2024, 1, 5), date(2024, 4, 18)]
NEW_DATES = [date(2025, 9, 2), date(2025, 11, 14), date(2026, 2, 27), date(2026, 5, 9)]


def render(d: date, what: str, who: str, why: str) -> str:
    when = d.strftime("%B %Y")
    text = f"{what} | When: {when} | Involving: {who} | {why}"
    return f"[Date: {d.strftime('%B %d, %Y')} ({d.isoformat()})] {text}"


def main() -> None:
    print(f"loading {MODEL} on CPU ...", flush=True)
    ce = CrossEncoder(MODEL, max_length=512, device="cpu",
                      activation_fn=torch.nn.Identity())  # RAW LOGITS, not sigmoid

    pairs, meta = [], []

    for i, (who, w_old, w_new, why, q_cur, q_short) in enumerate(SCENARIOS):
        d_old = OLD_DATES[i % len(OLD_DATES)]
        d_new = NEW_DATES[i % len(NEW_DATES)]

        doc_old = render(d_old, f"{who} {w_old}", who, why)
        doc_new = render(d_new, f"{who} {w_new}", who, why)
        # C: same content, dates exchanged
        doc_old_swap = render(d_new, f"{who} {w_old}", who, why)
        doc_new_swap = render(d_old, f"{who} {w_new}", who, why)

        for q, qstyle in ((q_cur, "long"), (q_short, "short")):
            pairs += [(q, doc_old), (q, doc_new), (q, doc_old_swap), (q, doc_new_swap)]
            meta.append({"scen": i, "who": who, "qstyle": qstyle, "kind": "A/C"})

        # B: date-anchored. Same content, only the date differs. Query names one date.
        stable = f"{who} {w_new}"
        d_a, d_b = d_old, d_new
        doc_a = render(d_a, stable, who, why)
        doc_b = render(d_b, stable, who, why)
        q_b = f"what did {who} do in {d_b.strftime('%B %Y')}"
        pairs += [(q_b, doc_a), (q_b, doc_b)]

        # D: relevance control. The long query against an unrelated person's fact,
        # to establish what a genuine relevance margin looks like on this scale.
        j = (i + 7) % len(SCENARIOS)
        o_who, _, o_new, o_why, _, _ = SCENARIOS[j]
        pairs.append((q_cur, render(NEW_DATES[j % len(NEW_DATES)],
                                    f"{o_who} {o_new}", o_who, o_why)))
        meta.append({"scen": i, "who": who, "qstyle": "anchored", "kind": "B",
                     "target": "b", "d_a": d_a.isoformat(), "d_b": d_b.isoformat()})

    print(f"scoring {len(pairs)} pairs ...", flush=True)
    scores = list(map(float, ce.predict(pairs, batch_size=16, show_progress_bar=True)))

    # unpack
    k = 0
    A = {"long": [], "short": []}      # (old, new)
    C = {"long": [], "short": []}      # (old_swap, new_swap)
    B = []                             # (non-matching date, matching date)
    D = []                             # irrelevant control
    for i in range(len(SCENARIOS)):
        for qstyle in ("long", "short"):
            s_old, s_new, s_old_sw, s_new_sw = scores[k:k + 4]
            k += 4
            A[qstyle].append((s_old, s_new))
            C[qstyle].append((s_old_sw, s_new_sw))
        s_a, s_b, s_ctrl = scores[k:k + 3]
        k += 3
        B.append((s_a, s_b))
        D.append(s_ctrl)

    out = {}
    print("\n=== A. KNOWLEDGE UPDATE: does the teacher put the NEWER correct fact first? ===")
    for qstyle in ("long", "short"):
        deltas = [new - old for old, new in A[qstyle]]
        wins = sum(1 for d in deltas if d > 0)
        print(f"  query style {qstyle:5s}: newer wins {wins}/{len(deltas)} "
              f"({100 * wins / len(deltas):.0f}%), mean margin {statistics.mean(deltas):+.3f}, "
              f"median {statistics.median(deltas):+.3f}, "
              f"range [{min(deltas):+.3f}, {max(deltas):+.3f}]")
        out[f"A_{qstyle}"] = {"wins": wins, "n": len(deltas),
                              "mean_margin": statistics.mean(deltas)}

    print("\n=== B. DATE ANCHORED: query names a date; only the date prefix differs ===")
    deltas_b = [b - a for a, b in B]
    wins_b = sum(1 for d in deltas_b if d > 0)
    print(f"  matching date wins {wins_b}/{len(deltas_b)} ({100 * wins_b / len(deltas_b):.0f}%), "
          f"mean margin {statistics.mean(deltas_b):+.3f}, median {statistics.median(deltas_b):+.3f}, "
          f"mean |margin| {statistics.mean(abs(d) for d in deltas_b):.3f}")
    out["B"] = {"wins": wins_b, "n": len(deltas_b), "mean_margin": statistics.mean(deltas_b)}

    print("\n=== C. DATE SWAP: same content, dates exchanged. How far does the score move? ===")
    for qstyle in ("long", "short"):
        moves = []
        for (o, n), (osw, nsw) in zip(A[qstyle], C[qstyle]):
            moves.append(abs(o - osw))
            moves.append(abs(n - nsw))
        flips = sum(1 for (o, n), (osw, nsw) in zip(A[qstyle], C[qstyle])
                    if (n > o) != (nsw > osw))
        print(f"  query style {qstyle:5s}: mean |score shift| when only the date changes "
              f"{statistics.mean(moves):.3f} (median {statistics.median(moves):.3f}, "
              f"max {max(moves):.3f}); preference flipped in {flips}/{len(A[qstyle])} scenarios")
        out[f"C_{qstyle}"] = {"mean_abs_shift": statistics.mean(moves), "flips": flips,
                              "n": len(A[qstyle])}

    # scale reference: how big is a "real" relevance margin for comparison
    print("\n=== D. RELEVANCE CONTROL: same query vs an unrelated person's fact ===")
    ctrl = [statistics.mean(A["long"][i]) - D[i] for i in range(len(D))]
    print(f"  mean(relevant) - irrelevant: mean {statistics.mean(ctrl):+.3f}, "
          f"median {statistics.median(ctrl):+.3f}, min {min(ctrl):+.3f}")
    out["D_relevance_margin"] = statistics.mean(ctrl)

    print("\n=== SCALE REFERENCE ===")
    all_scores = [s for s in scores]
    print(f"  all {len(all_scores)} scores: min {min(all_scores):+.3f}, "
          f"median {statistics.median(all_scores):+.3f}, max {max(all_scores):+.3f}")
    print("  For comparison, our stage-2 training margins have median 7.83 logits.")

    with open("teacher_temporal_probe.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote teacher_temporal_probe.json")


if __name__ == "__main__":
    main()
