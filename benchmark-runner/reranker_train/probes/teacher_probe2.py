"""Follow-up controls on bge-reranker-v2-m3.

Probe 1 found: with a "now/currently" query the teacher puts the NEWER correct fact first in
only 7/20 cases, mean margin -0.399 logits, against a relevant-vs-irrelevant scale of +13.6.

Two questions that decides what that means:

E. DIRECTION. Ask the opposite question ("what did X do first / originally"), where the OLDER
   document is correct. If the teacher scores near chance in BOTH directions it is temporally
   indifferent. If it prefers older in both, it has a genuine old-document bias (FRESCO's claim).

F. LENGTH CONFOUND. FRESCO attributes the bias to "semantically richer" documents. Pad the
   NEWER document with extra true detail so it is the longer/richer one, and see whether the
   preference follows length rather than date.

G. EXPLICIT RECENCY WORDING. Does the query word matter? Compare "now", "currently",
   "most recent", "latest", and no marker at all.
"""

import json
import statistics
from datetime import date

import torch
from sentence_transformers import CrossEncoder

MODEL = "BAAI/bge-reranker-v2-m3"

SCENARIOS = [
    ("Priya", "lives in the Fremont apartment on Bryant Street",
     "lives in the Ballard apartment on 24th Ave", "after the lease ended", "live"),
    ("Marcus", "works as a backend engineer at Zenith Logistics",
     "works as a staff engineer at Corvid Systems", "after accepting the offer", "work"),
    ("Elena", "drives a 2016 Subaru Outback",
     "drives a 2024 Kia EV6", "after selling the Subaru", "drive"),
    ("Tomas", "is training for a half marathon",
     "is training for a full marathon", "after finishing the half in April", "train for"),
    ("Anika", "uses Postgres with pgvector for the search index",
     "uses Qdrant for the search index", "after the migration", "use for search"),
    ("Devon", "has a standing Tuesday therapy appointment",
     "has a standing Thursday therapy appointment", "after the schedule change", "schedule"),
    ("Rosa", "is allergic to penicillin only",
     "is allergic to penicillin and sulfa drugs", "after the reaction in clinic", "react to"),
    ("Kenji", "manages the payments team of four engineers",
     "manages the platform team of eleven engineers", "after the reorg", "manage"),
    ("Farida", "pays 1,450 dollars a month in rent",
     "pays 1,780 dollars a month in rent", "after the renewal", "pay in rent"),
    ("Owen", "keeps the deploy pipeline on Jenkins",
     "keeps the deploy pipeline on GitHub Actions", "after retiring the Jenkins box", "use for CI"),
    ("Lucia", "is reading a biography of Ada Lovelace",
     "is reading a novel by Tove Jansson", "after finishing the biography", "read"),
    ("Bram", "stores session state in Redis",
     "stores session state in the Postgres table directly", "after the Redis outage", "store state in"),
    ("Nadia", "has a cat named Pepper",
     "has a cat named Pepper and a dog named Juno", "after adopting the dog", "keep as a pet"),
    ("Ilya", "runs the nightly job at 02:00 UTC",
     "runs the nightly job at 05:30 UTC", "after the timezone complaint", "run the job"),
    ("Sofia", "commutes by bus from Ravenna",
     "commutes by bike from Wallingford", "after moving", "commute"),
    ("Hector", "uses a 27-inch monitor at the desk",
     "uses two 24-inch monitors at the desk", "after the equipment refresh", "use at the desk"),
    ("Yuki", "is on the 15 milligram dose",
     "is on the 30 milligram dose", "after the titration", "take as a dose"),
    ("Colin", "handles billing questions for the EU region",
     "handles billing questions for all regions", "after the coverage change", "cover"),
    ("Amara", "flies out of Midway for work trips",
     "flies out of O'Hare for work trips", "after the airline switch", "fly from"),
    ("Peter", "keeps the API contract at version 2",
     "keeps the API contract at version 4", "after the breaking change", "run as API version"),
]

OLD_DATES = [date(2023, 3, 10), date(2023, 7, 22), date(2024, 1, 5), date(2024, 4, 18)]
NEW_DATES = [date(2025, 9, 2), date(2025, 11, 14), date(2026, 2, 27), date(2026, 5, 9)]

PAD = (" The change was reviewed with the team, written up in the shared notes, and confirmed "
       "again the following week by two other people who were present at the time.")


def render(d: date, what: str, who: str, why: str) -> str:
    text = f"{what} | When: {d.strftime('%B %Y')} | Involving: {who} | {why}"
    return f"[Date: {d.strftime('%B %d, %Y')} ({d.isoformat()})] {text}"


def frac(deltas):
    return sum(1 for x in deltas if x > 0), len(deltas), statistics.mean(deltas), statistics.median(deltas)


def main() -> None:
    print(f"loading {MODEL} ...", flush=True)
    ce = CrossEncoder(MODEL, max_length=512, device="cpu", activation_fn=torch.nn.Identity())

    MARKERS = ["now", "currently", "most recently", "these days", ""]
    pairs = []
    for i, (who, w_old, w_new, why, verb) in enumerate(SCENARIOS):
        d_old, d_new = OLD_DATES[i % 4], NEW_DATES[i % 4]
        doc_old = render(d_old, f"{who} {w_old}", who, why)
        doc_new = render(d_new, f"{who} {w_new}", who, why)
        doc_new_pad = render(d_new, f"{who} {w_new}", who, why + PAD)
        doc_old_pad = render(d_old, f"{who} {w_old}", who, why + PAD)

        # G: five recency markers, correct answer is NEW
        for m in MARKERS:
            q = f"what does {who} {verb} {m}".strip()
            pairs += [(q, doc_old), (q, doc_new)]
        # E: reversed direction, correct answer is OLD
        q_first = f"what did {who} {verb} originally, before the change"
        pairs += [(q_first, doc_old), (q_first, doc_new)]
        # F: length confound. "now" query; NEW is padded (longer), then OLD is padded.
        q_now = f"what does {who} {verb} now"
        pairs += [(q_now, doc_old), (q_now, doc_new_pad)]
        pairs += [(q_now, doc_old_pad), (q_now, doc_new)]

    print(f"scoring {len(pairs)} pairs ...", flush=True)
    sc = list(map(float, ce.predict(pairs, batch_size=16, show_progress_bar=False)))

    per = len(MARKERS) * 2 + 2 + 4
    G = {m: [] for m in MARKERS}
    E, F_newpad, F_oldpad = [], [], []
    for i in range(len(SCENARIOS)):
        b = i * per
        k = b
        for m in MARKERS:
            G[m].append(sc[k + 1] - sc[k])   # new - old, positive is correct
            k += 2
        E.append(sc[k] - sc[k + 1])          # old - new, positive is correct
        k += 2
        F_newpad.append(sc[k + 1] - sc[k])   # padded-new - old
        k += 2
        F_oldpad.append(sc[k + 1] - sc[k])   # new - padded-old

    out = {}
    print("\n=== G. RECENCY MARKER: does the query wording rescue it? (correct = NEWER) ===")
    for m in MARKERS:
        w, n, mu, md = frac(G[m])
        label = m if m else "(no marker)"
        print(f"  {label:14s}: newer wins {w:2d}/{n} ({100*w/n:3.0f}%)  mean {mu:+.3f}  median {md:+.3f}")
        out[f"G_{label}"] = {"wins": w, "n": n, "mean": mu}

    print("\n=== E. REVERSED DIRECTION: 'originally, before the change' (correct = OLDER) ===")
    w, n, mu, md = frac(E)
    print(f"  older wins {w}/{n} ({100*w/n:.0f}%)  mean {mu:+.3f}  median {md:+.3f}")
    out["E"] = {"wins": w, "n": n, "mean": mu}

    print("\n=== F. LENGTH CONFOUND on a 'now' query (correct = NEWER in both rows) ===")
    w, n, mu, md = frac(F_newpad)
    print(f"  newer padded longer : newer wins {w}/{n} ({100*w/n:3.0f}%)  mean {mu:+.3f}")
    out["F_newpad"] = {"wins": w, "n": n, "mean": mu}
    w2, n2, mu2, md2 = frac(F_oldpad)
    print(f"  older padded longer : newer wins {w2}/{n2} ({100*w2/n2:3.0f}%)  mean {mu2:+.3f}")
    out["F_oldpad"] = {"wins": w2, "n": n2, "mean": mu2}
    print(f"  swing attributable to padding: {mu - mu2:+.3f} logits")
    out["F_padding_swing"] = mu - mu2

    with open("teacher_probe2.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote teacher_probe2.json")


if __name__ == "__main__":
    main()
