"""
Convert BEAM (Mohammadta/BEAM, ICLR 2026) conversations into the quality
benchmark's dataset format.

The 100K split holds 20 conversations, each with three to five dated sessions
and 20 probing questions (2 per ability across 10 abilities). We freeze a 4-conversation
subset chosen for domain diversity; every conversation carries the same ability
mix, so domain is the only axis worth spreading.

Run:
    uv run --with pyarrow python -m hindsight_benchmark.beam_convert

Writes datasets/beam_128k_subset.json. The output is committed so every model
runs identical data; re-running the converter must be a no-op diff.
"""

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets"
OUTPUT_PATH = DATASETS_DIR / "beam_128k_subset.json"

PARQUET_URL = (
    "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/"
    "data/100K-00000-of-00001.parquet"
)

# Domain-diverse picks from the 20-conversation 100K split:
# 1 = Coding, 13 = Asking Recommendation, 16 = Lifestyle, 18 = Therapy and
# Emotional Support. Frozen; changing this set invalidates all quality results.
SELECTED_CONVERSATION_IDS = ["1", "13", "16", "18"]

# Gold answer field per ability. instruction_following, preference_following,
# and summarization carry only a rubric; their qa entries have answer=None and
# the judge grades against the rubric points.
ANSWER_FIELDS = ("answer", "ideal_answer", "ideal_response")


def parse_time_anchor(anchor: str) -> datetime:
    """Parse BEAM's 'March-15-2024' anchors to a UTC datetime."""
    return datetime.strptime(anchor, "%B-%d-%Y").replace(tzinfo=timezone.utc)


def _session_date(session: list[dict], previous: datetime | None) -> datetime:
    for msg in session:
        anchor = msg.get("time_anchor")
        if anchor and anchor != "None":
            return parse_time_anchor(anchor)
    if previous is None:
        raise ValueError("First session has no time anchor")
    return previous + timedelta(days=1)


def _parse_rubric(raw) -> list[str]:
    """Rubrics arrive as a stringified Python list; keep the raw text if not."""
    if isinstance(raw, list):
        return [str(r) for r in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(r) for r in parsed]
        except (ValueError, SyntaxError):
            pass
        return [raw]
    return []


def convert_conversation(row: dict) -> dict:
    """Convert one BEAM parquet row to the quality dataset format."""
    sessions = []
    previous_date = None
    for session in row["chat"]:
        date = _session_date(session, previous_date)
        previous_date = date
        sessions.append({
            "date_time": date.isoformat(),
            "messages": [
                {"role": m["role"], "content": m["content"]} for m in session
            ],
        })

    probing = row["probing_questions"]
    if isinstance(probing, str):
        probing = ast.literal_eval(probing)

    qa = []
    for ability in sorted(probing):
        for q in probing[ability]:
            answer = next(
                (q[f] for f in ANSWER_FIELDS if q.get(f)), None
            )
            qa.append({
                "question": q["question"],
                "answer": answer,
                "rubric": _parse_rubric(q.get("rubric")),
                "ability": ability,
            })

    seed = row["conversation_seed"]
    return {
        "sample_id": f"beam-100k-{row['conversation_id']}",
        "category": seed["category"],
        "title": seed["title"],
        "sessions": sessions,
        "qa": qa,
    }


def convert_rows(rows: list[dict]) -> list[dict]:
    by_id = {r["conversation_id"]: r for r in rows}
    missing = [cid for cid in SELECTED_CONVERSATION_IDS if cid not in by_id]
    if missing:
        raise ValueError(f"Selected conversations not in parquet: {missing}")
    return [convert_conversation(by_id[cid]) for cid in SELECTED_CONVERSATION_IDS]


def main():
    import io
    import urllib.request

    import pyarrow.parquet as pq

    print(f"Downloading {PARQUET_URL} ...")
    with urllib.request.urlopen(PARQUET_URL) as resp:
        data = resp.read()
    print(f"  {len(data) / 1024 / 1024:.1f} MB")

    table = pq.read_table(io.BytesIO(data))
    rows = table.to_pylist()
    converted = convert_rows(rows)

    for conv in converted:
        n_msgs = sum(len(s["messages"]) for s in conv["sessions"])
        n_chars = sum(len(m["content"]) for s in conv["sessions"] for m in s["messages"])
        print(
            f"  {conv['sample_id']} [{conv['category']}]: "
            f"{len(conv['sessions'])} sessions, {n_msgs} messages, "
            f"{n_chars} chars, {len(conv['qa'])} questions"
        )

    OUTPUT_PATH.write_text(json.dumps(converted, indent=2, ensure_ascii=False))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
