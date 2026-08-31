from datetime import datetime, timezone

import pytest

from hindsight_benchmark.beam_convert import (
    convert_conversation,
    convert_rows,
    parse_time_anchor,
)


def _row(conversation_id="1"):
    return {
        "conversation_id": conversation_id,
        "conversation_seed": {"category": "Coding", "title": "Budget tracker"},
        "chat": [
            [
                {"role": "user", "content": "hello", "time_anchor": "March-15-2024"},
                {"role": "assistant", "content": "hi", "time_anchor": "None"},
            ],
            [
                # No anchor: falls back to previous session date + 1 day
                {"role": "user", "content": "next", "time_anchor": "None"},
            ],
        ],
        "probing_questions": str({
            "information_extraction": [
                {"question": "What did I build?", "answer": "A tracker",
                 "rubric": "['tracker']"},
            ],
            "summarization": [
                {"question": "Summarize the project.",
                 "rubric": "['budget tracker', 'Flask']"},
            ],
            "abstention": [
                {"question": "What color was it?",
                 "ideal_response": "No information about color.",
                 "rubric": "not-a-list"},
            ],
        }),
    }


def test_parse_time_anchor():
    assert parse_time_anchor("March-15-2024") == datetime(2024, 3, 15, tzinfo=timezone.utc)


def test_convert_conversation_sessions_and_dates():
    conv = convert_conversation(_row())
    assert conv["sample_id"] == "beam-100k-1"
    assert conv["category"] == "Coding"
    assert [s["date_time"] for s in conv["sessions"]] == [
        "2024-03-15T00:00:00+00:00",
        "2024-03-16T00:00:00+00:00",
    ]
    assert conv["sessions"][0]["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_convert_conversation_qa_answer_fields():
    qa = {q["ability"]: q for q in convert_conversation(_row())["qa"]}
    # 'answer' field used directly
    assert qa["information_extraction"]["answer"] == "A tracker"
    assert qa["information_extraction"]["rubric"] == ["tracker"]
    # rubric-only ability: answer is None, rubric parsed
    assert qa["summarization"]["answer"] is None
    assert qa["summarization"]["rubric"] == ["budget tracker", "Flask"]
    # 'ideal_response' mapped to answer; unparseable rubric kept as raw text
    assert qa["abstention"]["answer"] == "No information about color."
    assert qa["abstention"]["rubric"] == ["not-a-list"]


def test_convert_rows_selection_order_and_missing():
    rows = [_row(cid) for cid in ("18", "16", "13", "1", "2")]
    converted = convert_rows(rows)
    assert [c["sample_id"] for c in converted] == [
        "beam-100k-1", "beam-100k-13", "beam-100k-16", "beam-100k-18",
    ]
    with pytest.raises(ValueError, match="not in parquet"):
        convert_rows([_row("1")])
