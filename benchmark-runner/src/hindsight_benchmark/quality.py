"""
Quality benchmark — measures answer accuracy via Hindsight memory recall on LoComo conv-43.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

from .benchmark import DATASETS_DIR, LEADERBOARD_DIR

ANSWER_GENERATOR_MODEL = "gemini-2.5-flash-lite"
JUDGE_MODEL = "gemini-2.5-flash-lite"

# ── prompts ───────────────────────────────────────────────────────────────────

CONTEXT_INSTRUCTIONS = """**Understanding the Retrieved Context:**
The context contains memory facts extracted from previous conversations, each with its source chunk.

1. **Fact**: A high-level summary/atomic fact (e.g., "User loves hiking in mountains")
   - This is the searchable summary of what was stored

2. **Source Chunk**: The actual raw conversation where the fact was extracted from
   - **This is your primary source for detailed information**
   - Look here for specifics, context, quotes, and evidence
   - Prioritize information from chunks when facts seem ambiguous

3. **Temporal Information**:
   - "occurred": When the event actually happened
   - "mentioned": When it was discussed in conversation
   - Use this to understand the timeline and resolve conflicts (prefer more recent info)

4. **Context**: Additional metadata about the conversation session

**Date Calculations (CRITICAL - read carefully):**
- When calculating days between two dates: count the days from Date A to Date B as (B - A)
- Example: Jan 1 to Jan 8 = 7 days (not 8)
- "X days ago" from Question Date means: Question Date minus X days
- When a fact says "three weeks ago" on a certain mentioned date, that refers to 3 weeks before THAT mentioned date, NOT the question date
- Always convert relative times ("last Friday", "two weeks ago") to absolute dates BEFORE comparing
- Double-check your arithmetic - off-by-one errors are very common
- **Important**: Read questions carefully for time anchors. "How many days ago did X happen when Y happened?" asks for the time between X and Y, NOT between X and the question date

**Handling Relative Times in Facts:**
- If a fact says "last Friday" or "two weeks ago", anchor it to the fact's "mentioned" date, NOT the question date
- First convert ALL relative references to absolute dates, then answer the question
- Show your date conversion work in your reasoning

**Counting Questions (CRITICAL for "how many" questions):**
- **Scan ALL facts first** - go through every single fact before counting, don't stop early
- **List each item explicitly in your reasoning** before giving the count: "1. X, 2. Y, 3. Z = 3 total"
- **Check all facts and chunks** before giving your final count
- **Watch for duplicates**: The same item may appear in multiple facts. Deduplicate by checking if two facts refer to the same underlying item/event
- **Watch for different descriptions of same thing**: "Dr. Patel (ENT specialist)" and "the ENT specialist" might be the same doctor
- **Don't over-interpret**: A project you "completed" is different from a project you're "leading"
- **Don't double-count**: If the same charity event is mentioned in two conversations, it's still one event

**Disambiguation Guidance (CRITICAL - many errors come from over-counting):**
- **Assume overlap by default**: If two facts describe similar events (same type, similar timeframe, similar details), assume they are the SAME event unless there's clear evidence they are different
- If a person has a name AND a role mentioned, check if they're the same person before counting separately
- If an amount is mentioned multiple times on different dates, check if it's the same event or different events
- When facts reference the same underlying event from different sessions, count it once
- **Check for aliases**: "my college roommate's wedding" and "Emily's wedding" might be the same event
- **Check for time period overlap**: Two "week-long breaks" mentioned in overlapping time periods are likely the same break
- **When in doubt, undercount**: It's better to miss a duplicate than to count the same thing twice

**Question Interpretation (read carefully):**
- "How many X before Y?" - count only X that happened BEFORE Y, not Y itself
- "How many properties viewed before making an offer on Z?" - count OTHER properties, not Z
- "How many X in the last week/month?" - calculate the exact date range from the question date, then filter
- Pay attention to qualifiers like "before", "after", "initially", "currently", "in total"

**When to Say "I Don't Know":**
- If the question asks about something not in the retrieved context, say "I don't have information about X"
- If comparing two things (e.g., "which happened first, X or Y?") but only one is mentioned, explicitly say the other is missing
- Don't guess or infer dates that aren't explicitly stated in the facts or chunks
- If you cannot find a specific piece of information after checking all facts and chunks, admit it
- **Partial knowledge is OK**: If asked about two things and you only have info on one, provide what you know and note what's missing (don't just say "I don't know")

**For Recommendation/Preference Questions (tips, suggestions, advice):**
- **DO NOT invent specific recommendations** (no made-up product names, course names, paper titles, channel names, etc.)
- **DO mention specific brands/products the user ALREADY uses** from the context
- Describe WHAT KIND of recommendation the user would prefer, referencing their existing tools/brands
- Keep answers concise - focus on key preferences (brand, quality level, specific interests) not exhaustive category lists
- First scan ALL facts for user's existing tools, brands, stated preferences

**How to Answer:**
1. Scan ALL facts to find relevant memories - don't stop after finding a few
2. **Read the source chunks carefully** - they contain the actual details you need
3. Convert all relative times to absolute dates
4. Use temporal information to understand when things happened
5. Synthesize information from multiple facts if needed
6. If facts conflict, prefer more recent information
7. Double-check any date calculations before answering
8. **For counting questions ("how many")**: First list each unique item in your reasoning (1. X, 2. Y, 3. Z...), then count them
9. **For recommendations**: Reference the user's existing tools, experiences, or preferences explicitly

"""

LOCOMO_JUDGE_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
        (1) a question (posed by one user to another user),
        (2) a 'gold' (ground truth) answer,
        (3) a generated answer
    which you will score as CORRECT/WRONG.

    The point of the question is to ask about something one user should know about the other user based on their prior conversations.
    The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
    Question: Do you remember what I got the last time I went to Hawaii?
    Gold answer: A shell necklace
    The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

    For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.
    There's an edge case where the actual answer can't be found in the data and in that case the gold answer will say so (e.g. 'You did not mention this information.'); if the generated answer says that it cannot be answered or it doesn't know all the details, it should be counted as CORRECT.
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def _direct_recall(api_url: str, bank_id: str, query: str, timeout: float = 60.0):
    url = f"{api_url}/v1/default/banks/{bank_id}/memories/recall"
    payload = {
        "query": query,
        "budget": "low",
        "max_tokens": 4096,
        "include": {"entities": {"max_tokens": 500}},
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    class SimpleRecallResult:
        def __init__(self, r):
            self.text = r.get("text", "")
            self.context = r.get("context")
            self.occurred_start = r.get("occurred_start")
            self.occurred_end = r.get("occurred_end")
            self.mentioned_at = r.get("mentioned_at")
            self.chunk_id = r.get("chunk_id")

    class SimpleChunk:
        def __init__(self, c):
            self.text = c.get("text", "") if isinstance(c, dict) else ""

    class SimpleRecallResponse:
        def __init__(self, d):
            self.results = [SimpleRecallResult(r) for r in d.get("results", [])]
            raw_chunks = d.get("chunks") or {}
            self.chunks = {k: SimpleChunk(v) for k, v in raw_chunks.items()}
            raw_entities = d.get("entities") or {}
            self.entities = {
                k: type("E", (), {"observations": v.get("observations", []) if isinstance(v, dict) else []})()
                for k, v in raw_entities.items()
            }

    return SimpleRecallResponse(data)


def _parse_date(date_string: str) -> datetime:
    dt = datetime.strptime(date_string, "%I:%M %p on %d %B, %Y")
    return dt.replace(tzinfo=timezone.utc)


def _format_context(recall_response) -> str:
    results = recall_response.results or []
    chunks = recall_response.chunks or {}
    entities = recall_response.entities or {}

    if not results and not entities:
        return "No memories found."

    parts = []
    for i, fact in enumerate(results, 1):
        fp = [f"Fact {i}: {fact.text}"]
        if fact.context:
            fp.append(f"Context: {fact.context}")
        if fact.occurred_start or fact.occurred_end:
            occ = fact.occurred_start or ""
            if fact.occurred_end and fact.occurred_end != fact.occurred_start:
                occ += f" to {fact.occurred_end}"
            fp.append(f"Occurred: {occ}")
        if fact.mentioned_at:
            fp.append(f"Mentioned: {fact.mentioned_at}")
        if fact.chunk_id and chunks:
            chunk = chunks.get(fact.chunk_id)
            if chunk and chunk.text:
                fp.append(f'Source:\n  "{chunk.text}"')
        parts.append("\n".join(fp))

    if entities:
        ep = ["=== Entity Observations ==="]
        for name, state in entities.items():
            obs = getattr(state, "observations", None) or (state.get("observations", []) if isinstance(state, dict) else [])
            if obs:
                ep.append(f"\nEntity: {name}")
                for o in obs:
                    ep.append(f"  - {o.get('text', '') if isinstance(o, dict) else getattr(o, 'text', '')}")
        if len(ep) > 1:
            parts.append("\n".join(ep))

    return "\n\n---\n\n".join(parts)


def _save_quality_result(provider_id: str, model_id: str, result: dict):
    """Merge quality result into the unified leaderboard file."""
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    path = LEADERBOARD_DIR / f"{provider_id}-{model_id}.json"
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing["model_id"] = model_id
    existing["provider_id"] = provider_id
    existing["quality"] = result
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    return path

# ── benchmark class ───────────────────────────────────────────────────────────

class QualityBenchmark:
    """Measures answer accuracy via Hindsight memory recall on LoComo conv-43."""

    DATASET_PATH = DATASETS_DIR / "locomo_quality.json"
    TARGET_CONVERSATION = "conv-43"

    def __init__(self, gemini_api_key: str = None, openai_api_key: str = None):
        if gemini_api_key:
            print(f"Using {ANSWER_GENERATOR_MODEL} for judge and generator")
            self.llm_client = OpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=120.0,
            )
            self.model_name = ANSWER_GENERATOR_MODEL
        elif openai_api_key:
            print("GEMINI_API_KEY not set, falling back to OpenAI gpt-4o-mini")
            self.llm_client = OpenAI(api_key=openai_api_key, timeout=90.0)
            self.model_name = "gpt-4o-mini"
        else:
            self.llm_client = None
            self.model_name = None

    def run(self, model_id: str, provider_id: str, api_url: str, max_questions: Optional[int] = None) -> Dict[str, Any]:
        try:
            from hindsight_client import Hindsight
        except ImportError:
            print("Error: hindsight_client not installed.")
            sys.exit(1)

        with open(self.DATASET_PATH) as f:
            dataset = json.load(f)

        item = next((i for i in dataset if i["sample_id"] == self.TARGET_CONVERSATION), None)
        if item is None:
            raise ValueError(f"Conversation {self.TARGET_CONVERSATION} not found in dataset")

        sample_id = item["sample_id"]
        print(f"\n=== Quality Benchmark ===")
        print(f"Model: {provider_id}/{model_id}")
        print(f"Conversation: {sample_id}")
        print(f"Hindsight API: {api_url}")

        client = Hindsight(base_url=api_url)
        run_ts = int(time.time())
        bank_id = (
            f"qb_{provider_id}_{model_id}_{sample_id}_{run_ts}"
            .replace("/", "_").replace("-", "_").replace(".", "_")
        )
        print(f"\nCreating fresh bank: {bank_id}")
        client.create_bank(bank_id=bank_id)

        print("\nIngesting conversation...")
        conv = item["conversation"]
        speaker_a = conv["speaker_a"]
        speaker_b = conv["speaker_b"]
        session_keys = sorted(k for k in conv if k.startswith("session_") and not k.endswith("_date_time"))
        last_session_date = None

        for session_key in session_keys:
            if not isinstance(conv.get(session_key), list):
                continue
            session_date = _parse_date(conv[f"{session_key}_date_time"])
            last_session_date = session_date
            client.retain(
                bank_id=bank_id,
                content=json.dumps(conv[session_key]),
                context=f"Conversation between {speaker_a} and {speaker_b} ({session_key})",
                timestamp=session_date,
                document_id=f"{sample_id}_{session_key}_{run_ts}",
            )
            print(f"  ✓ Ingested {session_key}", flush=True)

        qa_pairs = item["qa"][:max_questions] if max_questions else item["qa"]
        print(f"\nEvaluating {len(qa_pairs)} questions...")

        correct = 0
        for i, qa in enumerate(qa_pairs, 1):
            question = qa["question"]
            expected = qa.get("answer", "You did not mention this information.")

            recall_response = None
            for attempt in range(2):
                try:
                    recall_response = _direct_recall(api_url, bank_id, question)
                    break
                except Exception as e:
                    if attempt < 1:
                        print(f"  Recall attempt {attempt + 1} failed ({e}), retrying...", flush=True)
                        time.sleep(2)
                    else:
                        print(f"  Recall failed after 2 attempts: {e}", flush=True)
                        recall_response = type("obj", (), {"results": [], "chunks": {}, "entities": {}})()

            predicted = self._generate_answer(question, recall_response, last_session_date)
            is_correct = self._judge_answer(question, expected, predicted)
            print(f"  {'✓' if is_correct else '✗'} Q{i}: {question[:60]}...", flush=True)
            if is_correct:
                correct += 1

        total = len(qa_pairs)
        accuracy = round(correct / total * 100, 1) if total > 0 else 0
        print(f"\n=== Results ===")
        print(f"Accuracy: {accuracy}% ({correct}/{total})")

        result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "model_id": model_id,
            "provider_id": provider_id,
            "sample_id": sample_id,
        }
        path = _save_quality_result(provider_id, model_id, result)
        print(f"Results saved to {path}")
        return result

    def _generate_answer(self, question: str, recall_response, question_date: Optional[datetime] = None) -> str:
        if not self.llm_client:
            return recall_response.results[0].text if recall_response.results else "No memories found"

        context = _format_context(recall_response)
        qdate = question_date.strftime("%Y-%m-%d %H:%M:%S UTC") if question_date else "Not specified"
        prompt = f"""You are a helpful assistant that must answer user questions based on the previous conversations.

{CONTEXT_INSTRUCTIONS}**Answer Guidelines:**
1. Start by scanning retrieved context to understand the facts and events that happened and the timeline.
2. Reason about all the memories and find the right answer, considering the most recent memory as an update of the current facts.
3. If you have 2 possible answers, just say both.

In general the answer must be comprehensive and plenty of details from the retrieved context.

For quantitative/counting questions ("how many..."): First list each unique item in your reasoning (1. X, 2. Y, 3. Z...), scanning ALL facts, then count them for your answer.
If questions asks a location (where...?) make sure to include the location name.
For recommendation questions ("can you recommend...", "suggest...", "any tips..."): DO NOT give actual recommendations. Instead, describe what KIND the user would prefer based on their context. Example answer format: "The user would prefer recommendations for [category] that focus on [their interest]. They would not prefer [what to avoid based on context]."
For questions asking for help or instructions, consider the users' recent memories and previous interactions with the assistant to understand their current situation better (recent purchases, specific product models used..)
For specific number/value questions, use the context to understand what is the most up-to-date number based on recency, but also include the reasoning (in the answer) on previous possible values and why you think are less relevant.
For open questions, include as much details as possible from different sources that are relevant.
For questions where a specific entity/role is mentioned and it's different from your memory, just say the truth, don't make up anything just to fulfill the question. For example, if the question is about a specific sport, you should consider if the memories and the question are about the same sport. (e.g. american football vs soccer, shows vs podcasts)
For comparative questions, say you don't know the answer if you don't have information about both sides. (or more sides)
For questions related to time/date, carefully review the question date and the memories date to correctly answer the question.
For questions related to time/date calculation (e.g. How many days passed between X and Y?), carefully review the memories date to correctly answer the question and only provide an answer if you have information about both X and Y, otherwise say it's not possible to calculate and why.

Consider assistant's previous actions (e.g., bookings, reminders) as impactful to the user experiences.


Question: {question}
Question Date: {qdate}

Retrieved Context:
{context}


Answer:
"""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: Answer generation failed ({e})", flush=True)
            return "Error generating answer"

    def _judge_answer(self, question: str, expected: str, predicted: str) -> bool:
        if not self.llm_client:
            return expected.lower() in predicted.lower()

        prompt = f"""{LOCOMO_JUDGE_PROMPT}

Question: {question}
Gold answer: {expected}
Generated answer: {predicted}
First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true.

Respond with JSON: {{"reasoning": "...", "correct": true or false}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            return bool(json.loads(response.choices[0].message.content).get("correct", False))
        except Exception as e:
            print(f"Warning: Judge failed ({e}), falling back to string matching", flush=True)
            return expected.lower() in predicted.lower()
