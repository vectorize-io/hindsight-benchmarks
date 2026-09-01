"""
Quality benchmark — extraction quality via Hindsight memory recall on a frozen
BEAM 128K subset (4 conversations, 80 ability-tagged questions).

The answer context is built from the extracted facts only. Source chunks are
deliberately excluded: they contain the raw conversation verbatim, so including
them lets a weak extractor score like a strong one as long as retrieval lands
the right chunk. Facts-only makes the score attributable to the model that ran
retain(). Production recall does return chunks, so this is a harder setting
than production; the leaderboard labels it extraction quality.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI

from .benchmark import DATASETS_DIR, LEADERBOARD_DIR

ANSWER_GENERATOR_MODEL = "gemini-3.7-flash"
JUDGE_MODEL = "gemini-3.7-flash"
# Same model via Vertex AI (Google-hosted, billed to GCP credits).
VERTEX_JUDGE_MODEL = "google/gemini-3.7-flash"
VERTEX_JUDGE_PROXY_PORT = 8812

DATASET_NAME = "beam_128k_subset"
CONTEXT_MODE = "facts_only"

# Abort the run when generator or judge errors pass this count; the score would
# measure API availability, not extraction.
MAX_LLM_ERRORS = 4

# One BEAM session is ~90 extraction chunks; slow models need well over the
# client's synchronous per-request budget, so retain runs async and is polled.
RETAIN_POLL_INTERVAL_S = 15
# The largest BEAM session takes ~27 min with thinking off and ~3x that with a
# reasoning model, so an hour rejects work that is progressing normally. This
# is a stuck-job backstop, not a performance bound; the daemon reports failures
# through the operation status, which is what actually ends a bad run.
RETAIN_DEADLINE_S = int(os.getenv("HINDSIGHT_BENCH_RETAIN_DEADLINE_S", "21600"))

# ── token counting ────────────────────────────────────────────────────────────

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    TOKEN_COUNTER = "cl100k_base"

    def count_tokens(text: str) -> int:
        return len(_ENCODER.encode(text))
except ImportError:
    TOKEN_COUNTER = "chars_div_4"

    def count_tokens(text: str) -> int:
        # Rough approximation. The result JSON records which counter ran;
        # mixing counters across a fleet shifts efficiency scores.
        return len(text) // 4

# ── prompts ───────────────────────────────────────────────────────────────────

CONTEXT_INSTRUCTIONS = """**Understanding the Retrieved Context:**
The context contains memory facts extracted from previous conversations between the user and the assistant.

1. **Fact**: An atomic fact extracted from the conversation (e.g., "User loves hiking in mountains")
   - Facts are all you have. There is no raw transcript available.

2. **Temporal Information**:
   - "occurred": When the event actually happened
   - "mentioned": When it was discussed in conversation
   - Use this to understand the timeline and resolve conflicts (prefer more recent info)

**Date Calculations (CRITICAL - read carefully):**
- When calculating days between two dates: count the days from Date A to Date B as (B - A)
- "X days ago" from Question Date means: Question Date minus X days
- When a fact says "three weeks ago" on a certain mentioned date, that refers to 3 weeks before THAT mentioned date, NOT the question date
- Always convert relative times ("last Friday", "two weeks ago") to absolute dates BEFORE comparing
- Double-check your arithmetic - off-by-one errors are very common

**Counting and Ordering Questions:**
- Scan ALL facts before counting or ordering - don't stop early
- List each item explicitly in your reasoning before giving the count or order
- Watch for duplicates: the same item may appear in multiple facts. Deduplicate by checking if two facts refer to the same underlying item/event
- When in doubt, undercount: it's better to miss a duplicate than to count the same thing twice

**Contradictions:**
- If the facts contain contradictory information about the question topic, say so explicitly and describe both statements rather than picking one silently

**When to Say "I Don't Know":**
- If the question asks about something not in the retrieved context, say "I don't have information about X"
- Don't guess or infer details that aren't stated in the facts
- Partial knowledge is OK: if asked about two things and you only have info on one, provide what you know and note what's missing

**How to Answer:**
1. Scan ALL facts to find relevant memories - don't stop after finding a few
2. Convert all relative times to absolute dates
3. Use temporal information to understand when things happened
4. Synthesize information from multiple facts if needed
5. If facts conflict, prefer more recent information unless the question asks about the conflict itself
6. Double-check any date calculations before answering

"""

JUDGE_PROMPT = """Your task is to label a generated answer to a question as CORRECT or WRONG.
You will be given:
    (1) the question (asked by a user about their own prior conversations with an assistant),
    (2) grading material: a gold answer, a rubric of required points, or both,
    (3) the generated answer.

Grade generously on form, strictly on substance:
- The generated answer may be much longer than the gold answer. As long as it contains the key information from the gold answer, it is CORRECT.
- For time questions, different formats of the same date/period are CORRECT ("May 7th" vs "7 May").
- When a rubric is given, the generated answer must cover the substance of the rubric points. Minor wording differences are fine; missing or contradicting a rubric point is WRONG.
- If the gold answer says the information was never mentioned or is unanswerable, the generated answer is CORRECT if it says it doesn't know or lacks that information, and WRONG if it invents an answer.
- An answer that hedges but still commits to wrong specifics is WRONG.
"""

# ── helpers ───────────────────────────────────────────────────────────────────


def _facts_only_recall(client, bank_id: str, query: str, query_timestamp: Optional[datetime] = None):
    """Recall extracted facts. Chunks stay excluded: they hold the raw
    conversation and would decouple the score from the model's extraction.
    Entities stay excluded too: hindsight 0.9.x fills observations from mental
    models, which this pipeline does not build, so the block is always empty.
    query_timestamp anchors relative-time resolution to the conversation's
    era instead of today's wall clock."""
    return client.recall(
        bank_id=bank_id,
        query=query,
        budget="low",
        max_tokens=4096,
        include_chunks=False,
        query_timestamp=query_timestamp.isoformat() if query_timestamp else None,
    )


def _format_context(recall_response) -> str:
    """Facts-only context: extracted facts, no chunks."""
    results = recall_response.results or []

    if not results:
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
        parts.append("\n".join(fp))

    return "\n\n---\n\n".join(parts)


def _format_grading_material(answer: Optional[str], rubric: list[str]) -> str:
    parts = []
    if answer:
        parts.append(f"Gold answer: {answer}")
    if rubric:
        points = "\n".join(f"- {r}" for r in rubric)
        parts.append(f"Rubric (required points):\n{points}")
    return "\n".join(parts) if parts else "Gold answer: (none provided)"


def _retain_with_polling(client, bank_id: str, content: str, context: str, timestamp: datetime, document_id: str):
    """Submit retain asynchronously and poll until the server finishes.

    A synchronous retain of a full BEAM session blows through any reasonable
    HTTP timeout (a TimeoutError with an empty message, ~10 minutes in). The
    async path returns an operation id immediately; extraction progress is
    the server's business.
    """
    resp = client.retain(
        bank_id=bank_id,
        content=content,
        context=context,
        timestamp=timestamp,
        document_id=document_id,
        retain_async=True,
    )
    op_ids = resp.operation_ids or ([resp.operation_id] if resp.operation_id else [])
    if not op_ids:
        raise RuntimeError(f"Async retain returned no operation id for {document_id}")

    deadline = time.time() + RETAIN_DEADLINE_S
    pending = list(op_ids)
    while pending:
        if time.time() > deadline:
            raise RuntimeError(f"Retain of {document_id} exceeded {RETAIN_DEADLINE_S}s (pending: {pending})")
        time.sleep(RETAIN_POLL_INTERVAL_S)
        # client.operations is the raw async API; run its coroutines through
        # the same loop helper the sync wrapper methods use.
        from hindsight_client.hindsight_client import _run_async

        still_pending = []
        for op_id in pending:
            status = _run_async(client.operations.get_operation_status(bank_id, op_id))
            # Server enum: pending, processing, completed, failed, cancelled, not_found
            if status.status == "completed":
                continue
            if status.status in ("failed", "cancelled", "not_found"):
                raise RuntimeError(
                    f"Retain operation {op_id} for {document_id} {status.status}: {status.error_message}"
                )
            still_pending.append(op_id)
        pending = still_pending


def _count_stored_fact_tokens(client, bank_id: str) -> Optional[int]:
    """Sum tokens over all fact texts stored in the bank.

    This is the extraction footprint of the model under test: what it chose to
    write into memory. Verbose extractors pay here. Returns None when the
    listing fails, so efficiency reads as missing rather than as a perfect
    zero.
    """
    total = 0
    offset = 0
    page_size = 500
    try:
        expected = None
        while True:
            page = client.list_memories(bank_id=bank_id, limit=page_size, offset=offset)
            items = page.items or []
            if expected is None:
                expected = page.total or 0
            if not items:
                break
            total += sum(count_tokens(str(it.get("text", ""))) for it in items)
            offset += page_size
            # Bounded by the server-reported total, so a server that ignores
            # `offset` cannot loop this forever.
            if len(items) < page_size or offset >= expected:
                break
        return total
    except Exception as e:
        print(f"Warning: could not count stored facts ({e})", flush=True)
        return None


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
    """Extraction quality via Hindsight memory recall on the BEAM 128K subset."""

    DATASET_PATH = DATASETS_DIR / f"{DATASET_NAME}.json"

    def __init__(
        self,
        vertex_project: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        vertex_judge_port: int = VERTEX_JUDGE_PROXY_PORT,
    ):
        if vertex_project:
            from .gcp import start_token_proxy, vertex_openai_upstream
            start_token_proxy(vertex_openai_upstream(vertex_project), vertex_judge_port)
            self.llm_client = OpenAI(
                api_key="vertex-proxy",
                base_url=f"http://127.0.0.1:{vertex_judge_port}/v1",
                timeout=120.0,
            )
            self.model_name = VERTEX_JUDGE_MODEL
        elif gemini_api_key:
            self.llm_client = OpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=120.0,
            )
            self.model_name = ANSWER_GENERATOR_MODEL
        else:
            raise ValueError(
                "A vertex_project or GEMINI_API_KEY is required: the "
                f"generator/judge is pinned to {JUDGE_MODEL}."
            )
        print(f"Using {self.model_name} for generator and judge")
        self._verify_judge()

    def _verify_judge(self):
        """One throwaway completion before any spend: a judge model id that
        does not resolve would otherwise surface as accuracy 0.0 after the
        full ingest cost."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                max_tokens=100,
            )
            content = (response.choices[0].message.content or "").strip()
        except Exception as e:
            raise RuntimeError(f"Judge model {self.model_name} preflight failed: {e}") from e
        if not content:
            raise RuntimeError(f"Judge model {self.model_name} preflight returned empty content")
        print(f"Judge preflight ok ({content[:20]!r})")

    def run(
        self,
        model_id: str,
        provider_id: str,
        api_url: str,
        max_questions_per_conversation: Optional[int] = None,
        max_conversations: Optional[int] = None,
        save: bool = True,
        reuse_bank_ts: Optional[int] = None,
        bank_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        """reuse_bank_ts: re-evaluate against banks from an earlier run (its
        bank-id timestamp) instead of ingesting. The extraction under test is
        already stored, so only recall, generation, and judging run. Used to
        recover a run whose judge failed mid-flight.

        bank_ts: stamp newly created banks with this instead of the wall clock.
        A caller that ingests once and then re-evaluates under several
        configurations has to know the stamp in advance; without it the ingest
        run picks its own timestamp at call time and no later run can name the
        banks it wrote."""
        try:
            from hindsight_client import Hindsight
        except ImportError:
            print("Error: hindsight_client not installed.")
            sys.exit(1)

        with open(self.DATASET_PATH) as f:
            dataset = json.load(f)
        if max_conversations:
            dataset = dataset[:max_conversations]

        print(f"\n=== Quality Benchmark ({DATASET_NAME}, {CONTEXT_MODE}) ===")
        print(f"Model: {provider_id}/{model_id}")
        print(f"Conversations: {[c['sample_id'] for c in dataset]}")
        print(f"Hindsight API: {api_url}")

        # 600s per request: one retain() call carries a full session (up to
        # ~265k chars, ~90 server-side extraction calls); the client default of
        # 300s times out on slow providers.
        client = Hindsight(base_url=api_url, timeout=600.0)
        run_ts = int(time.time())

        # The engine version is a test condition like hardware: retrieval and
        # extraction plumbing change between releases, and rows from different
        # versions coexist on the board.
        try:
            hindsight_version = client.get_version().api_version
        except Exception:
            hindsight_version = None

        correct = 0
        total = 0
        per_ability: Dict[str, Dict[str, int]] = {}
        stored_fact_tokens = 0
        stored_tokens_known = True
        self.generation_errors = 0
        self.judge_errors = 0

        for item in dataset:
            sample_id = item["sample_id"]
            effective_bank_ts = reuse_bank_ts or bank_ts or run_ts
            bank_id = (
                f"qb_{provider_id}_{model_id}_{sample_id}_{effective_bank_ts}"
                .replace("/", "_").replace("-", "_").replace(".", "_")
            )

            last_session_date = datetime.fromisoformat(item["sessions"][-1]["date_time"])
            if reuse_bank_ts:
                print(f"\n[{sample_id}] Reusing bank {bank_id}")
            else:
                print(f"\n[{sample_id}] Creating bank {bank_id}")
                client.create_bank(bank_id=bank_id)
                for si, session in enumerate(item["sessions"], 1):
                    session_date = datetime.fromisoformat(session["date_time"])
                    t0 = time.time()
                    _retain_with_polling(
                        client,
                        bank_id=bank_id,
                        content=json.dumps(session["messages"]),
                        context=f"Chat between user and assistant about {item['title']} (session {si})",
                        timestamp=session_date,
                        document_id=f"{sample_id}_session_{si}_{run_ts}",
                    )
                    print(f"  ✓ Ingested session {si}/{len(item['sessions'])} ({time.time() - t0:.0f}s)", flush=True)

            bank_tokens = _count_stored_fact_tokens(client, bank_id)
            if bank_tokens is None:
                stored_tokens_known = False
            else:
                stored_fact_tokens += bank_tokens
                print(f"  Stored fact tokens: {bank_tokens}")

            qa_pairs = item["qa"]
            if max_questions_per_conversation:
                qa_pairs = qa_pairs[:max_questions_per_conversation]
            print(f"  Evaluating {len(qa_pairs)} questions...")

            for i, qa in enumerate(qa_pairs, 1):
                question = qa["question"]
                ability = qa.get("ability", "unknown")

                recall_response = None
                for attempt in range(2):
                    try:
                        recall_response = _facts_only_recall(client, bank_id, question, last_session_date)
                        break
                    except Exception as e:
                        if attempt < 1:
                            print(f"  Recall attempt {attempt + 1} failed ({e}), retrying...", flush=True)
                            time.sleep(2)
                        else:
                            print(f"  Recall failed after 2 attempts: {e}", flush=True)
                            recall_response = type("obj", (), {"results": []})()

                predicted = self._generate_answer(question, recall_response, last_session_date)
                is_correct = self._judge_answer(question, qa.get("answer"), qa.get("rubric") or [], predicted)
                print(f"  {'✓' if is_correct else '✗'} [{ability}] Q{i}: {question[:60]}...", flush=True)
                if not is_correct:
                    expected_str = qa.get("answer") or f"rubric: {(qa.get('rubric') or [])[:2]}"
                    print(f"      expected: {str(expected_str)[:150]}", flush=True)
                    print(f"      predicted: {predicted[:150]}", flush=True)

                # Abort rather than let a broken judge/generator masquerade as
                # bad extraction: past this threshold the accuracy is about the
                # API weather, not the model.
                if self.generation_errors > MAX_LLM_ERRORS or self.judge_errors > MAX_LLM_ERRORS:
                    raise RuntimeError(
                        f"Aborting: {self.generation_errors} generation errors, "
                        f"{self.judge_errors} judge errors (max {MAX_LLM_ERRORS})"
                    )

                total += 1
                stats = per_ability.setdefault(ability, {"correct": 0, "total": 0})
                stats["total"] += 1
                if is_correct:
                    correct += 1
                    stats["correct"] += 1

        accuracy = round(correct / total * 100, 1) if total > 0 else 0
        print(f"\n=== Results ===")
        print(f"Accuracy: {accuracy}% ({correct}/{total})")
        for ability in sorted(per_ability):
            s = per_ability[ability]
            print(f"  {ability}: {s['correct']}/{s['total']}")

        partial = bool(max_questions_per_conversation or max_conversations)
        result = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "dataset": DATASET_NAME,
            "judge_model": JUDGE_MODEL,
            "context_mode": CONTEXT_MODE,
            "hindsight_version": hindsight_version,
            "stored_fact_tokens": stored_fact_tokens if stored_tokens_known else None,
            "token_counter": TOKEN_COUNTER,
            "per_ability": per_ability,
            "generation_errors": self.generation_errors,
            "judge_errors": self.judge_errors,
            "partial": partial,
            "model_id": model_id,
            "provider_id": provider_id,
            "sample_ids": [c["sample_id"] for c in dataset],
        }
        if save:
            path = _save_quality_result(provider_id, model_id, result)
            print(f"Results saved to {path}")
        else:
            print("Not saving results (--no-save)")
        return result

    def _generate_answer(self, question: str, recall_response, question_date: Optional[datetime] = None) -> str:
        context = _format_context(recall_response)
        qdate = question_date.strftime("%Y-%m-%d %H:%M:%S UTC") if question_date else "Not specified"
        prompt = f"""You are a helpful assistant that must answer user questions based on memories of previous conversations.

{CONTEXT_INSTRUCTIONS}**Answer Guidelines:**
1. Start by scanning the retrieved facts to understand what happened and the timeline.
2. Reason about all the memories and find the right answer, considering the most recent memory as an update of the current facts.
3. If you have 2 possible answers, just say both.

The answer must be comprehensive, using the details from the retrieved context that are relevant.

For quantitative/counting questions ("how many..."): first list each unique item in your reasoning, scanning ALL facts, then count them for your answer.
For ordering questions, list the items with when each happened, then give the order.
If a question asks a location (where...?) make sure to include the location name.
For questions where the facts contradict each other on the asked topic, point out the contradiction explicitly instead of silently picking one side.
For questions related to time/date, carefully review the question date and the memory dates to correctly answer the question.
For questions related to time/date calculation (e.g. How many days passed between X and Y?), only provide an answer if you have information about both X and Y, otherwise say it's not possible to calculate and why.
If the retrieved facts do not contain the asked information, say you don't have that information. Do not invent details.

Question: {question}
Question Date: {qdate}

Retrieved Context:
{context}


Answer:
"""
        for attempt in range(2):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt < 1:
                    print(f"Warning: answer generation failed ({e}), retrying...", flush=True)
                    time.sleep(3)
                else:
                    print(f"Warning: answer generation failed twice ({e})", flush=True)
        self.generation_errors += 1
        return "Error generating answer"

    def _judge_answer(self, question: str, expected: Optional[str], rubric: list[str], predicted: str) -> bool:
        grading = _format_grading_material(expected, rubric)
        prompt = f"""{JUDGE_PROMPT}

Question: {question}
{grading}
Generated answer: {predicted}

First, provide a short (one sentence) explanation of your reasoning. Short reasoning is preferred.
If it's correct, set correct=true.

Respond with JSON: {{"reasoning": "...", "correct": true or false}}"""

        for attempt in range(2):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                return bool(json.loads(response.choices[0].message.content).get("correct", False))
            except Exception as e:
                if attempt < 1:
                    print(f"Warning: judge failed ({e}), retrying...", flush=True)
                    time.sleep(3)
                else:
                    print(f"Warning: judge failed twice ({e})", flush=True)
        # Counted, scored wrong, and aborted past MAX_LLM_ERRORS. A silent
        # string-match fallback would grade rubric-only questions wrong forever
        # while looking like a judgment.
        self.judge_errors += 1
        return False
