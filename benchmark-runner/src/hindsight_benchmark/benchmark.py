"""
Core benchmark logic for Hindsight fact extraction.

Supports two backends:
- llama0: registry models served via llama-server (full ~700-token prompt, json_schema)
- mlx: custom model path served via mlx_lm.server (short ~150-token prompt, json_object)
"""

import asyncio
import json
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

BENCHMARK_RUNNER_DIR = Path(__file__).parent.parent.parent
RESULTS_DIR = BENCHMARK_RUNNER_DIR.parent / "results"
DATASETS_DIR = BENCHMARK_RUNNER_DIR / "datasets"

# ── prompts ───────────────────────────────────────────────────────────────────

# Concise fact extraction prompt — exact copy from hindsight-api fact_extraction.py
FULL_PROMPT = """Extract SIGNIFICANT facts from text. Be SELECTIVE - only extract facts worth remembering long-term.

LANGUAGE REQUIREMENT: Detect the language of the input text. All extracted facts, entity names, descriptions, and other output MUST be in the SAME language as the input. Do not translate to another language.

Extract ONLY 'world' and 'assistant' type facts.

══════════════════════════════════════════════════════════════════════════
SELECTIVITY - CRITICAL (Reduces 90% of unnecessary output)
══════════════════════════════════════════════════════════════════════════

ONLY extract facts that are:
✅ Personal info: names, relationships, roles, background
✅ Preferences: likes, dislikes, habits, interests (e.g., "Alice likes coffee")
✅ Significant events: milestones, decisions, achievements, changes
✅ Plans/goals: future intentions, deadlines, commitments
✅ Expertise: skills, knowledge, certifications, experience
✅ Important context: projects, problems, constraints
✅ Sensory/emotional details: feelings, sensations, perceptions that provide context
✅ Observations: descriptions of people, places, things with specific details

DO NOT extract:
❌ Generic greetings: "how are you", "hello", pleasantries without substance
❌ Pure filler: "thanks", "sounds good", "ok", "got it", "sure"
❌ Process chatter: "let me check", "one moment", "I'll look into it"
❌ Repeated info: if already stated, don't extract again

CONSOLIDATE related statements into ONE fact when possible.

══════════════════════════════════════════════════════════════════════════
FACT FORMAT - BE CONCISE
══════════════════════════════════════════════════════════════════════════

1. **what**: Core fact - concise but complete (1-2 sentences max)
2. **when**: Temporal info if mentioned. "N/A" if none. Use day name when known.
3. **where**: Location if relevant. "N/A" if none.
4. **who**: People involved with relationships. "N/A" if just general info.
5. **why**: Context/significance ONLY if important. "N/A" if obvious.

CONCISENESS: Capture the essence, not every word. One good sentence beats three mediocre ones.

══════════════════════════════════════════════════════════════════════════
COREFERENCE RESOLUTION
══════════════════════════════════════════════════════════════════════════

Link generic references to names when both appear:
- "my roommate" + "Emily" → use "Emily (user's roommate)"
- "the manager" + "Sarah" → use "Sarah (the manager)"

══════════════════════════════════════════════════════════════════════════
CLASSIFICATION
══════════════════════════════════════════════════════════════════════════

fact_kind:
- "event": Specific datable occurrence (set occurred_start/end)
- "conversation": Ongoing state, preference, trait (no dates)

fact_type:
- "world": About user's life, other people, external events
- "assistant": Interactions with assistant (requests, recommendations)

══════════════════════════════════════════════════════════════════════════
TEMPORAL HANDLING
══════════════════════════════════════════════════════════════════════════

Use "Event Date" from input as reference for relative dates.
- "yesterday" relative to Event Date, not today
- For events: set occurred_start AND occurred_end (same for point events)
- For conversation facts: NO occurred dates

══════════════════════════════════════════════════════════════════════════
ENTITIES
══════════════════════════════════════════════════════════════════════════

Include: people names, organizations, places, key objects, abstract concepts (career, friendship, etc.)
Always include "user" when fact is about the user.

══════════════════════════════════════════════════════════════════════════
EXAMPLES
══════════════════════════════════════════════════════════════════════════

Example 1 - Selective extraction (Event Date: June 10, 2024):
Input: "Hey! How's it going? Good morning! So I'm planning my wedding - want a small outdoor ceremony. Just got back from Emily's wedding, she married Sarah at a rooftop garden. It was nice weather. I grabbed a coffee on the way."

Output: ONLY 2 facts (skip greetings, weather, coffee):
1. what="User planning wedding, wants small outdoor ceremony", who="user", why="N/A", entities=["user", "wedding"]
2. what="Emily married Sarah at rooftop garden", who="Emily (user's friend), Sarah", occurred_start="2024-06-09", entities=["Emily", "Sarah", "wedding"]

Example 2 - Professional context:
Input: "Alice has 5 years of Kubernetes experience and holds CKA certification. She's been leading the infrastructure team since March. By the way, she prefers dark roast coffee."

Output: ONLY 2 facts (skip coffee preference - too trivial):
1. what="Alice has 5 years Kubernetes experience, CKA certified", who="Alice", entities=["Alice", "Kubernetes", "CKA"]
2. what="Alice leads infrastructure team since March", who="Alice", entities=["Alice", "infrastructure"]

══════════════════════════════════════════════════════════════════════════
QUALITY OVER QUANTITY
══════════════════════════════════════════════════════════════════════════

Ask: "Would this be useful to recall in 6 months?" If no, skip it.

IMPORTANT: Sensory/emotional details and observations that provide meaningful context
about experiences ARE important to remember, even if they seem small (e.g., how food
tasted, how someone looked, how loud music was). Extract these if they characterize
an experience or person.

OUTPUT FORMAT: Respond with ONLY a raw JSON object. No code fences, no markdown, no explanation. Format: {"facts": [...]}"""

SHORT_PROMPT = """Extract significant facts from the conversation. Be selective - only facts worth remembering long-term.

Output JSON with a "facts" array. Each fact has:
- what: the core fact (1-2 sentences, concise)
- when: temporal info or "N/A"
- where: location or "N/A"
- who: people involved or "N/A"
- why: context/significance or "N/A"
- fact_type: "world" or "assistant"
- fact_kind: "event" (datable) or "conversation" (ongoing state)
- occurred_start / occurred_end: ISO date strings for events, null for conversation facts
- entities: [{"text": "name"}] list of people/orgs/places mentioned

Skip greetings, filler, and trivial info. Consolidate related facts."""


# ── schema (Pydantic models mirroring hindsight-api ExtractedFact) ─────────────

class Entity(BaseModel):
    """An entity extracted from text."""
    text: str = Field(
        description="The specific, named entity as it appears in the fact. Must be a proper noun or specific identifier."
    )


class FactCausalRelation(BaseModel):
    """Causal relationship from this fact to a PREVIOUS fact."""
    target_index: int = Field(
        description="Index of the PREVIOUS fact this relates to (0-based). "
                    "MUST be less than this fact's position in the list. "
                    "Example: if this is fact #5, target_index can only be 0, 1, 2, 3, or 4."
    )
    relation_type: Literal["caused_by"] = Field(
        description="How this fact relates to the target fact: 'caused_by' = this fact was caused by the target fact"
    )
    strength: float = Field(
        description="Strength of relationship (0.0 to 1.0). 1.0 = strong, 0.5 = moderate",
        ge=0.0,
        le=1.0,
        default=1.0,
    )


class ExtractedFact(BaseModel):
    """A single extracted fact."""
    model_config = ConfigDict(
        json_schema_mode="validation",
        json_schema_extra={"required": ["what", "when", "where", "who", "why", "fact_type"]},
    )

    what: str = Field(description="Core fact - concise but complete (1-2 sentences)")
    when: str = Field(description="When it happened. 'N/A' if unknown.")
    where: str = Field(description="Location if relevant. 'N/A' if none.")
    who: str = Field(description="People involved with relationships. 'N/A' if general.")
    why: str = Field(description="Context/significance if important. 'N/A' if obvious.")

    fact_kind: str = Field(default="conversation", description="'event' or 'conversation'")
    occurred_start: str | None = Field(default=None, description="ISO timestamp for events")
    occurred_end: str | None = Field(default=None, description="ISO timestamp for event end")
    fact_type: Literal["world", "assistant"] = Field(description="'world' or 'assistant'")
    entities: list[Entity] | None = Field(default=None, description="People, places, concepts")
    causal_relations: list[FactCausalRelation] | None = Field(
        default=None, description="Links to previous facts (target_index < this fact's index)"
    )

    @field_validator("entities", mode="before")
    @classmethod
    def ensure_entities_list(cls, v):
        if v is None:
            return []
        return v


class FactExtractionResponse(BaseModel):
    """Response containing all extracted facts."""
    facts: list[ExtractedFact] = Field(description="List of extracted factual statements")


FACT_SCHEMA = FactExtractionResponse.model_json_schema()


def load_dataset(name: str) -> list[str]:
    """Load a named dataset from the datasets directory."""
    path = DATASETS_DIR / f"{name}.json"
    if not path.exists():
        available = [p.stem for p in DATASETS_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"Dataset '{name}' not found. Available: {', '.join(sorted(available))}"
        )
    with open(path) as f:
        return json.load(f)


USER_MESSAGE_TEMPLATE = """Extract facts from the following text chunk. Only extract facts from the TEXT section — do not treat the event date or context header as facts.

Chunk: 1/1
Event Date: Monday, June 10, 2024 (2024-06-10T00:00:00+00:00)
Context: Benchmark test

Text:
{text}"""


# ── result types ──────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    test_index: int
    latency_s: float
    num_facts: int
    valid_json: bool
    success: bool
    retries: int = 0   # number of retries needed (0 = succeeded on first attempt)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


@dataclass
class BenchmarkRun:
    timestamp: str
    model_id: str
    model_name: str
    provider_id: str = "local"  # provider identifier (e.g., "groq", "openai", "gemini", "ollama-cloud", "local")
    size_gb: float = 0.0
    dataset: str = "simple"
    concurrency: int = 1  # number of parallel requests sent simultaneously
    wall_s: float = 0.0   # actual total elapsed time from first request to last completion
    tests: list = field(default_factory=list)

    def success_rate(self) -> float:
        ok = [t for t in self.tests if t.success and t.valid_json]
        return len(ok) / len(self.tests) if self.tests else 0.0

    def mean_latency(self) -> float:
        lats = [t.latency_s for t in self.tests if t.success]
        return statistics.mean(lats) if lats else 0.0


# ── result persistence ────────────────────────────────────────────────────────

LEADERBOARD_DIR = RESULTS_DIR / "leaderboard" / "llm"


def _result_path(provider_id: str, model_id: str) -> Path:
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    safe = model_id.replace("/", "_").replace(" ", "-").lower()
    return LEADERBOARD_DIR / f"{provider_id}-{safe}.json"


def save_run(provider_id: str, model_id: str, run: BenchmarkRun):
    path = _result_path(provider_id, model_id)

    ok = [t for t in run.tests if t.success and t.valid_json]
    avg_latency_s = round(statistics.mean(t.latency_s for t in ok), 3) if ok else None
    wall_s = round(run.wall_s, 3)
    if ok and wall_s > 0:
        throughput_rps = round(len(ok) / wall_s, 3)
        total_completion_tokens = sum(t.completion_tokens for t in ok)
        total_prompt_tokens = sum(t.prompt_tokens for t in ok)
        total_facts = sum(t.num_facts for t in ok)
        completion_toks_s = round(total_completion_tokens / wall_s, 1) if total_completion_tokens else None
        total_toks_s = round((total_prompt_tokens + total_completion_tokens) / wall_s, 1) if total_prompt_tokens else None
        out_in_ratio = round(total_completion_tokens / total_prompt_tokens, 3) if total_prompt_tokens else None
        tokens_per_fact = round(total_completion_tokens / total_facts, 1) if total_facts else None
    else:
        throughput_rps = None
        completion_toks_s = None
        total_toks_s = None
        out_in_ratio = None
        tokens_per_fact = None

    extraction_data = {
        **{k: v for k, v in asdict(run).items() if k != "tests"},
        "summary": {
            "success": len(ok),
            "total": len(run.tests),
            "wall_s": wall_s,
            "avg_latency_s": avg_latency_s,
            "throughput_rps": throughput_rps,
            "completion_toks_s": completion_toks_s,
            "total_toks_s": total_toks_s,
            "out_in_ratio": out_in_ratio,
            "tokens_per_fact": tokens_per_fact,
        },
        "tests": [asdict(t) for t in run.tests],
    }

    # Read existing unified file and merge (preserves quality data if present)
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)

    existing["model_id"] = model_id
    existing["provider_id"] = provider_id
    existing["retain"] = extraction_data

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def load_runs(provider_id: str, model_id: str) -> Optional[dict]:
    """Load the single run for a model (we only keep the latest)."""
    path = _result_path(provider_id, model_id)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_latest_run(provider_id: str, model_id: str) -> Optional[dict]:
    """Load the latest run (same as load_runs since we only keep one)."""
    return load_runs(provider_id, model_id)


# ── inference helpers ─────────────────────────────────────────────────────────

MAX_RETRIES = 1


def _strip_code_fences(content: str) -> str:
    """Strip markdown code fences and handle duplicate JSON output.

    Models sometimes output the JSON twice: once raw, then again in a code fence.
    If the content has 'Extra data' after the first valid JSON object, we truncate
    to just the first object.
    """
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    # Handle duplicate output: first JSON object followed by extra data (e.g. a second copy)
    try:
        json.loads(content)
    except json.JSONDecodeError as e:
        if "Extra data" in str(e) and e.pos > 0:
            content = content[:e.pos].strip()

    return content


def _validate(content: str) -> tuple[bool, str, int]:
    """Validate raw JSON content. Returns (ok, error_cause, num_facts).

    Lenient validation: only requires valid JSON with a non-empty 'facts' array
    where each fact has a non-empty 'what' field. Does not enforce full Pydantic
    schema — models without grammar constraints may omit optional fields.
    """
    content = _strip_code_fences(content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        return False, f"invalid_json: {e} | raw: {content[:300]}", 0
    if not isinstance(parsed, dict) or "facts" not in parsed:
        return False, f"missing_facts_key | raw: {content[:200]}", 0
    facts = parsed["facts"]
    if not isinstance(facts, list) or not facts:
        return False, f"empty_facts | raw: {content[:300]}", 0
    empty_what = [i for i, f in enumerate(facts) if not isinstance(f, dict) or not f.get("what", "").strip()]
    if empty_what:
        return False, f"empty_what at indices {empty_what} | raw: {content[:300]}", len(facts)
    return True, "", len(facts)


async def _call_inference(client: AsyncOpenAI, text: str, use_short_prompt: bool, model: str, reasoning_effort: str = "", extra_body: Optional[dict] = None) -> TestResult:
    """Call OpenAI-compatible API asynchronously."""
    user_msg = USER_MESSAGE_TEMPLATE.format(text=text)
    prompt = SHORT_PROMPT if use_short_prompt else FULL_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_msg},
    ]

    # Build kwargs for chat.completions.create
    kwargs = {
        "model": model or "default",  # Some servers require a model field
        "messages": messages,
    }

    # reasoning_effort is the parameter OpenAI and vLLM both take, and the one
    # the Hindsight daemon sends, so the fast benchmark and the quality run
    # exercise the same thinking level. Models that steer thinking through
    # other body fields (Gemma/Qwen chat_template_kwargs) pass them in
    # extra_body instead, which is forwarded verbatim.
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if extra_body:
        kwargs["extra_body"] = dict(extra_body)

    # GPT-5 models have different parameter requirements
    model_name = (model or "").lower()
    if "gpt-5" in model_name:
        kwargs["max_completion_tokens"] = 16384
        # GPT-5 only supports temperature=1 (default), so omit it
    else:
        kwargs["max_tokens"] = 16384
        kwargs["temperature"] = 0.1

    if use_short_prompt:
        # mlx_lm.server supports json_object mode only
        kwargs["response_format"] = {"type": "json_object"}

    last_error = ""
    t0 = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(**kwargs, timeout=120.0)
            content = response.choices[0].message.content or ""
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            ok, cause, num_facts = _validate(content)
            if ok:
                return TestResult(0, time.time() - t0, num_facts, True, True,
                                  retries=attempt - 1,
                                  prompt_tokens=prompt_tokens,
                                  completion_tokens=completion_tokens)
            last_error = f"attempt {attempt}: {cause}"
        except Exception as e:
            last_error = f"attempt {attempt}: exception: {e}"

    return TestResult(0, time.time() - t0, 0, False, False, retries=MAX_RETRIES, error=last_error)


# ── test runner ───────────────────────────────────────────────────────────────

async def _run_tests(run: BenchmarkRun, texts: list[str], url: str, use_short_prompt: bool, concurrency: int, api_key: str = "", model: str = "", reasoning_effort: str = "", extra_body: Optional[dict] = None):
    """Run all texts against url asynchronously, populating run.tests. Respects concurrency."""
    t_start = time.time()

    # Create OpenAI client with base_url pointing to the server
    client = AsyncOpenAI(
        base_url=f"{url}/v1",
        api_key=api_key or "dummy",  # Some servers don't check, but client requires non-empty
    )

    # Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)

    async def run_single(idx: int, text: str) -> TestResult:
        async with semaphore:
            result = await _call_inference(client, text, use_short_prompt, model, reasoning_effort, extra_body)
            result.test_index = idx
            # Print as soon as result is ready
            retry_str = f"  [{result.retries} retr{'y' if result.retries == 1 else 'ies'}]" if result.retries else ""
            status = f"OK  {result.latency_s:.2f}s  ({result.num_facts} facts){retry_str}" if result.success else f"FAIL  {result.error[:80]}"
            print(f"  [test {idx}] {status}")
            return result

    # Run all tests concurrently
    tasks = [run_single(idx, text) for idx, text in enumerate(texts, 1)]
    results = await asyncio.gather(*tasks)

    run.wall_s = time.time() - t_start
    run.tests = sorted(results, key=lambda r: r.test_index)


def _print_result(result):
    if result.success:
        retry_str = f"  [{result.retries} retr{'y' if result.retries == 1 else 'ies'}]" if result.retries else ""
        print(f"OK  {result.latency_s:.2f}s  ({result.num_facts} facts){retry_str}")
    else:
        print(f"FAIL  {result.error[:80]}")


# ── remote url backend ────────────────────────────────────────────────────────

def run_url(url: str, model_id: str, model_name: str, provider_id: str = "remote", concurrency: int = 1, dataset: str = "", api_key: str = "", model: str = "", reasoning_effort: str = "", extra_body: Optional[dict] = None) -> BenchmarkRun:
    texts = load_dataset(dataset)
    print(f"\n{'='*60}")
    print(f"  {model_name} via {url}  [concurrency={concurrency}]  [dataset={dataset}, {len(texts)} texts]")
    print(f"{'='*60}\n")

    run = BenchmarkRun(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_id=model_id,
        model_name=model_name,
        provider_id=provider_id,
        size_gb=0.0,
        dataset=dataset,
        concurrency=concurrency,
    )
    asyncio.run(_run_tests(run, texts, url, use_short_prompt=False, concurrency=concurrency, api_key=api_key, model=model, reasoning_effort=reasoning_effort, extra_body=extra_body))
    return run


def run_gemini(model_name_api: str, model_id: str, model_name: str, provider_id: str = "gemini", concurrency: int = 1, dataset: str = "", api_key: str = "") -> BenchmarkRun:
    """Run benchmark using Google Gemini API."""
    try:
        from google import genai
    except ImportError:
        raise RuntimeError("google-genai is required. Install with: pip install google-genai")

    texts = load_dataset(dataset)
    print(f"\n{'='*60}")
    print(f"  {model_name} (Gemini API)  [concurrency={concurrency}]  [dataset={dataset}, {len(texts)} texts]")
    print(f"{'='*60}\n")

    run = BenchmarkRun(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_id=model_id,
        model_name=model_name,
        provider_id=provider_id,
        size_gb=0.0,
        dataset=dataset,
        concurrency=concurrency,
    )

    client = genai.Client(api_key=api_key)

    async def _call_gemini_inference(text: str, use_short_prompt: bool) -> TestResult:
        """Call Gemini API asynchronously."""
        user_msg = USER_MESSAGE_TEMPLATE.format(text=text)
        prompt = SHORT_PROMPT if use_short_prompt else FULL_PROMPT
        full_prompt = f"{prompt}\n\n{user_msg}"

        start = time.time()
        retries = 0

        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name_api,
                    contents=full_prompt,
                    config={
                        "temperature": 0.1,
                        "max_output_tokens": 32768,
                    }
                )

                content = response.text.strip()
                content = _strip_code_fences(content)

                # Parse and validate JSON
                data = json.loads(content)
                facts = data.get("facts", [])

                # Count tokens from usage metadata
                prompt_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
                completion_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

                latency = time.time() - start
                return TestResult(
                    test_index=0,
                    latency_s=latency,
                    num_facts=len(facts),
                    valid_json=True,
                    success=True,
                    retries=retries,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    error="",
                )
            except Exception as e:
                retries = attempt + 1
                if attempt == MAX_RETRIES - 1:
                    return TestResult(
                        test_index=0,
                        latency_s=time.time() - start,
                        num_facts=0,
                        valid_json=False,
                        success=False,
                        retries=retries,
                        prompt_tokens=0,
                        completion_tokens=0,
                        error=str(e)[:200],
                    )
                await asyncio.sleep(1)

        return TestResult(test_index=0, latency_s=0, num_facts=0, valid_json=False, success=False, retries=0, prompt_tokens=0, completion_tokens=0, error="Max retries exceeded")

    async def _run_gemini_tests(texts: list[str], concurrency_limit: int):
        """Run all texts with concurrency control."""
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def run_single(idx: int, text: str) -> TestResult:
            async with semaphore:
                result = await _call_gemini_inference(text, use_short_prompt=False)
                result.test_index = idx
                status = f"{result.latency_s:.2f}s  ({result.num_facts} facts)" if result.success else f"FAIL  {result.error[:80]}"
                retry_str = f"  [{result.retries} retr{'y' if result.retries == 1 else 'ies'}]" if result.retries else ""
                print(f"  [test {idx}] {'OK' if result.success else 'FAIL'}  {status}{retry_str}")
                return result

        wall_start = time.time()
        tasks = [run_single(idx, text) for idx, text in enumerate(texts, 1)]
        results = await asyncio.gather(*tasks)
        run.wall_s = time.time() - wall_start
        run.tests = sorted(results, key=lambda r: r.test_index)

    asyncio.run(_run_gemini_tests(texts, concurrency))
    return run


# ── llama0 backend ────────────────────────────────────────────────────────────

def run_llama0(model_id: str, model_name: str, size_gb: float, provider_id: str = "local", port: int = 9500, concurrency: int = 1, dataset: str = "") -> BenchmarkRun:
    from llama0 import Llama0

    texts = load_dataset(dataset)
    print(f"\n{'='*60}")
    print(f"  {model_name} ({size_gb:.1f} GB) via llama0  [concurrency={concurrency}]  [dataset={dataset}, {len(texts)} texts]")
    print(f"{'='*60}")

    run = BenchmarkRun(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_id=model_id,
        model_name=model_name,
        provider_id=provider_id,
        size_gb=size_gb,
        dataset=dataset,
        concurrency=concurrency,
    )

    print(f"  Starting server on port {port}...")
    llm = Llama0(model_id, port=port, n_ctx=131072, n_parallel=concurrency, verbose=False)
    llm.start(timeout=90)
    print(f"  Server ready\n")

    try:
        asyncio.run(_run_tests(run, texts, llm.url, use_short_prompt=False, concurrency=concurrency))
    finally:
        llm.stop()

    return run


# ── mlx backend ───────────────────────────────────────────────────────────────

def _start_mlx_server(model_path: Path, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server",
         "--model", str(model_path),
         "--port", str(port),
         "--log-level", "WARNING"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    url = f"http://localhost:{port}"
    for attempt in range(90):
        time.sleep(2)
        try:
            requests.get(f"{url}/v1/models", timeout=2)
            print(f"  Server ready (attempt {attempt + 1})")
            return proc
        except Exception:
            pass
        if proc.poll() is not None:
            stderr = proc.stderr.read().decode()
            raise RuntimeError(f"mlx_lm.server crashed:\n{stderr}")
    proc.terminate()
    raise RuntimeError("mlx_lm.server did not become ready in time.")


def run_mlx(model_path: Path, model_id: str, model_name: str, provider_id: str = "local", size_gb: float = 0.0, port: int = 9600, concurrency: int = 1, dataset: str = "") -> BenchmarkRun:
    texts = load_dataset(dataset)
    print(f"\n{'='*60}")
    print(f"  {model_name} via mlx_lm.server  [concurrency={concurrency}]  [dataset={dataset}, {len(texts)} texts]")
    print(f"  Path: {model_path}")
    print(f"{'='*60}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if size_gb == 0.0:
        size_bytes = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        size_gb = size_bytes / 1024 ** 3

    run = BenchmarkRun(
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_id=model_id,
        model_name=model_name,
        provider_id=provider_id,
        size_gb=size_gb,
        dataset=dataset,
        concurrency=concurrency,
    )

    url = f"http://localhost:{port}"
    print(f"  Starting mlx_lm.server on port {port}...")
    proc = _start_mlx_server(model_path, port)

    try:
        asyncio.run(_run_tests(run, texts, url, use_short_prompt=True, concurrency=concurrency))
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("  Server stopped.")

    return run
