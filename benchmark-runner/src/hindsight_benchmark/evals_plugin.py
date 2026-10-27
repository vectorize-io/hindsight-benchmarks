"""Pytest plugin that `hindsight-evals` loads into Hindsight's system evals.

It runs inside the evals' own venv, so it may only use the stdlib and what that
venv has. Two jobs:

* judge with `codex exec` (a ChatGPT subscription) instead of the Gemini judge;
* after each test, read the bank's LLM calls back from the server trace and
  append their token counts to $HS_BENCH_TOKENS_OUT, so cost is real usage.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest
from hindsight_system_evals import judge

_SCHEMA = {
    "type": "object",
    "properties": {"meets_criteria": {"type": "boolean"}, "reasoning": {"type": "string"}},
    "required": ["meets_criteria", "reasoning"],
    "additionalProperties": False,
}
# A thread semaphore, not an asyncio one: every test runs on its own event loop.
_SLOTS = threading.BoundedSemaphore(int(os.getenv("HS_BENCH_JUDGE_CONCURRENCY", "4")))


def _codex(prompt: str) -> str:
    with _SLOTS, tempfile.TemporaryDirectory() as tmp:
        schema, out = Path(tmp, "schema.json"), Path(tmp, "out.txt")
        schema.write_text(json.dumps(_SCHEMA))
        # --ignore-user-config: the user's hooks would otherwise see every judge prompt.
        # An empty working dir and a read-only sandbox: the judge has nothing to look at.
        cmd = [
            "codex", "exec", "-m", judge.judge_model(),
            "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--ignore-rules",
            "-s", "read-only", "-C", tmp, "--output-schema", str(schema), "-o", str(out), "-",
        ]
        done = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=300)
        if done.returncode != 0:
            raise RuntimeError(f"codex judge failed ({done.returncode}): {done.stderr[-500:]}")
        return out.read_text()


async def _judge_once(response: str, criteria: str, context: str | None, temperature: float) -> judge.Verdict:
    # codex exec has no temperature knob, so the judge's confirmations are plain re-asks.
    prompt = (
        judge._SYSTEM
        + "\n\n"
        + (f"## Context\n{context}\n\n" if context else "")
        + f"## Criterion\n{criteria}\n\n## Answer to grade\n{response}"
    )
    text = await asyncio.to_thread(_codex, prompt)
    try:
        parsed = json.loads(text[text.index("{") : text.rindex("}") + 1])
    except ValueError:
        return judge.Verdict(False, f"judge reply was not JSON: {text[:200]}")
    return judge.Verdict(bool(parsed.get("meets_criteria")), str(parsed.get("reasoning", "")))


judge._judge_once = _judge_once


@pytest.fixture(autouse=True)
async def _record_tokens(request: pytest.FixtureRequest, client, bank_id: str):
    yield
    out = os.getenv("HS_BENCH_TOKENS_OUT")
    if not out:
        return
    from hindsight_client_api.api.llm_traces_api import LLMTracesApi

    # ponytail: counts the test's own bank only; a test that makes extra banks is undercounted.
    api = LLMTracesApi(client._api_client)
    totals = {"calls": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
              "decision_calls": 0, "decision_input_tokens": 0}
    for call_type, prefix in (("llm", ""), ("decision", "decision_")):
        offset = 0
        while True:
            page = await api.list_llm_requests(bank_id, call_type=call_type, limit=500, offset=offset)
            for c in page.items:
                totals[prefix + "calls"] += 1
                totals[prefix + "input_tokens"] += c.input_tokens or 0
                if not prefix:
                    totals["cached_tokens"] += c.cached_tokens or 0
                    totals["output_tokens"] += c.output_tokens or 0
            offset += len(page.items)
            if not page.items or offset >= page.total:
                break
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps({"nodeid": request.node.nodeid, **totals}) + "\n")
