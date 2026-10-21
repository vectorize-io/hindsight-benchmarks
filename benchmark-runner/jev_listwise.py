"""Hindsight Choice/tournament ranking, vendored from f7dd3f4fd7420f7beec60c32c965e5e5cf7be066.

_rank_once and _rank preserve the upstream scoring algorithm. Transport is delegated
 to the benchmark for pinned model validation, usage accounting, retries and pacing.
Pruning is disabled, matching Hindsight's default.
"""

import asyncio
import math
from types import SimpleNamespace

import toktok

_RANK_INSTRUCTIONS_PREFIX = "Which candidate answers the question: "
_OPTION_KEY_OVERHEAD = 7
_MAX_QUERY_TOKENS = 2000


def count_tokens(text):
    return toktok._encoding("o200k_base").count(text)


def truncate_to_tokens(text, limit):
    truncated, _ = toktok.truncate(text, max(0, limit), "o200k_base")
    return SimpleNamespace(text=truncated)


class HindsightChoice:
    MAX_OPTIONS = 250
    SHORTLIST = 12
    MAX_QUESTION_TOKENS = 26000

    def __init__(self, remote):
        self.remote = remote
        self.model = remote.config["model"]
        self.usage = {}
        self.call_durations = []

    async def _ask(self, body):
        result, duration = await self.remote.post(body)
        if result["model"] != self.model:
            raise ValueError("Jev response model differs from pinned model")
        probabilities = result["answers"]["rank"]["probabilities"]
        expected = body["questions"]["rank"]["criteria"].keys()
        if probabilities.keys() != expected or not all(
            isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
            for value in probabilities.values()
        ):
            raise ValueError("Invalid Choice probability mapping")
        for key, value in result.get("usage", {}).items():
            self.usage[key] = self.usage.get(key, 0) + value
        self.call_durations.append(duration)
        return result

    async def score(self, query, docs):
        cap = min(_MAX_QUERY_TOKENS, max(50, (self.MAX_QUESTION_TOKENS - 200) // 2))
        if count_tokens(query) > cap:
            query = truncate_to_tokens(query, cap).text
        order = await self._rank(query, docs, list(range(len(docs))))
        if sorted(order) != list(range(len(docs))):
            raise ValueError("Choice did not rank every candidate exactly once")
        scores = [0.0] * len(docs)
        for position, index in enumerate(order):
            scores[index] = (len(order) - position) / len(order)
        self.usage["choice_calls"] = len(self.call_durations)
        return scores, self.usage, sum(self.call_durations)

    async def _rank_once(
        self, query: str, docs: list[str], indices: list[int]
    ) -> list[int]:
        """Rank one group of candidates, returning their indices best first."""
        body = {
            "state": f"Question: {query}",
            "model": self.model,
            "questions": {
                "rank": {
                    "type": "choice",
                    "instructions": f"{_RANK_INSTRUCTIONS_PREFIX}{query}",
                    "criteria": {
                        f"c{position}": docs[index]
                        for position, index in enumerate(indices)
                    },
                }
            },
        }
        result = await self._ask(body)
        probabilities = result["answers"]["rank"]["probabilities"]
        # Sort the option positions, not the indices themselves: the option key encodes
        # the position, and two candidates can carry the same index-independent text.
        by_probability = sorted(
            range(len(indices)),
            key=lambda position: -float(probabilities[f"c{position}"]),
        )
        return [indices[position] for position in by_probability]

    async def _rank(self, query: str, docs: list[str], indices: list[int]) -> list[int]:
        """Rank a whole pool best first, in rounds when it exceeds option or token limits.

        Each round's probabilities are normalised within its own call, so the winners
        are ranked against each other in a finals round. Candidates that do not make the
        finals maintain their caller input order (initial RRF rank) behind the finalists,
        discarding intra-group model ranks since probabilities across separate rounds
        are not on a shared scale (#4599).
        """
        if not indices:
            return []

        # Available token budget for candidate options in one choice question.
        state_tokens = count_tokens(f"Question: {query}")
        # +30: cushion for the question's JSON framing (type, keys) around the instructions.
        instr_tokens = count_tokens(f"{_RANK_INSTRUCTIONS_PREFIX}{query}") + 30
        net_budget = max(50, self.MAX_QUESTION_TOKENS - state_tokens - instr_tokens)

        # Pre-truncate outlier documents that individually exceed the question budget,
        # and copy docs so we don't mutate caller's list.
        effective_docs = list(docs)
        doc_tokens: dict[int, int] = {}
        for index in indices:
            t = count_tokens(effective_docs[index])
            if t + _OPTION_KEY_OVERHEAD > net_budget:
                cap = max(10, net_budget - _OPTION_KEY_OVERHEAD)
                effective_docs[index] = truncate_to_tokens(
                    effective_docs[index], cap
                ).text
                t = count_tokens(effective_docs[index])
            doc_tokens[index] = t

        # Pack candidates into groups bounded by MAX_OPTIONS and net_budget.
        groups: list[list[int]] = []
        curr_group: list[int] = []
        curr_tokens = 0
        for index in indices:
            item_tokens = doc_tokens[index] + _OPTION_KEY_OVERHEAD
            if curr_group and (
                len(curr_group) >= self.MAX_OPTIONS
                or curr_tokens + item_tokens > net_budget
            ):
                groups.append(curr_group)
                curr_group = [index]
                curr_tokens = item_tokens
            else:
                curr_group.append(index)
                curr_tokens += item_tokens
        if curr_group:
            groups.append(curr_group)

        # Fast path: all candidates fit into a single group.
        if len(groups) == 1:
            return await self._rank_once(query, effective_docs, groups[0])

        # Groups with >= 2 candidates are ranked via Jev Choice questions in parallel.
        # Single-candidate groups skip the preliminary round: a 1-candidate Choice
        # is rejected by Jev ("criteria must map 2 or more options") and the winner is trivial.
        multi_indices = [i for i, g in enumerate(groups) if len(g) >= 2]
        ranked_groups: list[list[int]] = [list(g) for g in groups]
        if multi_indices:
            ranked_multi = await asyncio.gather(
                *(
                    self._rank_once(query, effective_docs, groups[i])
                    for i in multi_indices
                )
            )
            for i, r in zip(multi_indices, ranked_multi):
                ranked_groups[i] = r

        # Advance the top of each group to the finals; the rest fall back to RRF order (#4599).
        # The finals is one Choice, so the finalists must fit MAX_OPTIONS too: long documents
        # can split a pool into more than MAX_OPTIONS // SHORTLIST groups, so the per-group
        # quota shrinks, and past MAX_OPTIONS groups the lowest-input-ranked winners overflow
        # into the rest.
        quota = max(1, min(self.SHORTLIST, self.MAX_OPTIONS // len(ranked_groups)))
        finalists = [index for group in ranked_groups for index in group[:quota]]
        rest = [
            index for group in ranked_groups for index in group[quota:]
        ] + finalists[self.MAX_OPTIONS :]
        finalists = finalists[: self.MAX_OPTIONS]
        index_pos = {idx: pos for pos, idx in enumerate(indices)}
        rest.sort(key=lambda idx: index_pos[idx])

        # Finals round: if finalists exceed budget, cap each doc evenly (budget // n).
        finalist_tokens = sum(
            doc_tokens[idx] + _OPTION_KEY_OVERHEAD for idx in finalists
        )
        if finalist_tokens > net_budget:
            cap = max(
                10,
                (net_budget - len(finalists) * _OPTION_KEY_OVERHEAD) // len(finalists),
            )
            for idx in finalists:
                if doc_tokens[idx] > cap:
                    effective_docs[idx] = truncate_to_tokens(
                        effective_docs[idx], cap
                    ).text

        ranked_finalists = await self._rank_once(query, effective_docs, finalists)
        return ranked_finalists + rest
