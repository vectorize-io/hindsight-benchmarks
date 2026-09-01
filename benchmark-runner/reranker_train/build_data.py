#!/usr/bin/env python3
"""
Build reranker distillation data in two stages, with a hard leakage guard.

Stage 1 (general relevance) — public IR triples, English + Chinese. A student
starting from a masked-LM backbone has no ranking knowledge at all, so this is
where it learns what relevance *is*. Public data only.

Stage 2 (domain adaptation) — synthetic conversational-memory documents we
generate ourselves, at Hindsight's real chunk size, in English and Chinese,
with synthetic queries and teacher-mined hard negatives. This is where it
learns *our* distribution. Generated from scratch, so it cannot leak.

Both stages are scored by a teacher reranker; the label is the teacher's margin
between the positive and the negative (MarginMSE), which the literature finds
beats pointwise MSE by margins comparable to scaling the backbone.

LEAKAGE POLICY — enforced, not documented:
  LoCoMo, LongMemEval and BEAM are our evaluation sets and must never be
  trained on. ShareGPT and UltraChat are excluded too, because LongMemEval's
  chat histories are built FROM them — training on ShareGPT would contaminate
  a benchmark we intend to report. MSC is excluded as same-genre/lineage.
Every emitted row carries a `provenance` field so the training set can be
audited after the fact.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx

# Corpora that must never appear in training data. Matched case-insensitively
# against dataset ids and provenance strings.
BLOCKED_CORPORA = [
    "locomo", "longmemeval", "long_memeval", "beam",
    "sharegpt", "share_gpt", "ultrachat",
    "msc", "multi_session_chat", "multi-session-chat",
]

TEACHER_DEFAULT = "BAAI/bge-reranker-v2-m3"
# Max rows kept per unique query in the public triplet sources. The "-all"
# configs would otherwise hand us one query several hundred times over.
PER_QUERY_CAP = 6
VERTEX_MODEL = "google/gemini-3.7-flash"

# Hindsight's DEFAULT_RETAIN_CHUNK_SIZE. Synthetic documents are generated at
# this size so the student trains on the length it will serve.
CHUNK_CHARS = 3000


def assert_not_blocked(name: str) -> None:
    low = name.lower()
    for bad in BLOCKED_CORPORA:
        if bad in low:
            raise SystemExit(
                f"LEAKAGE GUARD: refusing to use '{name}' — it matches blocked corpus '{bad}'. "
                f"These are evaluation sets (or their source material) and training on them "
                f"would invalidate the benchmark."
            )


@dataclass
class Triple:
    query: str
    positive: str
    negative: str
    label: float          # teacher margin: score(pos) - score(neg)
    language: str
    stage: str
    provenance: str


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------


class Teacher:
    """Scores (query, document) pairs with a cross-encoder on the GPU.

    Scores are RAW LOGITS. sentence-transformers applies nn.Sigmoid() by default
    when num_labels == 1, which squashes the teacher into (0, 1) and destroys
    resolution exactly where MarginMSE needs it. Measured on bge-reranker-v2-m3:
        default: [0.2919, 0.0000] -> margin  0.29
        raw:     [-0.8862, -11.0156] -> margin 10.13
    The irrelevant document lands on exactly 0.0 under the default. Distilling
    from that teaches the student a 35x-compressed target.
    """

    def __init__(self, model_id: str, batch_size: int = 64, max_length: int = 1024):
        from sentence_transformers import CrossEncoder
        import torch

        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        print(f"Loading teacher {model_id} (max_length={max_length}, raw logits) ...", flush=True)
        self.model = CrossEncoder(
            model_id,
            max_length=max_length,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"torch_dtype": torch.float16} if torch.cuda.is_available() else {},
        )

    def set_max_length(self, n: int, ref_len: int = 256, min_batch: int = 16) -> None:
        """Widen the window for stage 2 and shrink the batch to match.

        Attention memory grows with sequence length, so a batch tuned for short
        stage-1 passages OOMs at 2560 tokens: at batch 512 the allocator asked
        for 6.4GB with 2.9GB free. Scale the batch down by the length ratio.
        """
        self.max_length = n
        self.model.max_length = n
        scaled = max(min_batch, int(self.batch_size * ref_len / max(1, n)))
        if scaled != self.batch_size:
            print(f"  teacher batch {self.batch_size} -> {scaled} for max_length={n}", flush=True)
            self.batch_size = scaled

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        import torch

        if not pairs:
            return []
        return [
            float(x)
            for x in self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
            )
        ]


# ---------------------------------------------------------------------------
# Stage 1 — public IR triples
# ---------------------------------------------------------------------------


def load_public_triples(n_per_lang: int, seed: int) -> list[tuple[str, str, str, str, str]]:
    """Return (query, positive, negative, language, provenance) from public IR data.

    Mixes several parquet-native sources so stage 1 sees more than one query
    shape and both target languages. Every source is web/wiki retrieval data,
    disjoint from the memory benchmarks we evaluate on.
    """
    from datasets import load_dataset

    rng = random.Random(seed)

    # Mixed sources. MIRACL is human-annotated and per-language, which is what
    # finally gives stage 1 native CHINESE ranking data instead of relying on
    # cross-lingual transfer from English. HotpotQA adds multi-hop questions,
    # a different query shape from web search. MS MARCO stays as the canonical
    # web-ranking corpus. All are web/wiki IR data, disjoint from every memory
    # benchmark we evaluate on.
    # MIRACL alone cannot carry Chinese at scale: zh-triplet-all is 318,700 rows
    # but only 1,312 unique queries, so the per-query cap tops it out near 8k.
    # T2Ranking (large human-annotated Chinese web ranking) and DuReader
    # (Chinese QA retrieval) supply the volume. The plain `triplet` configs are
    # one row per query, so they are diverse by construction.
    # Sizes verified against the datasets-server. The plain `triplet` configs
    # are one row per query and TOO SMALL to hit their targets
    # (dureader:triplet is 80,416 rows; t2ranking:triplet ~90,467), and every
    # shortfall silently backfilled as English MS MARCO — which collapsed
    # Chinese to ~15% while the shares claimed 35%. The `-all` configs have the
    # volume (dureader 3.08M, t2ranking 5.10M) and PER_QUERY_CAP keeps them
    # diverse. MIRACL stays small on purpose: it only has 1,312 zh queries.
    direct_sources = [
        ("sentence-transformers/T2Ranking", "triplet-all", "zh", 0.28),
        ("sentence-transformers/dureader", "triplet-all", "zh", 0.18),
        # MIRACL shares are deliberately tiny: after PER_QUERY_CAP it can only
        # ever yield ~7.9k zh (1,312 queries) and ~17.2k en (2,863 queries).
        # Asking for more would trip the under-delivery guard and halt the run.
        ("sentence-transformers/miracl", "zh-triplet-all", "zh", 0.008),
        ("sentence-transformers/miracl", "en-triplet-all", "en", 0.015),
        ("sentence-transformers/hotpotqa", "triplet-all", "en", 0.25),
    ]
    out: list[tuple[str, str, str, str, str]] = []
    for src_id, cfg, lang, share in direct_sources:
        assert_not_blocked(src_id)
        target = int(n_per_lang * share)
        try:
            print(f"  loading {src_id}:{cfg} (target {target}) ...", flush=True)
            # The `-all` configs enumerate every positive/negative combination
            # GROUPED BY QUERY, so a plain prefix returns hundreds of rows for a
            # handful of queries. Measured: miracl zh-triplet-all is 318,700 rows
            # but only 1,312 unique queries. Cap per query so the sample spreads.
            ds = load_dataset(src_id, cfg, split="train", streaming=True)
            got, seen_per_q, scanned = 0, {}, 0
            bucket: list[tuple[str, str, str, str, str]] = []
            for row in ds:
                scanned += 1
                # Guard first: previously this sat after `got += 1`, so a source
                # matching nothing never reached it and scanned to the end.
                if scanned > 12_000_000:
                    break
                a, p, n = row.get("anchor"), row.get("positive"), row.get("negative")
                if not (a and p and n):
                    continue
                if seen_per_q.get(a, 0) >= PER_QUERY_CAP:
                    continue
                seen_per_q[a] = seen_per_q.get(a, 0) + 1
                bucket.append((a, p, n, lang, f"{src_id}:{cfg}"))
                got += 1
                if got >= target:
                    break
            # A source that quietly under-delivers gets backfilled by MS MARCO,
            # which is how the Chinese share silently halved. Fail loudly.
            if got == 0:
                raise SystemExit(f"{src_id}:{cfg} yielded 0 usable rows — check the config/schema.")
            if got < target * 0.5:
                raise SystemExit(
                    f"{src_id}:{cfg} delivered {got} of {target} ({got/target:.0%}). "
                    f"The shortfall would backfill as English and skew the language mix. "
                    f"Use a larger config or lower this source's share."
                )
            rng.shuffle(bucket)
            out.extend(bucket)
            print(
                f"    got {got} rows over {len(seen_per_q)} unique queries ({lang}), "
                f"scanned {scanned:,}",
                flush=True,
            )
        except Exception as exc:
            # No soft-fail here. A dataset that fails to load, or a stream that
            # dies at row 100k, would otherwise be backfilled by MS MARCO and
            # silently turn a 47%-Chinese mix into a mostly-English one. That is
            # the same failure the guards below exist to prevent, so it halts.
            raise SystemExit(
                f"{src_id}:{cfg} failed to load: {type(exc).__name__}: {str(exc)[:200]}\n"
                f"Halting rather than backfilling the shortfall with English."
            ) from exc

    # Aggregate check: nothing downstream enforces the language mix, so verify
    # it here while we can still tell which source under-delivered.
    zh_rows = sum(1 for r in out if r[3] == "zh")
    zh_target = sum(share for _, _, lang, share in direct_sources if lang == "zh")
    zh_actual_frac = zh_rows / max(1, len(out))
    print(f"  direct sources: {len(out)} rows, {zh_rows} zh ({zh_actual_frac:.1%} of pre-backfill)", flush=True)
    if zh_target > 0 and zh_rows < 0.8 * int(n_per_lang * zh_target):
        raise SystemExit(
            f"Chinese under-delivered: {zh_rows} rows against a target of "
            f"{int(n_per_lang * zh_target)}. The shortfall would backfill as English."
        )

    # MS MARCO is id-only and needs the join below.
    n_per_lang = max(0, n_per_lang - len(out))
    if n_per_lang == 0:
        return out

    ds_id = "sentence-transformers/msmarco"
    assert_not_blocked(ds_id)

    # The `unicamp-dl/mmarco` and `miracl/miracl` repos ship as loading scripts,
    # which current `datasets` refuses to execute; the sources above are the
    # parquet-native equivalents. MIRACL zh-triplet covers the Chinese gap.
    #
    # Every MS MARCO config is ID-only; the text lives in the `corpus` and `queries`
    # configs and has to be joined in. Reading ids as text yields nothing, which
    # is a silent infinite scan rather than an error — hence the yield guard.
    print(f"  reading {ds_id}:bert-ensemble-margin-mse (ids + teacher margin) ...", flush=True)
    pairs = load_dataset(ds_id, "bert-ensemble-margin-mse", split="train", streaming=True)

    wanted: list[tuple[str, str, str]] = []
    need_q: set[str] = set()
    need_d: set[str] = set()
    for row in pairs:
        qid, pid, nid = str(row["query_id"]), str(row["positive_id"]), str(row["negative_id"])
        wanted.append((qid, pid, nid))
        need_q.add(qid)
        need_d.update((pid, nid))
        if len(wanted) >= n_per_lang:
            break
    if not wanted:
        raise SystemExit("bert-ensemble-margin-mse yielded no rows; stage 1 cannot run.")
    print(f"    {len(wanted)} id-triples, need {len(need_q)} queries and {len(need_d)} passages", flush=True)

    def resolve(config: str, id_field: str, text_field: str, needed: set[str], label: str) -> dict[str, str]:
        """One streaming pass, keeping only the ids we actually referenced."""
        found: dict[str, str] = {}
        scanned = 0
        for row in load_dataset(ds_id, config, split="train", streaming=True):
            scanned += 1
            rid = str(row.get(id_field, ""))
            if rid in needed:
                found[rid] = row.get(text_field) or ""
                if len(found) == len(needed):
                    break
            # Fail fast rather than scanning a huge table forever when the
            # schema is not what we assumed.
            if scanned == 200_000 and not found:
                raise SystemExit(
                    f"{config}: scanned {scanned} rows and matched none. "
                    f"Expected id field '{id_field}' and text field '{text_field}', saw {list(row.keys())}."
                )
            if scanned % 2_000_000 == 0:
                print(f"      {label}: scanned {scanned:,}, matched {len(found):,}/{len(needed):,}", flush=True)
        print(f"    {label}: resolved {len(found):,}/{len(needed):,}", flush=True)
        return found

    # Verified schemas: queries = (query_id, query), corpus = (passage_id, passage).
    queries = resolve("queries", "query_id", "query", need_q, "queries")
    corpus = resolve("corpus", "passage_id", "passage", need_d, "corpus")

    for qid, pid, nid in wanted:
        q, p, n = queries.get(qid), corpus.get(pid), corpus.get(nid)
        if q and p and n:
            out.append((q, p, n, "en", f"{ds_id}:bert-ensemble-margin-mse"))
    print(f"    total after join: {len(out)} triples", flush=True)
    if not out:
        raise SystemExit("No stage-1 triples from any source; cannot run.")
    return out


def _row_to_triple(row: dict, rng: random.Random) -> tuple[str, str, str] | None:
    """Normalise the several shapes these datasets ship in into (q, pos, neg)."""
    q = row.get("query") or row.get("question")
    if not q:
        return None

    # mMARCO triples style
    pos = row.get("positive") or row.get("positive_passage")
    neg = row.get("negative") or row.get("negative_passage")
    if isinstance(pos, str) and isinstance(neg, str) and pos and neg:
        return q, pos, neg

    # MIRACL style: lists of passage dicts
    pos_list = row.get("positive_passages") or []
    neg_list = row.get("negative_passages") or []
    if pos_list and neg_list:
        p = rng.choice(pos_list)
        n = rng.choice(neg_list)
        pt = p.get("text") if isinstance(p, dict) else None
        nt = n.get("text") if isinstance(n, dict) else None
        if pt and nt:
            return q, pt, nt
    return None


# ---------------------------------------------------------------------------
# Stage 2 — synthetic in-domain documents and queries
# ---------------------------------------------------------------------------


DOC_PROMPT = {
    "en": (
        "Write a realistic excerpt from a long-running conversation between {relationship}, "
        "of about {chars} characters, {span}. It should read like a transcript segment: several "
        "turns, concrete specifics (names, dates, places, numbers, preferences, plans, things that "
        "changed over time). Topic: {topic}. Output ONLY the excerpt text, no preamble."
    ),
    "zh": (
        "请写一段大约 {chars} 个字符的真实对话摘录，对话双方是{relationship}，内容{span}。"
        "应像一段对话记录：包含多轮往复、具体细节（姓名、日期、地点、数字、偏好、计划、随时间变化的事情）。"
        "主题：{topic}。只输出对话文本，不要任何前言。"
    ),
}

RELATIONSHIPS_ZH = [
    "两位挚友", "一对夫妻", "成年子女与父母", "兄弟姐妹", "多年的同事",
    "导师与学生", "室友", "交往一年的情侣", "大学时期的老朋友",
]
SPANS_ZH = ["跨越数周", "持续数月", "接续一年前的话题", "跨越两个季节"]

QUERY_PROMPT = {
    "en": (
        "Here is an excerpt from someone's conversation history:\n\n{doc}\n\n"
        "Write {n} distinct questions that this excerpt — and specifically this excerpt — answers. "
        "They should read like questions a person would later ask their assistant to recall something. "
        "Return ONLY a JSON object: {{\"queries\": [\"...\"]}}"
    ),
    "zh": (
        "以下是某人对话历史中的一段摘录：\n\n{doc}\n\n"
        "请写出 {n} 个不同的问题，这些问题正是这段摘录能够回答的。"
        "问题应当像一个人稍后向助手询问、想要回忆起某件事时会问的那样。"
        "只返回 JSON 对象：{{\"queries\": [\"...\"]}}"
    ),
}

# Diversity matters more than volume here: queries drawn from a small topic set
# produce near-duplicate training pairs no matter how many you generate. These
# are combined with a relationship and a time span below, so the effective
# space is topics x relationships x spans rather than topics alone.
TOPICS = [
    "planning a move to a new city", "a long-running home renovation", "training for a race",
    "a career change and job interviews", "managing a chronic health condition",
    "a family member's wedding", "learning a musical instrument", "a side business getting off the ground",
    "adopting and raising a dog", "a book club and reading habits", "recurring travel for work",
    "cooking and dietary changes", "a hobby photography project", "supporting an aging parent",
    "a graduate degree part time", "repairing a relationship after a rough patch",
    "budgeting and paying down debt", "a community volunteering commitment",
    "a garden across several seasons", "switching to a new team at work",
    "buying a first home and the mortgage process", "recovering from a sports injury",
    "planning a multi-country trip", "a child starting school", "learning a new language",
    "selling a car and buying another", "a long-distance friendship", "quitting smoking",
    "restoring a vintage bicycle", "a dispute with a landlord", "starting therapy",
    "a promotion and new responsibilities", "planning a milestone birthday party",
    "dealing with a difficult coworker", "a home automation project", "fostering cats",
    "a weekly climbing session", "writing a novel in evenings", "tracking sleep and energy",
    "a kitchen garden and preserving food", "moving a parent into assisted living",
    "an amateur astronomy hobby", "a running injury and physiotherapy", "learning to sail",
    "a podcast side project", "switching to a plant-based diet", "renovating a bathroom",
    "a difficult medical diagnosis in the family", "planning retirement savings",
    "a neighbourhood dispute about parking", "coaching a youth sports team",
    "a long commute and its effects", "adopting an exercise routine after illness",
    "reconnecting with an estranged sibling", "a wedding anniversary trip",
    "managing a team through a reorg", "learning woodworking", "a pet's chronic illness",
    "moving in with a partner", "preparing for a certification exam",
]

RELATIONSHIPS = [
    "two close friends", "a married couple", "an adult child and their parent",
    "two siblings", "long-time colleagues", "a mentor and mentee",
    "roommates", "a couple who have been dating a year", "two old university friends",
]

SPANS = [
    "spread over a few weeks", "spanning several months",
    "picking up a thread from a year earlier", "across two seasons",
]


class Vertex:
    """Gemini through the Vertex OpenAI-compatible endpoint, ADC-authenticated.

    Chris's distillation run hit 429/500 storms on this exact path and needed a
    resumable watcher, so retry and backoff are here from the start.
    """

    def __init__(self, project: str, model: str = VERTEX_MODEL, timeout: float = 180.0):
        self.url = (
            f"https://aiplatform.googleapis.com/v1beta1/projects/{project}"
            f"/locations/global/endpoints/openapi/chat/completions"
        )
        self.model = model
        self.client = httpx.Client(timeout=timeout)
        self._token = ""
        self._token_at = 0.0

    def _auth(self) -> str:
        import subprocess

        if not self._token or time.time() - self._token_at > 300:
            out = subprocess.run(["gcloud", "auth", "print-access-token"], capture_output=True, text=True, timeout=60)
            if out.returncode != 0:
                raise RuntimeError(f"token mint failed: {out.stderr[-300:]}")
            self._token = out.stdout.strip()
            self._token_at = time.time()
        return self._token

    def chat(self, prompt: str, json_mode: bool = False, max_retries: int = 5) -> str | None:
        body: dict = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 1.0}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
            body["temperature"] = 0.0
        delay = 2.0
        for attempt in range(max_retries):
            try:
                r = self.client.post(self.url, json=body, headers={"Authorization": f"Bearer {self._auth()}"})
                if r.status_code == 200:
                    choice = r.json()["choices"][0]
                    return (choice.get("message") or {}).get("content")
                if r.status_code in (429, 500, 503, 504):
                    time.sleep(delay + random.random())
                    delay = min(delay * 2, 60)
                    self._token = ""  # a 401 hiding behind a 500 is cheap to rule out
                    continue
                print(f"    vertex {r.status_code}: {r.text[:160]}", flush=True)
                return None
            except Exception as exc:
                print(f"    vertex {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return None


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def generate_synthetic_docs(
    vertex: Vertex, lang: str, n_docs: int, out_path: Path, seed: int, workers: int = 32
) -> list[str]:
    """Generate n_docs conversational-memory documents, resuming if interrupted.

    Generation is network-bound: each call is ~5-15s, so doing this serially
    costs hours for a few hundred documents. Requests run concurrently and
    results are appended under a lock so an interrupted run still resumes.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    rng = random.Random(seed)
    docs: list[str] = []
    if out_path.exists():
        docs = [json.loads(l)["text"] for l in out_path.open()]
        print(f"  resuming {lang}: {len(docs)} docs already generated", flush=True)
    remaining = n_docs - len(docs)
    if remaining <= 0:
        return docs[:n_docs]

    lock = threading.Lock()
    fh = out_path.open("a")

    def one(_i: int) -> str | None:
        # Sample the three axes independently so documents differ in who is
        # talking and over what period, not just what about.
        topic = rng.choice(TOPICS)
        rel = rng.choice(RELATIONSHIPS if lang == "en" else RELATIONSHIPS_ZH)
        span = rng.choice(SPANS if lang == "en" else SPANS_ZH)
        text = vertex.chat(
            DOC_PROMPT[lang].format(chars=CHUNK_CHARS, topic=topic, relationship=rel, span=span)
        )
        if not text or len(text) < CHUNK_CHARS * 0.4:
            return None
        # A Chinese prompt that answers in English is a silent failure; drop it.
        if lang == "zh" and not has_cjk(text):
            return None
        text = text[: CHUNK_CHARS * 2]
        with lock:
            docs.append(text)
            fh.write(json.dumps({"text": text, "lang": lang, "topic": topic}, ensure_ascii=False) + "\n")
            fh.flush()
            if len(docs) % 50 == 0:
                print(f"    {lang}: {len(docs)}/{n_docs} docs", flush=True)
        return text

    # Over-issue slightly so drops (short output, wrong language) still land us
    # at the target without a second pass.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, i) for i in range(int(remaining * 1.25) + 5)]
        for f in as_completed(futures):
            f.result()
            if len(docs) >= n_docs:
                break
    fh.close()
    print(f"  {lang}: {len(docs)} docs total", flush=True)
    return docs[:n_docs]


def generate_queries(
    vertex: Vertex, docs: list[str], lang: str, per_doc: int, out_path: Path, workers: int = 32
) -> list[tuple[str, int]]:
    """Return (query, doc_index) pairs, resuming if interrupted. Concurrent."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    pairs: list[tuple[str, int]] = []
    done_idx: set[int] = set()
    if out_path.exists():
        for line in out_path.open():
            rec = json.loads(line)
            pairs.append((rec["query"], rec["doc_index"]))
            done_idx.add(rec["doc_index"])
        print(f"  resuming {lang} queries: {len(pairs)} for {len(done_idx)} docs", flush=True)

    todo = [i for i in range(len(docs)) if i not in done_idx]
    if not todo:
        return pairs

    lock = threading.Lock()
    fh = out_path.open("a")
    done_count = [0]

    def one(i: int) -> None:
        raw = vertex.chat(QUERY_PROMPT[lang].format(doc=docs[i][:4000], n=per_doc), json_mode=True)
        if not raw:
            return
        try:
            queries = json.loads(raw).get("queries", [])
        except json.JSONDecodeError:
            return
        with lock:
            for q in queries[:per_doc]:
                min_len = 4 if lang == "zh" else 8
                if not isinstance(q, str) or len(q) < min_len:
                    continue
                if lang == "zh" and not has_cjk(q):
                    continue
                pairs.append((q, i))
                fh.write(json.dumps({"query": q, "doc_index": i, "lang": lang}, ensure_ascii=False) + "\n")
            fh.flush()
            done_count[0] += 1
            if done_count[0] % 50 == 0:
                print(f"    {lang}: queries for {done_count[0]}/{len(todo)} docs", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(one, i) for i in todo]):
            f.result()
    fh.close()
    print(f"  {lang}: {len(pairs)} queries total", flush=True)
    return pairs


def mine_negatives(
    teacher: Teacher,
    query: str,
    positive: str,
    pool: list[str],
    pos_score: float,
    n_negatives: int,
    ceiling_margin: float,
    rank_lo: int,
    rank_hi: int,
    pool_sample: int,
    rng: random.Random,
) -> list[tuple[str, float]]:
    """Return up to n_negatives (text, score), filtered for false negatives.

    Taking the teacher's single top-scoring non-positive is the policy MOST
    likely to pick a document that genuinely answers the query: with ~25 docs
    per topic and recall-style queries ("when is the wedding?"), the top
    non-positive is often a true positive in disguise.

    NV-Retriever's fix, used here (TopK-MarginPos): drop anything scoring within
    `ceiling_margin` LOGITS of the positive, then take negatives from a rank band
    rather than the very top. An absolute margin rather than a percentage,
    because these are raw logits that are routinely negative and a percentage
    threshold inverts below zero.
    """
    # Scoring the FULL pool per query is quadratic and unaffordable: 1500 docs
    # x 18k queries is 27M teacher forwards at 2560 tokens. Sample a candidate
    # set instead — large enough to contain hard negatives, small enough to run.
    others = [d for d in pool if d != positive]
    candidates = rng.sample(others, min(pool_sample, len(others))) if others else []
    if not candidates:
        return []
    scores = teacher.score([(query, c) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda t: -t[1])

    # Positive-anchored ceiling, NV-Retriever TopK-MarginPos: an ABSOLUTE margin
    # below the positive. A percentage threshold only means something on a
    # positive similarity scale; these are raw logits that are routinely
    # negative for relevant pairs (the teacher scores a relevant pair at -0.89),
    # and a percentage ceiling inverts there — admitting exactly the false
    # negatives it exists to drop — and collapses to zero width near 0.
    ceiling = pos_score - ceiling_margin
    kept = [(t, sc) for t, sc in ranked if sc < ceiling]
    band = kept[rank_lo:rank_hi] or kept
    return band[:n_negatives]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_triples(path: Path, triples: list[Triple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for t in triples:
            fh.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
    print(f"Wrote {len(triples)} triples -> {path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both")
    ap.add_argument("--out-dir", default=str(Path.home() / "train" / "data"))
    ap.add_argument("--teacher", default=TEACHER_DEFAULT)
    ap.add_argument("--project", default=os.environ.get("VERTEX_PROJECT", "model-benchmark-506614"))
    ap.add_argument("--stage1-per-lang", type=int, default=150_000)
    ap.add_argument("--stage2-docs-per-lang", type=int, default=400)
    ap.add_argument("--stage2-queries-per-doc", type=int, default=6)
    ap.add_argument("--negatives-per-query", type=int, default=6)
    ap.add_argument("--neg-ceiling-margin", type=float, default=0.75,
                    help="Drop candidates within this many logits of the positive (NV-Retriever TopK-MarginPos).")
    ap.add_argument("--mine-query-batch", type=int, default=16,
                    help="Queries scored per teacher call when mining. One query at a time "
                         "left the A100 at ~50%% occupancy.")
    ap.add_argument("--neg-pool-sample", type=int, default=48,
                    help="Candidates scored per query when mining negatives.")
    ap.add_argument("--neg-rank-lo", type=int, default=1)
    ap.add_argument("--neg-rank-hi", type=int, default=30)
    ap.add_argument("--replay-frac", type=float, default=0.08,
                    help="Stage-1 rows replayed into stage 2, as a fraction of stage-2 size.")
    ap.add_argument("--stage2-teacher-maxlen", type=int, default=2560)
    ap.add_argument("--score-batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    teacher = Teacher(args.teacher, batch_size=args.score_batch)

    if args.stage in ("1", "both"):
        print("\n=== STAGE 1: public general-relevance triples ===", flush=True)
        raw = load_public_triples(args.stage1_per_lang, args.seed)
        print(f"  scoring {len(raw)} triples with the teacher ...", flush=True)
        triples: list[Triple] = []
        B = 2000
        for start in range(0, len(raw), B):
            chunk = raw[start : start + B]
            pos_scores = teacher.score([(q, p) for q, p, _, _, _ in chunk])
            neg_scores = teacher.score([(q, n) for q, _, n, _, _ in chunk])
            for (q, p, n, lang, prov), ps, ns in zip(chunk, pos_scores, neg_scores):
                triples.append(Triple(q, p, n, ps - ns, lang, "stage1", prov))
            print(f"    scored {min(start + B, len(raw))}/{len(raw)}", flush=True)
        write_triples(out_dir / "stage1.jsonl", triples)

    if args.stage in ("2", "both"):
        print("\n=== STAGE 2: synthetic in-domain triples (EN + ZH) ===", flush=True)
        vertex = Vertex(args.project)
        all_triples: list[Triple] = []
        for lang in ("en", "zh"):
            docs = generate_synthetic_docs(
                vertex, lang, args.stage2_docs_per_lang, out_dir / f"synth_docs_{lang}.jsonl", args.seed
            )
            qpairs = generate_queries(
                vertex, docs, lang, args.stage2_queries_per_doc, out_dir / f"synth_queries_{lang}.jsonl"
            )
            # 3000 Chinese characters is ~2000+ tokens. Scoring at the stage-1
            # window would truncate the LABEL, reproducing inside the label
            # pipeline the exact failure this project exists to fix.
            teacher.set_max_length(args.stage2_teacher_maxlen)
            print(
                f"  {lang}: {len(docs)} docs, {len(qpairs)} queries; "
                f"mining negatives (teacher max_length={teacher.max_length}) ...",
                flush=True,
            )
            QB = args.mine_query_batch
            for start in range(0, len(qpairs), QB):
                block = qpairs[start : start + QB]
                # One flat teacher call for the whole block: positives first,
                # then every sampled candidate for every query.
                flat: list[tuple[str, str]] = [(q, docs[di]) for q, di in block]
                cand_sets: list[list[str]] = []
                for q, di in block:
                    pos = docs[di]
                    others = [d for d in docs if d != pos]
                    cands = rng.sample(others, min(args.neg_pool_sample, len(others)))
                    cand_sets.append(cands)
                    flat.extend((q, c) for c in cands)
                scores = teacher.score(flat)

                pos_scores = scores[: len(block)]
                off = len(block)
                for bi, (q, di) in enumerate(block):
                    pos = docs[di]
                    ps = pos_scores[bi]
                    cands = cand_sets[bi]
                    cs = scores[off : off + len(cands)]
                    off += len(cands)
                    ranked = sorted(zip(cands, cs), key=lambda t: -t[1])
                    ceiling = ps - args.neg_ceiling_margin
                    kept = [(t, sc) for t, sc in ranked if sc < ceiling]
                    band = kept[args.neg_rank_lo : args.neg_rank_hi] or kept
                    for neg_text, neg_score in band[: args.negatives_per_query]:
                        all_triples.append(
                            Triple(q, pos, neg_text, ps - neg_score, lang, "stage2",
                                   f"synthetic-vertex:{VERTEX_MODEL}")
                        )
                done = min(start + QB, len(qpairs))
                if done % 200 < QB:
                    print(f"    {lang}: {done}/{len(qpairs)} queries -> {len(all_triples)} triples", flush=True)
            # Checkpoint per language. Mining is the most expensive step here and
            # losing it to a later failure costs hours.
            write_triples(out_dir / f"stage2_partial_{lang}.jsonl", all_triples)
        # Rehearsal. Stage 2 is exclusively 3000-char transcripts; without a
        # replay slice the student can forget short-fact ranking, which
        # production also depends on.
        s1_path = out_dir / "stage1.jsonl"
        if args.replay_frac > 0 and not s1_path.exists():
            print(
                f"  WARNING: replay requested but {s1_path.name} is absent, so no rehearsal "
                f"rows were mixed in. Stage 2 is all long chunks; short-fact quality may regress.",
                flush=True,
            )
        if args.replay_frac > 0 and s1_path.exists():
            k = int(len(all_triples) * args.replay_frac)
            # Reservoir sample over a line-by-line read. Iterating the file
            # object splits ONLY on \n, unlike str.splitlines(), which also
            # breaks on U+2028/U+2029/U+0085 — characters that appear literally
            # in our Chinese rows because we write with ensure_ascii=False.
            reservoir: list[str] = []
            with s1_path.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if len(reservoir) < k:
                        reservoir.append(line)
                    else:
                        j = rng.randrange(i + 1)
                        if j < k:
                            reservoir[j] = line
            added = 0
            for line in reservoir:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                r["stage"] = "stage2-replay"
                all_triples.append(Triple(**r))
                added += 1
            print(f"  replayed {added} stage-1 rows into stage 2 (rehearsal)", flush=True)
        write_triples(out_dir / "stage2.jsonl", all_triples)
        print(f"  stage2 written: {len(all_triples)} triples", flush=True)

    # Final audit: prove nothing blocked slipped in.
    print("\n=== leakage audit ===", flush=True)
    for f in sorted(out_dir.glob("stage*.jsonl")):
        provs: dict[str, int] = {}
        with f.open() as fh:
            for line in fh:
                p = json.loads(line)["provenance"]
                provs[p] = provs.get(p, 0) + 1
        for p in provs:
            assert_not_blocked(p)
        print(f"  {f.name}: {dict(sorted(provs.items(), key=lambda kv: -kv[1]))}", flush=True)
    print("  no blocked corpus present.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
