"""
Embeddings benchmark — measures retrieval quality of different embedding models via Hindsight recall() on LoComo conv-43.

Architecture:
  Each embedding model gets its own persistent Docker volume (the DB schema bakes in vector dimensions,
  so models with different dimensions cannot share a bank).

  IMPORTANT: The ground truth (GT) must be annotated per bank, not shared across embedding models.
  This is because the retain LLM is non-deterministic — it phrases/summarizes facts differently per
  ingest run, so fact texts in bank A will not exactly match fact texts in bank B even if the same
  conversation was ingested. Exact string matching against a cross-bank GT would fail.

  For each embedding model:
    Phase 1 (ingest + annotate, once per model):
      - Start a Hindsight container with the target embedding model + RRF reranker
      - Ingest conv-43 into a persistent bank
      - Annotate ground truth: recall(budget="high") per question, LLM identifies relevant facts
      - Save data_dir + bank_id to datasets/embedding_banks.json
      - Save per-model GT to datasets/locomo_embeddings_gt_{embedding_id}.json

    Phase 2 (benchmark, reusable):
      - Start container with same embedding model + fixed reranker (MiniLM-L6)
      - For each annotated question, call recall(budget="mid")
      - Match results against the per-model GT by exact text
      - Compute MRR and Recall@K metrics
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from .benchmark import DATASETS_DIR

EMBEDDINGS_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / "leaderboard" / "embeddings"
# Metadata file: tracks data_dir + bank_id + gt_path per embedding model
BANKS_META_PATH = DATASETS_DIR / "embedding_banks.json"

TARGET_CONVERSATION = "conv-43"
DATASET_PATH = DATASETS_DIR / "locomo_quality.json"
ANNOTATION_MODEL = "gemini-2.5-flash"

# Fixed reranker used for all embedding benchmark runs
FIXED_RERANKER_ID = "local-minilm-l6"
FIXED_RERANKER_PROVIDER = "local"
FIXED_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def gt_path_for_model(embedding_id: str) -> Path:
    """Return the per-model GT path for a given embedding model id."""
    safe_id = embedding_id.replace("/", "-").replace(".", "-")
    return DATASETS_DIR / f"locomo_embeddings_gt_{safe_id}.json"


class EmbeddingsBenchmark:
    """Measures retrieval quality of embedding models via Hindsight recall() on LoComo conv-43.

    The reranker is fixed (MiniLM-L6 cross-encoder) so only the embedding model changes.
    Each model gets its own GT annotation since fact texts differ per ingest.
    """

    def __init__(self, gemini_api_key: str = None):
        if gemini_api_key:
            self.llm_client = OpenAI(
                api_key=gemini_api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                timeout=120.0,
            )
            self.annotation_model = ANNOTATION_MODEL
        else:
            self.llm_client = None
            self.annotation_model = None

    def load_banks_meta(self) -> Dict[str, Any]:
        if not BANKS_META_PATH.exists():
            return {}
        with open(BANKS_META_PATH) as f:
            return json.load(f)

    def save_banks_meta(self, meta: Dict[str, Any]) -> None:
        BANKS_META_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(BANKS_META_PATH, "w") as f:
            json.dump(meta, f, indent=2)

    def get_bank_for_model(self, embedding_id: str) -> tuple[str, str] | tuple[None, None]:
        """Return (data_dir, bank_id) for an embedding model if it exists and data_dir is valid."""
        meta = self.load_banks_meta()
        entry = meta.get(embedding_id, {})
        data_dir = entry.get("data_dir")
        bank_id = entry.get("bank_id")
        if data_dir and bank_id and Path(data_dir).exists():
            return data_dir, bank_id
        return None, None

    def save_bank_for_model(self, embedding_id: str, data_dir: str, bank_id: str) -> None:
        meta = self.load_banks_meta()
        meta[embedding_id] = {"data_dir": data_dir, "bank_id": bank_id, "embedding_id": embedding_id}
        self.save_banks_meta(meta)

    def ingest_bank(self, api_url: str, embedding_id: str, run_ts: int) -> str:
        """Ingest LoComo conv-43 into a new bank. Returns bank_id."""
        from hindsight_client import Hindsight
        from .locomo import parse_locomo_date

        with open(DATASET_PATH) as f:
            dataset = json.load(f)

        item = next((i for i in dataset if i["sample_id"] == TARGET_CONVERSATION), None)
        if item is None:
            raise ValueError(f"Conversation {TARGET_CONVERSATION} not found in dataset")

        client = Hindsight(base_url=api_url)
        safe_id = embedding_id.replace("/", "-").replace(".", "-")
        bank_id = f"emb_{safe_id}_{run_ts}"
        print(f"\nCreating bank: {bank_id}")
        client.create_bank(bank_id=bank_id)

        conv = item["conversation"]
        speaker_a = conv["speaker_a"]
        speaker_b = conv["speaker_b"]
        session_keys = sorted(k for k in conv if k.startswith("session_") and not k.endswith("_date_time"))

        print(f"Ingesting conversation for embedding model '{embedding_id}'...")
        for session_key in session_keys:
            if not isinstance(conv.get(session_key), list):
                continue
            session_date = parse_locomo_date(conv[f"{session_key}_date_time"])
            client.retain(
                bank_id=bank_id,
                content=json.dumps(conv[session_key]),
                context=f"Conversation between {speaker_a} and {speaker_b} ({session_key})",
                timestamp=session_date,
                document_id=f"emb_{safe_id}_{session_key}_{run_ts}",
            )
            print(f"  ✓ Ingested {session_key}", flush=True)

        return bank_id

    def create_annotations(self, api_url: str, bank_id: str, embedding_id: str) -> Path:
        """Annotate ground truth relevance for 178 non-adversarial QA pairs against this model's bank.

        Same as the reranker annotation, but stored per-model. Skips if file already exists.
        Returns the path to the saved GT file.
        """
        if not self.llm_client:
            raise RuntimeError("LLM client required for annotation (set GEMINI_API_KEY)")

        gt_path = gt_path_for_model(embedding_id)
        if gt_path.exists():
            print(f"  GT already exists at {gt_path}, skipping annotation.")
            return gt_path

        with open(DATASET_PATH) as f:
            dataset = json.load(f)

        item = next((i for i in dataset if i["sample_id"] == TARGET_CONVERSATION), None)
        if item is None:
            raise ValueError(f"Conversation {TARGET_CONVERSATION} not found in dataset")

        qa_pairs = [qa for qa in item["qa"] if "answer" in qa]
        print(f"\nAnnotating {len(qa_pairs)} non-adversarial questions (budget=high)...")

        annotations = []
        for i, qa in enumerate(qa_pairs, 1):
            question = qa["question"]
            expected_answer = qa["answer"]

            recall_response = None
            for attempt in range(2):
                try:
                    recall_response = _direct_recall_raw(
                        api_url, bank_id, question, budget="high", max_tokens=16000
                    )
                    break
                except Exception as e:
                    if attempt < 1:
                        print(f"  Q{i} recall attempt failed ({e}), retrying...", flush=True)
                        time.sleep(2)
                    else:
                        print(f"  Q{i} recall failed after 2 attempts: {e}", flush=True)
                        recall_response = {"results": []}

            results = recall_response.get("results", [])
            if not results:
                print(f"  Q{i}: No results, skipping", flush=True)
                annotations.append({
                    "q_idx": i - 1, "question": question,
                    "expected_answer": expected_answer, "relevant_facts": [],
                })
                continue

            relevant_facts = self._annotate_relevant_facts(question, expected_answer, results)
            annotations.append({
                "q_idx": i - 1, "question": question,
                "expected_answer": expected_answer, "relevant_facts": relevant_facts,
            })
            found = len(relevant_facts)
            print(f"  Q{i}/{len(qa_pairs)}: {len(results)} candidates → {found} relevant | {question[:50]}...", flush=True)

        gt_data = {"annotations": annotations, "embedding_id": embedding_id, "bank_id": bank_id}
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_path, "w") as f:
            json.dump(gt_data, f, indent=2)
        print(f"\nGT saved to {gt_path}")
        return gt_path

    def run_with_bank(
        self,
        embedding_id: str,
        provider: str,
        model: str,
        api_url: str,
        bank_id: str,
    ) -> Dict[str, Any]:
        """Run embedding benchmark against an already-ingested bank with per-model GT."""
        gt_path = gt_path_for_model(embedding_id)
        if not gt_path.exists():
            raise FileNotFoundError(f"GT not found at {gt_path}. Run ingest+annotate first.")

        with open(gt_path) as f:
            gt_data = json.load(f)

        annotations = gt_data["annotations"]
        eval_annotations = [a for a in annotations if a["relevant_facts"]]
        skipped = len(annotations) - len(eval_annotations)
        if skipped:
            print(f"  Skipping {skipped} questions with no annotated relevant facts")

        print(f"\n=== Embeddings Benchmark ===")
        print(f"Embedding: {embedding_id} (provider={provider}, model={model})")
        print(f"Reranker: {FIXED_RERANKER_ID} ({FIXED_RERANKER_PROVIDER}/{FIXED_RERANKER_MODEL})")
        print(f"Bank: {bank_id}")
        print(f"Questions: {len(eval_annotations)}")
        print(f"Hindsight API: {api_url}")

        ranks_first_match: List[Optional[int]] = []
        recall_at_k = {1: 0, 3: 0, 5: 0}
        total_latency = 0.0
        successful_calls = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 5

        for i, ann in enumerate(eval_annotations, 1):
            question = ann["question"]
            relevant_set = set(ann["relevant_facts"])

            recall_response = None
            latency = 0.0
            call_succeeded = False
            for attempt in range(2):
                try:
                    t0 = time.time()
                    recall_response = _direct_recall_raw(
                        api_url, bank_id, question, budget="mid", max_tokens=8192
                    )
                    latency = time.time() - t0
                    call_succeeded = True
                    break
                except Exception as e:
                    err_str = str(e)
                    is_connection_err = any(
                        kw in err_str for kw in ("Connect call failed", "Cannot connect", "Connection refused")
                    )
                    if attempt < 1:
                        print(f"  Q{i} recall attempt failed ({err_str[:80]}), retrying...", flush=True)
                        time.sleep(2)
                    else:
                        print(f"  Q{i} recall failed after 2 attempts: {err_str[:80]}", flush=True)
                        recall_response = {"results": []}
                        if is_connection_err:
                            consecutive_failures += 1

            if call_succeeded:
                consecutive_failures = 0
                total_latency += latency
                successful_calls += 1

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"\n  ABORT: {consecutive_failures} consecutive connection failures.", flush=True)
                break

            results = recall_response.get("results", []) if recall_response else []
            returned_texts = [r.get("text", "") for r in results]

            first_rank = None
            for rank, text in enumerate(returned_texts, 1):
                if text in relevant_set:
                    first_rank = rank
                    break

            ranks_first_match.append(first_rank)

            top_texts = set(returned_texts[:1])
            if relevant_set & top_texts:
                recall_at_k[1] += 1
            top_texts = set(returned_texts[:3])
            if relevant_set & top_texts:
                recall_at_k[3] += 1
            top_texts = set(returned_texts[:5])
            if relevant_set & top_texts:
                recall_at_k[5] += 1

            rank_str = f"rank={first_rank}" if first_rank is not None else "not found"
            print(f"  Q{i}/{len(eval_annotations)} ({latency:.1f}s) {rank_str}: {question[:55]}...", flush=True)

        total = len(ranks_first_match)
        mrr = round(
            sum(1.0 / r for r in ranks_first_match if r is not None) / total, 4
        ) if total > 0 else 0.0
        r_at_1 = round(recall_at_k[1] / total, 4) if total > 0 else 0.0
        r_at_3 = round(recall_at_k[3] / total, 4) if total > 0 else 0.0
        r_at_5 = round(recall_at_k[5] / total, 4) if total > 0 else 0.0
        avg_latency_s = round(total_latency / successful_calls, 3) if successful_calls > 0 else 0.0

        print(f"\n=== Results ===")
        print(f"MRR:         {mrr:.4f}")
        print(f"Recall@1:    {r_at_1:.4f}")
        print(f"Recall@3:    {r_at_3:.4f}")
        print(f"Recall@5:    {r_at_5:.4f}")
        print(f"Avg latency: {avg_latency_s}s")

        result = {
            "embedding_id": embedding_id,
            "provider": provider,
            "model": model,
            "reranker_id": FIXED_RERANKER_ID,
            "recall_at_1": r_at_1,
            "recall_at_3": r_at_3,
            "recall_at_5": r_at_5,
            "mrr": mrr,
            "avg_latency_s": avg_latency_s,
            "total_questions": total,
            "sample_id": TARGET_CONVERSATION,
        }

        EMBEDDINGS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = EMBEDDINGS_RESULTS_DIR / f"{embedding_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out_path}")
        return result

    def _annotate_relevant_facts(
        self, question: str, expected_answer: str, results: list
    ) -> List[str]:
        fact_lines = "\n".join(
            f"[{i+1}] {r.get('text', '')}" for i, r in enumerate(results)
        )
        prompt = f"""Given a question and its expected answer, identify which of the following recalled facts contain information that is relevant to answering the question correctly.

Question: {question}
Expected answer: {expected_answer}

Recalled facts:
{fact_lines}

Return a JSON object with a "relevant_indices" key containing a list of 1-based indices of the relevant facts.
Only include facts that directly contain the information needed to answer the question.
If no facts are relevant, return {{"relevant_indices": []}}.

Respond with JSON only: {{"relevant_indices": [1, 3, ...]}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.annotation_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(response.choices[0].message.content)
            indices = data.get("relevant_indices", [])
            relevant = []
            for idx in indices:
                if isinstance(idx, int) and 1 <= idx <= len(results):
                    relevant.append(results[idx - 1].get("text", ""))
            return [t for t in relevant if t]
        except Exception as e:
            print(f"  Warning: Annotation LLM call failed ({e}), returning empty", flush=True)
            return []


def _direct_recall_raw(
    api_url: str, bank_id: str, query: str, budget: str = "mid", max_tokens: int = 8192, timeout: float = 180.0
) -> dict:
    url = f"{api_url}/v1/default/banks/{bank_id}/memories/recall"
    payload = {"query": query, "budget": budget, "max_tokens": max_tokens}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
