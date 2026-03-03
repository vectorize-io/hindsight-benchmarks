"""
Reranker benchmark — measures retrieval ranking quality via Hindsight recall() on LoComo conv-43.

Architecture:
  Phase 0 (annotation, one-time):
    - Ingest conv-43 into a temp bank with rrf reranker (no neural reranking)
    - For each of the 178 non-adversarial QA pairs, call recall(budget="high") to get candidates
    - Use LLM (gemini-2.5-flash) to identify which recalled facts are relevant
    - Save ground truth to datasets/locomo_reranker_gt.json

  Phase 2 (per reranker, using shared ingested bank):
    - For each question, call recall(budget="mid") and match results against annotated relevant facts
    - Compute MRR and Recall@K metrics
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from .benchmark import DATASETS_DIR

RERANKER_RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "results" / "leaderboard" / "reranker"
GT_PATH = DATASETS_DIR / "locomo_reranker_gt.json"

ANNOTATION_MODEL = "gemini-2.5-flash"
TARGET_CONVERSATION = "conv-43"
DATASET_PATH = DATASETS_DIR / "locomo_quality.json"


class RerankerBenchmark:
    """Measures retrieval ranking quality via Hindsight recall() on LoComo conv-43.

    The ingestion step always uses a constant model (gemini-2.5-flash).
    Only the reranker configuration changes per run.
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

    def create_annotations(self, api_url: str, bank_id: str, gt_path: Path = GT_PATH) -> Path:
        """Annotate ground truth relevance for the 178 non-adversarial QA pairs.

        Calls recall(budget="high") for each question, then uses LLM to identify
        which recalled facts contain information relevant to the expected answer.

        Idempotent — skips if gt_path already exists.
        Returns the path to the saved ground truth file.
        """
        if not self.llm_client:
            raise RuntimeError("LLM client required for annotation (set GEMINI_API_KEY)")

        try:
            from hindsight_client import Hindsight
        except ImportError:
            raise RuntimeError("hindsight_client not installed")

        with open(DATASET_PATH) as f:
            dataset = json.load(f)

        item = next((i for i in dataset if i["sample_id"] == TARGET_CONVERSATION), None)
        if item is None:
            raise ValueError(f"Conversation {TARGET_CONVERSATION} not found in dataset")

        # Only non-adversarial QA pairs (Q1-Q178: those with "answer" key)
        qa_pairs = [qa for qa in item["qa"] if "answer" in qa]
        print(f"\nAnnotating {len(qa_pairs)} non-adversarial questions (budget=high)...")

        client = Hindsight(base_url=api_url)
        annotations = []

        for i, qa in enumerate(qa_pairs, 1):
            question = qa["question"]
            expected_answer = qa["answer"]

            # Recall with high budget to get many candidates
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
                print(f"  Q{i}: No results recalled, skipping annotation", flush=True)
                annotations.append({
                    "q_idx": i - 1,
                    "question": question,
                    "expected_answer": expected_answer,
                    "relevant_facts": [],
                })
                continue

            # Use LLM to identify relevant facts
            relevant_facts = self._annotate_relevant_facts(question, expected_answer, results)
            annotations.append({
                "q_idx": i - 1,
                "question": question,
                "expected_answer": expected_answer,
                "relevant_facts": relevant_facts,
            })

            found = len(relevant_facts)
            print(f"  Q{i}/{len(qa_pairs)}: {len(results)} candidates → {found} relevant | {question[:50]}...", flush=True)

        gt_data = {"annotations": annotations}
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gt_path, "w") as f:
            json.dump(gt_data, f, indent=2)
        print(f"\nGround truth saved to {gt_path}")
        return gt_path

    def run_with_bank(
        self,
        reranker_id: str,
        provider: str,
        model: Optional[str],
        api_url: str,
        bank_id: str,
        gt_path: Path = GT_PATH,
    ) -> Dict[str, Any]:
        """Run reranker benchmark against an already-ingested shared bank.

        Loads ground truth, runs recall(budget="mid") for each annotated question,
        matches results against relevant_facts by exact text, and computes MRR and Recall@K.
        Saves result to results/leaderboard/reranker/{reranker_id}.json.
        """
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth not found at {gt_path}. Run create_annotations first.")

        with open(gt_path) as f:
            gt_data = json.load(f)

        annotations = gt_data["annotations"]
        # Only evaluate questions that have at least one relevant fact
        eval_annotations = [a for a in annotations if a["relevant_facts"]]
        skipped = len(annotations) - len(eval_annotations)
        if skipped:
            print(f"  Skipping {skipped} questions with no annotated relevant facts")

        print(f"\n=== Reranker Benchmark ===")
        print(f"Reranker: {reranker_id} (provider={provider}, model={model or 'default'})")
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
                print(f"\n  ABORT: {consecutive_failures} consecutive connection failures — daemon likely crashed.", flush=True)
                break

            results = recall_response.get("results", []) if recall_response else []
            returned_texts = [r.get("text", "") for r in results]

            # Find rank of first relevant fact (1-indexed, None if not found)
            first_rank = None
            for rank, text in enumerate(returned_texts, 1):
                if text in relevant_set:
                    first_rank = rank
                    break

            ranks_first_match.append(first_rank)

            # Recall@K: does any relevant fact appear in top-K?
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
            "reranker_id": reranker_id,
            "provider": provider,
            "model": model,
            "recall_at_1": r_at_1,
            "recall_at_3": r_at_3,
            "recall_at_5": r_at_5,
            "mrr": mrr,
            "avg_latency_s": avg_latency_s,
            "total_questions": total,
            "sample_id": TARGET_CONVERSATION,
        }

        RERANKER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RERANKER_RESULTS_DIR / f"{reranker_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {out_path}")
        return result

    def _annotate_relevant_facts(
        self, question: str, expected_answer: str, results: list
    ) -> List[str]:
        """Use LLM to identify which recalled facts contain relevant information.

        Returns list of exact fact text strings that are relevant.
        """
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
    """Call recall API and return raw JSON dict."""
    url = f"{api_url}/v1/default/banks/{bank_id}/memories/recall"
    payload = {"query": query, "budget": budget, "max_tokens": max_tokens}
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
