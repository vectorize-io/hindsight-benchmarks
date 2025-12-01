# Hindsight Benchmark Results

This repository contains benchmark results for the Hindsight memory system and visualization tools to inspect the results.

## LongMemEval

### Overview

LongMemEval is a comprehensive benchmark designed to evaluate long-term memory capabilities in conversational AI systems. It tests the system's ability to retrieve and reason about information across multiple conversation sessions.

**Explore the Dataset:** You can explore the LongMemEval dataset using the [LongMemEval Inspector](https://nicoloboschi.github.io/longmemeval-inspector/inspector.html).

### State-of-the-Art Comparison

The table below shows performance across different memory systems on the LongMemEval benchmark:

| Method | Single-session Preference | Single-session Assistant | Temporal Reasoning | Multi-session | Knowledge Update | Single-session User | Overall    |
|--------|---------------------------|--------------------------|--------------------|---------------|------------------|--------------------|------------|
| Zep gpt-4o-mini | 53.30%                    | 75.00% | 54.10%             | 47.40%        | 74.40%           | 92.90% | 63.80%     |
| Zep gpt-4o | 56.70%                    | 80.40% | 62.40%             | 57.90%        | 83.30%           | 92.90% | 71.00%     |
| Supermemory (gemini-3-pro) | 70.00%                    | 96.40% | 76.70%             | 71.40%        | 84.60%           | 97.10% | 85.86%     |
| Mastra gpt-4o (top_k=20) | 46.70%                    | 100.00% | 75.20%             | 76.70%        | 88.40%           | 97.10% | 80.05%     |
| **Hindsight** | **96.70%**                | **100.00%** | **91.7%**          | **98.50%**    | **94.9%**        | **100.00%** | **96.40%** |

**Key Highlights:**
- Hindsight achieves the highest overall accuracy at 96.40%
- Exceptional multi-session performance (98.50%), significantly outperforming other systems
- Perfect score (100.00%) on single-session user and assistant memory tasks
- Strong temporal reasoning capabilities (91.70%) and high knowledge update accuracy (94.90%)


### Benchmark Execution Details

#### Performance Metrics

**Latency**

Latency performance respected Hindsight's standard latency specifications. For detailed latency benchmarks and performance characteristics, see the [Hindsight documentation](https://github.com/vectorize-io/hindsight).

**Cost**

Hindsight uses **only open source models** for all memory operations. The benchmark execution costs are extremely competitive:

**Models Used:**
- **Answer Generator:** `openai/gpt-oss-20b` (via Groq)
- **Judge:** `openai/gpt-oss-120b` (via Groq)
- **Memory 'Retain' Pipeline:** `openai/gpt-oss-20b` (via Groq)

**Token Usage:**
- Input tokens: ~128M tokens
- Output tokens: ~25M tokens
- Groq pricing: $0.075 per 1M input tokens, $0.30 per 1M output tokens

**Total Cost Breakdown:**
- Input cost: 128M × $0.075/1M = **$9.60**
- Output cost: 25M × $0.30/1M = **$7.50**
- **Total Hindsight cost: $17.10**

**Per-Question Cost:**
- Average cost: $17.10 / 500 questions = **$0.034 per question**

**Note:** This cost represents **only the Hindsight memory system operations** (retain pipeline). It does not include the costs of the answer generator and judge models used for benchmark evaluation.

This demonstrates that Hindsight achieves state-of-the-art performance (96.40% accuracy) at a fraction of the cost of proprietary model-based systems, thanks to its exclusive use of open source models.

**Note on Cost Efficiency:**

The exceptionally low costs are achieved through:
1. **Sophisticated token reduction techniques** in the Retain pipeline that minimize LLM usage during memory storage
2. **LLM-free memory recalls** - The memory architecture does not use LLMs for retrieving memories, enabling **unlimited recall operations at zero LLM cost**

This means you can read and recall memories forever without incurring any LLM costs, making Hindsight extremely cost-effective for production deployments with high recall volumes.

**Infrastructure:**

The benchmark was executed on a **local MacBook** using a **local PostgreSQL database (pg0)**, demonstrating that Hindsight achieves exceptional performance without requiring specialized cloud infrastructure or expensive hardware.

#### System Configuration

These results were generated using the **first version of Hindsight**. The infrastructure and hardware configuration are not significant variables in performance - the key factor is the memory system architecture and model selection.

### Reproducibility

To reproduce these results, visit the main Hindsight repository:

**[github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)**

Follow the benchmark instructions in the repository documentation.

---

## LoComo Benchmark Results

### Overview

LoComo (Long Conversation Memory) is a benchmark designed to test memory systems on long, multi-turn conversations with questions requiring recall of specific details from earlier in the dialogue.

### Current Results

**Overall Performance:**
- **Accuracy:** 76.85%
- **Total Questions:** 1,542
- **Correct Answers:** 1,185
- **Conversations:** 10

**Note:** We skipped the **Adversarial category** as it is almost impossible to evaluate reliably due to the subjective and ambiguous nature of the questions in that category.

### Important Note on Benchmark Validity

While Hindsight achieves solid performance on LoComo, **we do not consider this benchmark to be a reliable indicator of memory system quality** due to significant flaws in the dataset design and evaluation methodology.

**Known Issues with LoComo:**

1. **Ambiguous Questions and Answers** - Many questions have multiple valid interpretations, and the "correct" answers are often subjective or incomplete
2. **Flawed Ground Truth** - The dataset contains numerous labeling errors and inconsistencies in what is marked as correct
3. **Limited Evaluation Scope** - The benchmark focuses on narrow recall tasks that don't reflect real-world memory system requirements
4. **Dataset Quality Issues** - Poor conversation design and unrealistic dialogue patterns

**References:**

- [Discussion of LoComo limitations in memory benchmarking literature](TODO: Add specific reference)
- [Analysis of conversational memory benchmark design flaws](TODO: Add specific reference)
- [Critique of long-context evaluation methodologies](TODO: Add specific reference)

For these reasons, we recommend focusing on **LongMemEval** as a more reliable indicator of memory system performance. LongMemEval provides better-quality ground truth, more realistic conversation scenarios, and a broader evaluation of memory capabilities.

### Reproducibility

To reproduce these results, visit the main Hindsight repository:

**[github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)**

Follow the LoComo benchmark instructions in the repository documentation.

---

## Exploring Results

To visualize the benchmark results, run:

```bash
./start-visualizer.sh
```

Then open http://localhost:9998 in your browser.
