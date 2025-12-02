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

**Models:** Hindsight uses **only open source models** via Groq
- Memory 'Retain' Pipeline: `openai/gpt-oss-20b`
- Answer Generator: `openai/gpt-oss-20b`
- Judge: `openai/gpt-oss-120b`

**Cost:** $17.10 total ($0.034 per question)
- 128M input tokens × $0.075/1M = $9.60
- 25M output tokens × $0.30/1M = $7.50
- *Note: Covers only Hindsight operations (retain pipeline), excludes evaluation costs*

**Cost Efficiency:** Exceptionally low costs achieved through sophisticated token reduction techniques in the Retain pipeline and **LLM-free memory recalls** - retrieving memories incurs zero LLM cost, enabling unlimited recall operations in production.

**Infrastructure:** Local MacBook with PostgreSQL (pg0) - no specialized cloud infrastructure required

**Latency:** Respected Hindsight's standard specifications. See [documentation](https://github.com/vectorize-io/hindsight) for details.

**System:** First version of Hindsight. 

### Reproducibility

To reproduce these results, visit the main Hindsight repository:

**[github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)**

Follow the benchmark instructions in the repository documentation.

---

## LoComo Benchmark Results

### Overview

LoComo (Long Conversation Memory) is a benchmark designed to test memory systems on long, multi-turn conversations with questions requiring recall of specific details from earlier in the dialogue.

### State-of-the-Art Comparison

The table below shows performance across different memory systems on the LoComo benchmark:

| Method | Single Hop J ↑ | Multi-Hop J ↑ | Open Domain J ↑ | Temporal J ↑ | Overall |
|--------|----------------|---------------|-----------------|--------------|---------|
| A-Mem* | 39.79 | 18.85 | 54.05 | 31.08 | 48.38 |
| LangMem | 62.23 | 47.92 | 71.12 | 23.43 | 58.10 |
| Zep (Mem0 paper) | 61.70 | 41.35 | 76.60 | 49.31 | 65.99 |
| Zep (Zep Blog Post) | - | - | - | - | 75.14 |
| OpenAI | 63.79 | 42.92 | 62.29 | 21.71 | 52.90 |
| Mem0 | 67.13 | 51.15 | 72.93 | 55.51 | 66.88 |
| Mem0 w Graph | 65.71 | 47.19 | 75.71 | 58.13 | 68.44 |
| **Hindsight** | **76.90** | **69.50** | **86.30** | **59.40** | **79.61** |

**Key Highlights:**
- Hindsight achieves the highest overall accuracy at 79.61%
- Best Single Hop performance (76.90%), significantly outperforming all other systems
- Strong Multi-Hop reasoning (69.50%) and Open Domain performance (86.30%)
- Consistent performance across all categories

**Note:** We skipped the **Adversarial category** as it is almost impossible to evaluate reliably due to the subjective and ambiguous nature of the questions in that category.

### Important Note on Benchmark Validity

While Hindsight achieves solid performance on LoComo, **we do not consider this benchmark to be a reliable indicator of memory system quality** due to significant flaws in the dataset design and evaluation methodology.

**Known Issues with LoComo:**

1. **Missing and Flawed Ground Truth** - Some categories have missing ground truth answers, speaker attribution errors, and inconsistencies in what is marked as correct
2. **Ambiguous Questions** - Many questions have multiple valid interpretations and lack sufficient specificity to have a single correct answer
3. **Insufficient Challenge** - Conversations are too short (16k-26k tokens), fitting within modern LLM context windows, failing to genuinely test memory retrieval capabilities
4. **Limited Evaluation Scope** - Lacks critical tests for knowledge updates and temporal reasoning that are essential for real-world memory systems
5. **Data Quality Issues** - Multimodal errors (image references without descriptions), poor conversation design, and unrealistic dialogue patterns

**References:**

- [https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/]
- [https://www.kdjingpai.com/en/ai-zhinengtijiyiban/]

For these reasons, we recommend focusing on **LongMemEval** as a more reliable indicator of memory system performance. LongMemEval provides better-quality ground truth, more realistic conversation scenarios, and a broader evaluation of memory capabilities.

### Reproducibility

To reproduce these results, visit the main Hindsight repository:

**[github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)**

Follow the LoComo benchmark instructions in the repository documentation.

---

## Exploring Results

To visualize the benchmark results:

```bash
cd visualizer-web
npm install
npm run dev
```

Then open http://localhost:9998 in your browser.

The visualizer provides:
- 📊 Interactive benchmark overview with category breakdowns
- 🔍 Advanced filtering (by category, correctness, item ID)
- 📝 Detailed question-level analysis with reasoning and retrieved memories
- 🎯 Beautiful, responsive UI built with Next.js and Tailwind CSS

For deployment options and more details, see [visualizer-web/README.md](./visualizer-web/README.md).
