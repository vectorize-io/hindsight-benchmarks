# Hindsight Benchmark Results

This repository contains benchmark results for the Hindsight memory system and visualization tools to inspect the results.

Explore the results yourself on the [Benchmarks Visualizer](https://benchmarks.hindsight.vectorize.io/)

## LongMemEval

### Overview

LongMemEval is a comprehensive benchmark designed to evaluate long-term memory capabilities in conversational AI systems. It tests the system's ability to retrieve and reason about information across multiple conversation sessions.

**Explore the Dataset:** You can explore the LongMemEval dataset using the [LongMemEval Inspector](https://nicoloboschi.github.io/longmemeval-inspector/inspector.html).

### State-of-the-Art Comparison

The table below shows performance across different memory systems on the LongMemEval benchmark (S setting, 500 questions):

| Method | Single-session User | Single-session Assistant | Single-session Preference | Knowledge Update | Temporal Reasoning | Multi-session | Overall |
|--------|---------------------|--------------------------|---------------------------|------------------|--------------------|---------------|---------|
| Full-context (GPT-4o) | 81.4% | 94.6% | 20.0% | 78.2% | 45.1% | 44.3% | 60.2% |
| Full-context (OSS-20B) | 38.6% | 80.4% | 20.0% | 60.3% | 31.6% | 21.1% | 39.0% |
| Zep (GPT-4o) | 92.9% | 80.4% | 56.7% | 83.3% | 62.4% | 57.9% | 71.2% |
| Supermemory (GPT-4o) | 97.1% | 96.4% | 70.0% | 88.5% | 76.7% | 71.4% | 81.6% |
| Supermemory (GPT-5) | 97.1% | 100.0% | 76.7% | 87.2% | 81.2% | 75.2% | 84.6% |
| Supermemory (Gemini-3) | 98.6% | 98.2% | 70.0% | 89.7% | 82.0% | 76.7% | 85.2% |
| Hindsight (OSS-20B) | 95.7% | 94.6% | 66.7% | 84.6% | 79.7% | 79.7% | 83.6% |
| Hindsight (OSS-120B) | **100.0%** | 98.2% | **86.7%** | 92.3% | 85.7% | 81.2% | 89.0% |
| **Hindsight (Gemini-3)** | 97.1% | 96.4% | 80.0% | **94.9%** | **91.0%** | **87.2%** | **91.4%** |

**Key Highlights:**
- **Hindsight with Gemini-3 Pro achieves 91.4% overall accuracy**, the best result across all systems and model backbones
- **Hindsight with OSS-120B achieves 89.0%**, outperforming Supermemory with GPT-4o (81.6%) and GPT-5 (84.6%)
- **+44.6 percentage point improvement**: Hindsight with OSS-20B (83.6%) vs Full-context OSS-20B baseline (39.0%) demonstrates that the memory architecture, not model size, drives performance
- The largest gains appear in long-horizon categories: multi-session improves from 21.1% to 79.7%, temporal reasoning from 31.6% to 79.7%
- Even with a smaller open-source 20B model, Hindsight surpasses Full-context GPT-4o (60.2%) and matches Supermemory+GPT-4o (81.6%)

**Cost Efficiency:** Exceptionally low costs achieved through sophisticated token reduction techniques in the Retain pipeline and **LLM-free memory recalls** - retrieving memories incurs zero LLM cost, enabling unlimited recall operations in production.

**Infrastructure:** Local MacBook with PostgreSQL - no specialized cloud infrastructure required

### Reproducibility

To reproduce these results, visit the main Hindsight repository:

**[github.com/vectorize-io/hindsight](https://github.com/vectorize-io/hindsight)**

Follow the benchmark instructions in the repository documentation.

---

## LoComo Benchmark Results

### Overview

LoComo (Long Conversation Memory) is a benchmark designed to test memory systems on long, multi-turn conversations with questions requiring recall of specific details from earlier in the dialogue.

### State-of-the-Art Comparison

The table below shows accuracy (%) by question type and overall for prior memory systems and Hindsight with different backbone models:

| Method | Single-Hop | Multi-Hop | Open Domain | Temporal | Overall |
|--------|------------|-----------|-------------|----------|---------|
| Backboard | 89.36 | 75.00 | 91.20 | 91.90 | 90.00 |
| Memobase (v0.0.37) | 70.92 | 46.88 | 77.17 | 85.05 | 75.78 |
| Zep | 74.11 | 66.04 | 67.71 | 79.79 | 75.14 |
| Mem0-Graph | 65.71 | 47.19 | 75.71 | 58.13 | 68.44 |
| Mem0 | 67.13 | 51.15 | 72.93 | 55.51 | 66.88 |
| LangMem | 62.23 | 47.92 | 71.12 | 23.43 | 58.10 |
| OpenAI | 63.79 | 42.92 | 62.29 | 21.71 | 52.90 |
| Hindsight (OSS-20B) | 74.11 | 64.58 | 90.96 | 76.32 | 83.18 |
| Hindsight (OSS-120B) | 76.79 | 62.50 | 93.68 | 79.44 | 85.67 |
| **Hindsight (Gemini-3)** | **86.17** | **70.83** | **95.12** | **83.80** | **89.61** |

**Key Highlights:**
- Across all backbone sizes, Hindsight consistently outperforms prior open memory systems such as Memobase, Zep, Mem0, and LangMem
- **Hindsight raises overall accuracy from 75.78% (Memobase) to 83.18%** with OSS-20B and **85.67%** with OSS-120B
- **Hindsight with Gemini-3 attains 89.61% overall accuracy** and the highest Open Domain score (95.12%), closely matching Backboard's 90.00% overall performance
- These results demonstrate that the gains from Hindsight's memory architecture on LongMemEval transfer to realistic, multi-session human conversations

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

---

## Exploring Results

To visualize the benchmark results:

```bash
cd visualizer
npm install
npm run dev
```

Then open http://localhost:9998 in your browser.

The visualizer provides:
- 📊 Interactive benchmark overview with category breakdowns
- 🔍 Advanced filtering (by category, correctness, item ID)
- 📝 Detailed question-level analysis with reasoning and retrieved memories
- 🎯 Beautiful, responsive UI built with Next.js and Tailwind CSS

For deployment options and more details, see [visualizer/README.md](./visualizer/README.md).
