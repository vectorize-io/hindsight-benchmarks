import { loadRerankerData } from '@/lib/reranker'
import RerankerTable from '@/components/leaderboard/RerankerTable'

export default function RerankerLeaderboardPage() {
  const rerankers = loadRerankerData()

  return (
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-heading font-bold text-foreground mb-2 gradient-primary-text">
          Reranker Leaderboard
        </h1>
        <p className="text-lg text-muted-foreground">
          Which reranker surfaces the most relevant facts at the top of{' '}
          <code className="px-1.5 py-0.5 bg-secondary rounded text-sm font-mono">recall()</code> results?
        </p>
      </div>

      <div className="mb-6 flex items-center gap-3 rounded-lg border border-border bg-secondary/30 px-4 py-3 text-sm text-muted-foreground">
        <span>Want to see another reranker here?</span>
        <a
          href="https://github.com/vectorize-io/hindsight-benchmarks/issues"
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium text-foreground underline underline-offset-4 hover:text-primary transition-colors"
        >
          Open a GitHub issue
        </a>
      </div>

      {rerankers.length > 0 ? (
        <RerankerTable rerankers={rerankers} />
      ) : (
        <div className="border border-border rounded-xl p-12 text-center bg-card">
          <h3 className="text-xl font-heading text-foreground mb-2">No Reranker Benchmark Results</h3>
          <p className="text-muted-foreground">
            Run{' '}
            <code className="px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">
              uv run python3 -u run_all_reranker.py
            </code>{' '}
            to generate reranker benchmark data.
          </p>
        </div>
      )}

      {/* About Section */}
      <div className="mt-12 pt-8 border-t border-border">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-heading font-bold text-foreground">About This Benchmark</h2>
        </div>
        <div className="border border-border rounded-lg overflow-hidden bg-card">
          <table className="min-w-full divide-y divide-border">
            <thead className="bg-secondary/50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider w-32">Metric</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider w-40">Value shown</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider">How it is measured</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border text-sm text-muted-foreground">
              <tr className="align-top">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">MRR</td>
                <td className="px-6 py-4 whitespace-nowrap">0 – 1</td>
                <td className="px-6 py-4">
                  <strong className="text-foreground">Mean Reciprocal Rank</strong> — for each question the
                  rank of the first relevant fact in the <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">recall()</code> results
                  is recorded, then averaged as 1/rank across all questions.
                  Higher is better; 1.0 means the relevant fact is always the top result.
                </td>
              </tr>
              <tr className="align-top bg-secondary/20">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Total Score</td>
                <td className="px-6 py-4 whitespace-nowrap">0 – 100</td>
                <td className="px-6 py-4">
                  Weighted composite: <strong className="text-foreground">MRR 70%</strong> + <strong className="text-foreground">Speed 15%</strong> + <strong className="text-foreground">Cost 15%</strong>.
                  MRR is scaled directly to 0–100 (MRR × 100).
                  Speed uses <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">100 × 1 / (1 + latency_s)</code> (1s reference).
                  Cost uses <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">100 × 0.001 / (0.001 + price_per_search)</code> — free rerankers score 100, Cohere ($0.002/search) scores ~33.
                  R@K metrics are shown for reference but not included in the total score, since MRR already captures ranking quality.
                </td>
              </tr>
              <tr className="align-top">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Cost</td>
                <td className="px-6 py-4 whitespace-nowrap">$ per recall()</td>
                <td className="px-6 py-4">
                  Published list price per reranker API call (search).
                  Free and local rerankers score 100. Cohere charges <strong className="text-foreground">$2.00 per 1,000 searches</strong> ($0.002/call).
                  Prices may have changed since testing.
                </td>
              </tr>
              <tr className="align-top bg-secondary/20">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">R@K</td>
                <td className="px-6 py-4 whitespace-nowrap">% of questions</td>
                <td className="px-6 py-4">
                  <strong className="text-foreground">Recall at K</strong> — the fraction of questions
                  where at least one relevant fact appears in the top-K results returned by{' '}
                  <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">recall()</code>.
                  R@1 asks: <em>"is the very first result relevant?"</em> — the reranker must put the right fact
                  in position 1 to score a point. R@5 asks: <em>"is any of the top 5 results relevant?"</em> —
                  giving the reranker more chances to surface the right fact. A good reranker scores
                  high on all three; a poor one only improves when K is large.
                </td>
              </tr>
              <tr className="align-top">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Latency</td>
                <td className="px-6 py-4 whitespace-nowrap">avg s/recall</td>
                <td className="px-6 py-4">
                  Mean wall-clock time per <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">recall()</code> call,
                  including reranking. Local models (MiniLM, FlashRank) use <strong className="text-foreground">sentence-transformers</strong> cross-encoders
                  running on CPU inside Docker — latency would be significantly lower on a machine with GPU support.
                </td>
              </tr>
              <tr className="align-top bg-secondary/20">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Setup</td>
                <td className="px-6 py-4 whitespace-nowrap">—</td>
                <td className="px-6 py-4">
                  Benchmark uses the <strong className="text-foreground">LoComo</strong> long-term conversation
                  dataset (conv-43, 165 questions with annotated ground truth).
                  All rerankers share the same ingested bank (retained with{' '}
                  <strong className="text-foreground">gemini-2.5-flash</strong>).
                  Candidates per recall: <strong className="text-foreground">300</strong> (budget=mid).
                  Ground truth was annotated by having{' '}
                  <strong className="text-foreground">gemini-2.5-flash</strong> identify relevant facts
                  from a high-budget recall on the same bank.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Interpretation notes */}
      <div className="mt-8 space-y-4">
        <h2 className="text-xl font-heading font-bold text-foreground">Interpreting the results</h2>

        <div className="rounded-lg border border-border bg-card p-5 text-sm text-muted-foreground space-y-2">
          <p className="font-semibold text-foreground">Why does a small local model (MiniLM-L6) rank at the top?</p>
          <p>
            A few factors combine here. First, the <strong className="text-foreground">ground truth has implicit lexical bias</strong>: relevant facts
            were annotated from a high-budget RRF recall (RRF is partly BM25-based, favouring lexical overlap), then labelled by an LLM.
            MiniLM-L6 is a cross-encoder trained on MS MARCO QA pairs — short factual sentences with strong lexical signal — which closely
            matches the style of LoComo personal memory facts (<em>"John plays basketball"</em>, <em>"Tim collects vinyl"</em>).
          </p>
          <p>
            Second, commercial models like Cohere and Voyage are optimised for large-scale document retrieval (legal, financial, technical corpora).
            They may be over-calibrated for longer, denser texts and underperform on short conversational facts at a pool of only 300 candidates.
          </p>
          <p>
            Third, with only 165 questions from a single conversation the <strong className="text-foreground">confidence intervals are wide</strong> —
            a handful of rankings on common questions can swing MRR by several points.
            The gap between MiniLM-L6 (0.788), Cohere v3 (0.784), and FlashRank (0.707) is likely not statistically significant.
          </p>
          <p className="text-xs pt-1 border-t border-border">
            <strong className="text-foreground">Bottom line:</strong> these numbers reflect performance on one specific task —
            conversational personal-memory retrieval with short facts. They should not be read as a general reranker ranking.
            For document retrieval or technical corpora, commercial models will likely perform better.
          </p>
        </div>

        <div className="rounded-lg border border-border bg-card p-5 text-sm text-muted-foreground space-y-2">
          <p className="font-semibold text-foreground">Why does Cohere v4 Fast score higher than v4 Pro?</p>
          <p>
            Counterintuitive but consistent with the domain mismatch above. <strong className="text-foreground">v4 Pro</strong> is optimised
            for complex, nuanced relevance judgements across long documents. On short conversational facts it may
            over-think the ranking and penalise simple lexical matches that are actually correct.
            <strong className="text-foreground"> v4 Fast</strong> uses a lighter scoring head that behaves more like a cross-encoder
            on short texts — closer to what this benchmark rewards.
            This pattern (smaller/faster variant winning on short-text tasks) is common in cross-encoder literature.
          </p>
        </div>

        <div className="rounded-lg border border-border bg-card p-5 text-sm text-muted-foreground space-y-2">
          <p className="font-semibold text-foreground">Why do Voyage models score lower than expected?</p>
          <p>
            Voyage Rerank 2 and 2-Lite are trained primarily on <strong className="text-foreground">passage and document retrieval</strong> tasks
            (BEIR, MTEB). The LoComo facts are much shorter and more conversational than their training distribution.
            Voyage models also apply strong semantic generalisation which can hurt precision when the relevant fact
            is nearly verbatim in the retrieved candidates — exactly what MRR@1 measures.
          </p>
        </div>
      </div>
    </main>
  )
}
