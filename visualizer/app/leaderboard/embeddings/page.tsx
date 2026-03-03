import { loadEmbeddingsData } from '@/lib/embeddings'
import EmbeddingsTable from '@/components/leaderboard/EmbeddingsTable'

export default function EmbeddingsLeaderboardPage() {
  const embeddings = loadEmbeddingsData()

  return (
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-heading font-bold text-foreground mb-2 gradient-primary-text">
          Embeddings Leaderboard
        </h1>
        <p className="text-lg text-muted-foreground">
          Which embedding model retrieves the most relevant facts from{' '}
          <code className="px-1.5 py-0.5 bg-secondary rounded text-sm font-mono">recall()</code>?
        </p>
      </div>

      <div className="mb-6 flex items-center gap-3 rounded-lg border border-border bg-secondary/30 px-4 py-3 text-sm text-muted-foreground">
        <span>Want to see another embedding model here?</span>
        <a
          href="https://github.com/vectorize-io/hindsight-benchmarks/issues"
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium text-foreground underline underline-offset-4 hover:text-primary transition-colors"
        >
          Open a GitHub issue
        </a>
      </div>

      {embeddings.length > 0 ? (
        <EmbeddingsTable embeddings={embeddings} />
      ) : (
        <div className="border border-border rounded-xl p-12 text-center bg-card">
          <h3 className="text-xl font-heading text-foreground mb-2">No Embeddings Benchmark Results</h3>
          <p className="text-muted-foreground">
            Run{' '}
            <code className="px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">
              uv run python3 -u run_all_embeddings.py
            </code>{' '}
            to generate embeddings benchmark data.
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
                  Cost uses <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">100 × 0.0001 / (0.0001 + price_per_query)</code>.
                  R@K metrics are shown for reference but not included in the total score.
                </td>
              </tr>
              <tr className="align-top">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">R@K</td>
                <td className="px-6 py-4 whitespace-nowrap">% of questions</td>
                <td className="px-6 py-4">
                  <strong className="text-foreground">Recall at K</strong> — the fraction of questions
                  where at least one relevant fact appears in the top-K results returned by{' '}
                  <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">recall()</code>.
                  R@1 asks: <em>&ldquo;is the very first result relevant?&rdquo;</em>
                  R@5 asks: <em>&ldquo;is any of the top 5 results relevant?&rdquo;</em>
                </td>
              </tr>
              <tr className="align-top bg-secondary/20">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Latency</td>
                <td className="px-6 py-4 whitespace-nowrap">avg s/recall</td>
                <td className="px-6 py-4">
                  Mean wall-clock time per <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">recall()</code> call,
                  including reranking. All models run with <strong className="text-foreground">MiniLM-L6</strong> cross-encoder reranker on CPU inside Docker — latency would be lower with GPU support.
                </td>
              </tr>
              <tr className="align-top">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Setup</td>
                <td className="px-6 py-4 whitespace-nowrap">—</td>
                <td className="px-6 py-4">
                  Benchmark uses the <strong className="text-foreground">LoComo</strong> long-term conversation
                  dataset (conv-43, 165 questions with annotated ground truth).
                  Each embedding model ingests the conversation into its own bank (separate volume, since dimensions are fixed at schema creation).
                  Fixed reranker: <strong className="text-foreground">MiniLM-L6</strong> (cross-encoder/ms-marco-MiniLM-L-6-v2) for all runs.
                  Candidates per recall: <strong className="text-foreground">300</strong> (budget=mid).
                  Ground truth is annotated <strong className="text-foreground">per model</strong> — each model gets its own GT file,
                  because the retain LLM is non-deterministic and phrases facts differently each run.
                  Annotation uses <strong className="text-foreground">gemini-2.5-flash</strong> to identify relevant facts
                  from a high-budget recall on each model&apos;s own bank.
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
          <p className="font-semibold text-foreground">Why does each embedding model need its own bank?</p>
          <p>
            Hindsight stores vectors in a PostgreSQL table with a fixed-width column set at bank creation time.
            The vector dimension is determined by the embedding model used during the first <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">retain()</code> call
            and cannot change afterwards. Switching embedding models requires a fresh database volume and
            re-ingesting the entire conversation from scratch.
          </p>
          <p className="text-xs pt-1 border-t border-border">
            <strong className="text-foreground">Practical implication:</strong> the ingestion latency is not counted in the benchmark results —
            only <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">recall()</code> latency is measured, which includes
            vector search + reranking. Ingestion is a one-time cost per model per conversation.
          </p>
        </div>

        <div className="rounded-lg border border-border bg-card p-5 text-sm text-muted-foreground space-y-2">
          <p className="font-semibold text-foreground">Why is the reranker fixed (MiniLM-L6)?</p>
          <p>
            To isolate the contribution of the embedding model. By holding the reranker constant across all runs,
            differences in MRR and Recall@K reflect only the quality of the embedding model&apos;s vector space —
            how well it represents semantic similarity for short conversational facts.
            The reranker is the top performer from the{' '}
            <a href="/leaderboard/reranker" className="text-foreground underline underline-offset-2 hover:text-primary transition-colors">
              Reranker Leaderboard
            </a>.
          </p>
        </div>
      </div>
    </main>
  )
}
