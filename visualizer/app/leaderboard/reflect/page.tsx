import { loadLeaderboardData } from '@/lib/leaderboard'
import LeaderboardTable from '@/components/leaderboard/LeaderboardTable'

export default function ReflectLeaderboardPage() {
  const allModels = loadLeaderboardData()

  // Only show models that are flagged for reflect benchmark
  const reflectModels = allModels.filter(m =>
    m.config.benchmarks?.includes('reflect')
  )
  const viableModels = reflectModels.filter(m =>
    m.reflectResult && m.reflectResult.total > 0
  )

  return (
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-heading font-bold text-foreground mb-2 gradient-primary-text">
          Reflect Leaderboard
        </h1>
        <p className="text-lg text-muted-foreground">
          Which model should I use for <code className="px-1.5 py-0.5 bg-secondary rounded text-sm font-mono">reflect()</code> operations?
        </p>
      </div>

      <div className="mb-6 flex items-center gap-3 rounded-lg border border-border bg-secondary/30 px-4 py-3 text-sm text-muted-foreground">
        <span>Want to see another model here?</span>
        <a
          href="https://github.com/vectorize-io/hindsight-benchmarks/issues"
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium text-foreground underline underline-offset-4 hover:text-primary transition-colors"
        >
          Open a GitHub issue
        </a>
      </div>

      {viableModels.length > 0 ? (
        <LeaderboardTable models={viableModels} type="mixed" variant="reflect" />
      ) : (
        <div className="border border-border rounded-xl p-12 text-center bg-card">
          <h3 className="text-xl font-heading text-foreground mb-2">No Reflect Benchmark Results</h3>
          <p className="text-muted-foreground">
            Run <code className="px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">uv run rag-benchmark reflect</code> to generate reflect benchmark data.
          </p>
        </div>
      )}

      {/* About Section */}
      <div className="mt-12 pt-8 border-t border-border">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-heading font-bold text-foreground">About This Benchmark</h2>
          <span className="text-xs text-muted-foreground font-mono bg-secondary px-2 py-1 rounded">Hindsight v0.4.13</span>
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
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Quality</td>
                <td className="px-6 py-4 whitespace-nowrap">% accuracy</td>
                <td className="px-6 py-4">
                  Accuracy on the <strong className="text-foreground">LoComo</strong> long-term conversation benchmark (conv-43, 242 questions).
                  Each question is answered using a single <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">reflect()</code> call
                  and judged by <strong className="text-foreground">gemini-2.5-flash-lite</strong>.
                  The retain/ingestion step always uses <strong className="text-foreground">gemini-2.5-flash</strong> as a constant baseline —
                  only the reflect step tests the model under evaluation.
                </td>
              </tr>
              <tr className="align-top bg-secondary/20">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Speed</td>
                <td className="px-6 py-4 whitespace-nowrap">avg latency (s/call)</td>
                <td className="px-6 py-4">
                  <strong className="text-foreground">Mean</strong> end-to-end wall-clock time per <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">reflect()</code> call
                  (arithmetic average across all 242 questions). This is <strong className="text-foreground">not</strong> a single LLM inference time —
                  each <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">reflect()</code> call runs a full agentic pipeline
                  internally with potentially several model calls.
                  The model under test powers the answer generation step; the retain model (gemini-2.5-flash) is fixed.
                  <br /><br />
                  Tests were run on a <strong className="text-foreground">local MacBook with a 1 Gbps internet connection from Europe</strong>.
                  Results will vary depending on your network conditions and server-side load.
                </td>
              </tr>
              <tr className="align-top">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Cost</td>
                <td className="px-6 py-4 whitespace-nowrap">$ input / $ output per 1M tokens</td>
                <td className="px-6 py-4">
                  Published list prices (USD per million tokens) as advertised by each provider.
                  Prices may have changed since testing.
                </td>
              </tr>
              <tr className="align-top bg-secondary/20">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Total Score</td>
                <td className="px-6 py-4 whitespace-nowrap">0 – 100</td>
                <td className="px-6 py-4">
                  Weighted composite: <strong className="text-foreground">Quality 60%</strong> + <strong className="text-foreground">Speed 25%</strong> + <strong className="text-foreground">Cost 15%</strong>.
                  Reliability is not included since the reflect endpoint does not involve schema conformance risk.
                  Speed uses <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">100 × 5 / (5 + latency_s)</code> (5s reference — calibrated for reflect() call durations);
                  Cost uses <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">100 × 0.001 / (0.001 + cost_per_req)</code>.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </main>
  )
}
