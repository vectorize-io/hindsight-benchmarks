import type { Metadata } from 'next'
import { loadLeaderboardData } from '@/lib/leaderboard'
import LeaderboardTable from '@/components/leaderboard/LeaderboardTable'

export const metadata: Metadata = {
  title: 'Retain Leaderboard — Best LLMs for Fact Extraction',
  description:
    'Ranked LLMs for the Hindsight retain() operation — compare extraction quality, efficiency, JSON conformance, speed, and cost across models from OpenAI, Google, Groq, and more.',
}

export default function LeaderboardPage() {
  const allModels = loadLeaderboardData()

  // Only models actually measured on this benchmark appear here; everything
  // else lives on the legacy leaderboard until it is re-run.
  const viableModels = allModels.filter(m => m.qualityResult && m.qualityResult.total > 0)

  return (
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-heading font-bold text-foreground mb-2 gradient-primary-text">
          Retain Leaderboard
        </h1>
        <p className="text-lg text-muted-foreground">
          Which model should I use for <code className="px-1.5 py-0.5 bg-secondary rounded text-sm font-mono">retain()</code> and observation consolidation?
        </p>
      </div>

      <div className="mb-6 flex items-center gap-3 rounded-lg border border-border bg-secondary/30 px-4 py-3 text-sm text-muted-foreground">
        <span>
          Results from the previous LoComo-based methodology are on the{' '}
          <a href="/leaderboard/retain-legacy" className="font-medium text-foreground underline underline-offset-4 hover:text-primary transition-colors">
            legacy leaderboard
          </a>.
        </span>
        <span className="mx-1 text-border">|</span>
        <span>Want another model here, or spotted a deployment issue?</span>
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
        <LeaderboardTable models={viableModels} type="mixed" />
      ) : (
        <div className="border border-border rounded-xl p-12 text-center bg-card">
          <h3 className="text-xl font-heading text-foreground mb-2">No Benchmark Results</h3>
          <p className="text-muted-foreground">
            Run the benchmark runner to generate model performance data.
          </p>
        </div>
      )}

      {/* About Section */}
      <div className="mt-12 pt-8 border-t border-border">
        <h2 className="text-2xl font-heading font-bold text-foreground mb-6">About This Benchmark</h2>
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
                  Extraction accuracy on a frozen subset of the <strong className="text-foreground">BEAM</strong> long-term
                  memory benchmark (ICLR 2026): 4 conversations from the 128K-token tier, 80 questions tagged across
                  ten ability categories (temporal reasoning, knowledge update, abstention, and others). Hover the
                  Quality cell for a model&apos;s per-ability breakdown.
                  The model under test powers the <strong className="text-foreground">retain step</strong> (fact extraction and schema structuring).
                  The answer context is built from the <strong className="text-foreground">extracted facts only</strong>.
                  Source chunks are excluded on purpose: they contain the raw conversation, and including them lets a
                  weak extractor score like a strong one. Production recall does return chunks, so this setting is
                  harder than production; the number measures extraction quality, not end-user accuracy.
                  Answer generation and judging use a fixed <strong className="text-foreground">gemini-3.7-flash</strong>,
                  which is also a ranked model on this board.
                </td>
              </tr>
              <tr className="align-top bg-secondary/20">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Efficiency</td>
                <td className="px-6 py-4 whitespace-nowrap">accuracy / stored tokens</td>
                <td className="px-6 py-4">
                  Quality accuracy divided by the tokens the model wrote into memory during ingestion
                  (accuracy points per 1,000 stored fact tokens). This counters verbose extraction: a model that
                  copies the conversation near-verbatim into its facts would score well on facts-only accuracy but
                  pays for the footprint here. Stored tokens also drive real recall and storage cost.
                </td>
              </tr>
              <tr className="align-top">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Speed</td>
                <td className="px-6 py-4 whitespace-nowrap">latency (s) · tok/s</td>
                <td className="px-6 py-4">
                  <strong className="text-foreground">Mean</strong> end-to-end latency per request (arithmetic average across all successful requests)
                  and output throughput (tokens/second), measured during the fact-extraction benchmark.
                  Results will vary depending on your network conditions,
                  geographic proximity to the provider&apos;s servers, and server-side load at the time of testing.
                  <br /><br />
                  <span className="text-amber-400 font-medium">Note:</span> this benchmark does <strong className="text-foreground">not</strong> enforce
                  or simulate rate limits. Actual throughput may be lower in production depending on your subscription
                  tier and the provider&apos;s rate-limiting policies.
                </td>
              </tr>
              <tr className="align-top bg-secondary/20">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Cost</td>
                <td className="px-6 py-4 whitespace-nowrap">$ input / $ output per 1M tokens</td>
                <td className="px-6 py-4">
                  Published list prices (USD per million tokens) for input and output tokens, as advertised
                  by each provider at the time of testing. Prices may have changed since then.
                  Subscription-based models are scored separately on value relative to their monthly fee.
                  Local models have no per-token cost and always score 100 on this dimension.
                </td>
              </tr>
              <tr className="align-top">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">JSON Conformance</td>
                <td className="px-6 py-4 whitespace-nowrap">success / total tests</td>
                <td className="px-6 py-4">
                  The fraction of fact-extraction requests that returned valid JSON conforming to the
                  required schema. A failed request is one that timed out, returned an HTTP error, or
                  produced malformed / schema-invalid JSON. The canonical suite is 50 extraction tests
                  at concurrency 4; rows measured under older conditions show their own test count and
                  are being re-run.
                </td>
              </tr>
              <tr className="align-top bg-secondary/20">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Total Score</td>
                <td className="px-6 py-4 whitespace-nowrap">0 – 100</td>
                <td className="px-6 py-4">
                  Weighted composite: <strong className="text-foreground">Quality 60%</strong> + <strong className="text-foreground">Efficiency 20%</strong> + <strong className="text-foreground">Speed 10%</strong> + <strong className="text-foreground">JSON Conformance 5%</strong> + <strong className="text-foreground">Cost 5%</strong>.
                  Each dimension is normalised to a 0–100 scale before weighting. Quality maps accuracy onto that
                  scale with fixed anchors, <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">25% → 0</code>
                  and <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">65% → 100</code>,
                  so accuracy differences carry the weight the raw range would compress. Speed uses the formula
                  <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">100 × 10 / (10 + latency_s)</code>
                  so a 10-second response scores 50; Cost uses
                  <code className="mx-1 px-1.5 py-0.5 bg-secondary rounded text-xs font-mono">100 × 0.001 / (0.001 + cost_per_req)</code>
                  so only genuinely free models approach 100.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </main>
  )
}
