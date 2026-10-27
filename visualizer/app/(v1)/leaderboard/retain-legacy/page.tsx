import type { Metadata } from 'next'
import Link from 'next/link'
import { loadLeaderboardData } from '@/lib/leaderboard'
import LeaderboardTable from '@/components/leaderboard/LeaderboardTable'

export const metadata: Metadata = {
  title: 'Retain Leaderboard (Legacy, LoComo)',
  description:
    'The previous LoComo-based retain() leaderboard, kept for reference. The current leaderboard uses the BEAM benchmark with a facts-only context.',
}

export default function LegacyLeaderboardPage() {
  const allModels = loadLeaderboardData()

  // Only models that actually have a LoComo score belong here. Matching on the
  // retain run instead would pull in every model benchmarked on BEAM, whose
  // numbers are not comparable to this board.
  const viableModels = allModels.filter(m =>
    m.legacyQualityResult && m.legacyQualityResult.total > 0
  )

  return (
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-4xl font-heading font-bold text-foreground mb-2 gradient-primary-text">
          Retain Leaderboard <span className="text-2xl align-middle text-muted-foreground">(legacy)</span>
        </h1>
        <p className="text-lg text-muted-foreground">
          The previous LoComo-based ranking, kept for reference while models are re-run on the current benchmark.
        </p>
      </div>

      <div className="mb-6 flex items-center gap-3 rounded-lg border border-border bg-secondary/30 px-4 py-3 text-sm text-muted-foreground">
        <span>These results use the retired LoComo methodology.</span>
        <Link
          href="/leaderboard/retain"
          className="font-medium text-foreground underline underline-offset-4 hover:text-primary transition-colors"
        >
          View the current leaderboard
        </Link>
      </div>

      {viableModels.length > 0 ? (
        <LeaderboardTable models={viableModels} type="mixed" variant="retain-legacy" />
      ) : (
        <div className="border border-border rounded-xl p-12 text-center bg-card">
          <h3 className="text-xl font-heading text-foreground mb-2">No Benchmark Results</h3>
          <p className="text-muted-foreground">No legacy results available.</p>
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
                  The model under test powers the <strong className="text-foreground">retain step</strong> (fact extraction and schema structuring).
                  Recall and answer generation use a fixed <strong className="text-foreground">gemini-2.5-flash-lite</strong> baseline,
                  and answers are judged by <strong className="text-foreground">gemini-2.5-flash-lite</strong>.
                  The score represents the percentage of questions answered correctly out of 242, including adversarial
                  questions that test whether the model hallucinates information that was never mentioned.
                </td>
              </tr>
              <tr className="align-top bg-secondary/20">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Speed</td>
                <td className="px-6 py-4 whitespace-nowrap">latency (s) · tok/s</td>
                <td className="px-6 py-4">
                  <strong className="text-foreground">Mean</strong> end-to-end latency per request (arithmetic average across all successful requests)
                  and output throughput (tokens/second), measured during the fact-extraction benchmark.
                  Tests were run on a <strong className="text-foreground">local MacBook with a 1 Gbps internet
                  connection from Europe</strong>. Results will vary depending on your network conditions,
                  geographic proximity to the provider&apos;s servers, and server-side load at the time of testing.
                </td>
              </tr>
              <tr className="align-top">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Cost</td>
                <td className="px-6 py-4 whitespace-nowrap">$ input / $ output per 1M tokens</td>
                <td className="px-6 py-4">
                  Published list prices (USD per million tokens) for input and output tokens, as advertised
                  by each provider at the time of testing. Prices may have changed since then.
                  Local models have no per-token cost and always score 100 on this dimension.
                </td>
              </tr>
              <tr className="align-top bg-secondary/20">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Reliability</td>
                <td className="px-6 py-4 whitespace-nowrap">success / total tests</td>
                <td className="px-6 py-4">
                  The fraction of fact-extraction requests that returned valid JSON conforming to the
                  required schema. A failed request is one that timed out, returned an HTTP error, or
                  produced malformed / schema-invalid JSON.
                </td>
              </tr>
              <tr className="align-top">
                <td className="px-6 py-4 font-semibold text-foreground whitespace-nowrap">Total Score</td>
                <td className="px-6 py-4 whitespace-nowrap">0 – 100</td>
                <td className="px-6 py-4">
                  Weighted composite: <strong className="text-foreground">Quality 40%</strong> + <strong className="text-foreground">Speed 25%</strong> + <strong className="text-foreground">Cost 20%</strong> + <strong className="text-foreground">Reliability 15%</strong>.
                  Each dimension is normalised to a 0–100 scale before weighting.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </main>
  )
}
