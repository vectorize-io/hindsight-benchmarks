// SDE-Bench's chart, shared by the coding-agents leaderboard and the Benchmarks page: per agent, no
// memory and then Hindsight (an arrow), fixed first try against cost per bug.
import { SCORE, logScore, type CodingRow, type Row } from '@/lib/data'

export const AGENT_ICON: Record<string, string> = {
  'claude-code': '/logos/harness/claude-code.png',
  codex: '/logos/harness/codex.svg',
  opencode: '/logos/harness/opencode.png',
}

// The operation boards' formula with fixed first try as the quality. Cost and wall time share the 30%
// the operation boards give to cost, each on the same log scale.
export function codingScore(rows: CodingRow[]) {
  const cheapest = Math.min(...rows.map((r) => r.cost_usd))
  const fastest = Math.min(...rows.map((r) => r.wall_s))
  return (r: CodingRow) => SCORE.quality * r.first_try + (SCORE.cost / 2) * (logScore(cheapest, r.cost_usd) + logScore(fastest, r.wall_s))
}

export function codingPoints(rows: CodingRow[]): Row[] {
  const score = codingScore(rows)
  // The chart: per agent, no memory, then Hindsight (arrow). The older gemini-2.5-flash-lite runs stay in the table only.
  const CHAIN = ['vanilla', 'hindsight-gemma']
  const points: Row[] = rows.filter((r) => CHAIN.includes(r.memory)).map((r) => ({
    id: r.agent + r.memory,
    model: r.agent,
    provider: r.model,
    icon: AGENT_ICON[r.agent],
    quality: r.first_try,
    score: score(r),
    cost: r.cost_usd,
    callsPerTask: 0,
    tokensPerTask: { input: 0, output: 0 },
    categories: {},
    split:
      r.memory === 'vanilla'
        ? undefined
        : { decisionModel: r.memory_models.join(' + '), llm: { calls: 0, input: 0, output: 0, cost: 0 }, decision: { calls: 0, input: 0, output: 0, cost: 0 } },
    extra: { Corrections: `${r.corrections.toFixed(2)} per bug`, Time: `${Math.round(r.wall_s)} s per bug` },
  }))
  return points
}

export const CODING_SCATTER_TEXT = {
  noun: 'agents',
  x: 'Cost per bug',
  unit: 'bug',
  quality: 'Fixed first try',
  pair: 'The same agent with Hindsight',
  costNote: 'Hindsight runs on gemma-4-26b-a4b here. Cost per bug is the agent’s tokens plus Hindsight’s own memory bill, averaged over the repeats.',
}
