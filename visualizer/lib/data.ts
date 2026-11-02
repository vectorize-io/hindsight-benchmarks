// Leaderboard data: real runs of Hindsight's system evals, two passes per model.
// `hindsight-evals --export-site visualizer/lib/results.json` writes the results;
// `hindsight-evals --export-structure visualizer/lib/eval-structure.json` the categories.

import STRUCTURE from './eval-structure.json'
import RESULTS from './results.json'
import CODING from './coding-agents.json'
import CONVERSATIONS from './conversations.json'

export const RUN = {
  hindsightVersion: '1.2.0',
  judge: 'gemini-3.1-flash-lite',
  repeats: 2,
  date: '2026-11-01',
}

export type Part = 'llm' | 'embeddings' | 'reranker'

export const REFERENCE: Record<Part, { model: string; provider: string }> = {
  llm: { model: 'gemma-4-26b-a4b', provider: 'Google' },
  embeddings: { model: 'bge-small-en-v1.5', provider: 'BAAI' },
  reranker: { model: 'ms-marco-MiniLM-L-6-v2', provider: 'cross-encoder' },
}

export const PART_LABEL: Record<Part, string> = {
  llm: 'LLM',
  embeddings: 'Embeddings',
  reranker: 'Reranker',
}

export type CategoryResult = { passed: number; total: number; failed?: string[] } // failed: case ids

export type Row = {
  id: string
  model: string
  label?: string // shown instead of model where two rows share it
  provider: string
  icon?: string
  openWeights?: boolean
  tag?: 'reference' | 'baseline' | 'hindsight' | 'published'
  quality: number // % of cases passed
  score?: number // combined: see SCORE
  categories: Record<string, CategoryResult>
  cost: number // USD, all-in, in the board's costUnit
  callsPerTask: number // LLM calls
  decisionCallsPerTask?: number
  // Per case, for rows that also run a decision model: what each model did and cost.
  split?: { decisionModel: string; llm: Work; decision: Work }
  tokensPerTask: { input: number; output: number }
  extra?: Record<string, string> // more measures, shown in the chart's hover card
  source?: Source // where its price and run came from, shown in the expanded row
  from?: string // the chart's arrow to this row starts at this row id, instead of the same model alone
}

export type Source = {
  openrouter_id: string
  price_per_1m: { input: number; output: number; cached_input: number }
  decision_price_per_1m_input: number | null
  ran: string
  judge: string
  repeats: number
}

export type Work = { calls: number; input: number; output: number; cost: number }

export type Tab = {
  id: string
  label: string
  description: string
  rows: Row[]
  categories?: Board['categories'] // when the tab is scored on another set than its board
}

export type Board = {
  id: string
  group: 'summary' | 'operation' // the LLM summary sums up the operation boards
  about: string // what the board is for, in the setup box
  setup?: [string, string][] // what is held fixed, when not the eval reference setup
  title: string
  short: string
  description: string
  swaps: Part[]
  fixed: Part[]
  categories: { id: string; label: string; cases: number; about: string; caseIds: string[] }[]
  benchmarks?: string[] // rate/cost tests at this commit, not graded cases
  tabs: Tab[]
  costUnit: string
}

const round = (x: number, d = 2) => Math.round(x * 10 ** d) / 10 ** d
const label = (id: string) => (id.charAt(0).toUpperCase() + id.slice(1)).replace(/_/g, ' ').replace('Multi hop', 'Multi-hop')

// ── categories: generated from the evals at the pinned commit ──────────────
type Cat = { id: string; label: string; cases: number; caseIds: string[]; about: string }

const CATS: Record<string, Cat[]> = Object.fromEntries(
  Object.entries(STRUCTURE.operations).map(([op, o]) => [
    op,
    o.categories.map((c) => ({ id: c.id, label: label(c.id), cases: c.cases.length, caseIds: c.cases, about: c.about })),
  ]),
)

// ── results ────────────────────────────────────────────────────────────────
type OpResult = {
  passed: number
  total: number
  cost_usd: number | null
  calls: number
  input_tokens: number
  output_tokens: number
  decision_calls: number
  decision_input_tokens: number
  decision_cost_usd: number | null
  categories: Record<string, CategoryResult>
}
type Decider = { model: string; provider: string; onLlmBoards?: boolean }
type Result = Omit<(typeof RESULTS.rows)[number], 'decision' | 'operations' | 'source'> & { decision: Decider | null; source?: Source; operations: Record<string, OpResult> }
const RUNS = RESULTS.rows as Result[]

// Per case: the LLM's calls, tokens and cost, and the decision model's.
function split(o: OpResult, decisionModel: string): Row['split'] {
  const per = (v: number) => v / o.total
  const dCost = o.decision_cost_usd ?? 0
  return {
    decisionModel,
    llm: { calls: per(o.calls), input: per(o.input_tokens), output: per(o.output_tokens), cost: (1000 * ((o.cost_usd ?? 0) - dCost)) / o.total },
    decision: { calls: per(o.decision_calls), input: per(o.decision_input_tokens), output: 0, cost: (1000 * dCost) / o.total },
  }
}

// One row per model that ran `op` with no decision model, plus runs whose decision
// model is marked for the LLM boards.
function llmRows(op: string): Row[] {
  return RUNS.filter((r) => (!r.decision || r.decision.onLlmBoards) && r.operations[op]).map((r) => {
    const o = r.operations[op]
    return {
      id: r.decision ? `${r.model}+${r.decision.model}` : r.model,
      model: r.model,
      label: r.decision ? `${r.model} + ${r.decision.model}` : undefined,
      provider: r.provider,
      icon: r.icon,
      openWeights: r.openWeights,
      source: r.source,
      quality: round((100 * o.passed) / o.total, 1),
      // Every category is present, so the row's breakdown lists all of them.
      categories: Object.fromEntries(CATS[op].map((c) => [c.id, o.categories[c.id] ?? { passed: 0, total: c.cases }])),
      cost: round((1000 * (o.cost_usd ?? 0)) / o.total),
      callsPerTask: round(o.calls / o.total, 1),
      decisionCallsPerTask: r.decision ? round(o.decision_calls / o.total, 1) : undefined,
      split: r.decision ? split(o, r.decision.model) : undefined,
      tokensPerTask: { input: Math.round(o.input_tokens / o.total), output: Math.round(o.output_tokens / o.total) },
    }
  })
}

// ── boards ─────────────────────────────────────────────────────────────────
const cats = (op: string) => CATS[op].map(({ id, label, cases, about, caseIds }) => ({ id, label, cases, about, caseIds }))
const benchmarks = (op: string) => STRUCTURE.operations[op as keyof typeof STRUCTURE.operations]?.benchmarks ?? []

// ── LLM summary: the operation boards that use an LLM, summed up per LLM, no runs of its own ──
// Recall is left out: it calls no LLM, so it would score every LLM the same.
const OPS = [
  ['retain', 'Retain'],
  ['consolidation', 'Consolidation'],
  ['reflect', 'Reflect'],
  ['refresh_pages', 'Refresh pages'],
] as const

function summaryRows(): Row[] {
  return RUNS.filter((r) => !r.decision || r.decision.onLlmBoards).flatMap((r) => {
    // The decision model plays no part in retain: a row with one takes its LLM's own retain run.
    const own = (op: string) => r.operations[op] ?? (op === 'retain' ? RUNS.find((x) => x.model === r.model && !x.decision)?.operations.retain : undefined)
    const ops = OPS.map(([op]) => own(op))
    if (ops.some((o) => !o)) return []
    const done = ops as OpResult[]
    const sum = (f: (o: OpResult) => number) => done.reduce((t, o) => t + f(o), 0)
    const cases = sum((o) => o.total)
    return [{
      id: r.decision ? `${r.model}+${r.decision.model}` : r.model,
      model: r.model,
      label: r.decision ? `${r.model} + ${r.decision.model}` : undefined,
      provider: r.provider,
      icon: r.icon,
      openWeights: r.openWeights,
      source: r.source,
      // Each operation weighs the same, however many cases it has.
      quality: round(done.reduce((t, o) => t + (100 * o.passed) / o.total, 0) / done.length, 1),
      categories: Object.fromEntries(OPS.map(([op], i) => [op, { passed: done[i].passed, total: done[i].total }])),
      // What running every operation 1k times costs.
      cost: round(sum((o) => (1000 * (o.cost_usd ?? 0)) / o.total)),
      callsPerTask: round(sum((o) => o.calls) / cases, 1),
      decisionCallsPerTask: r.decision ? round(sum((o) => o.decision_calls) / cases, 1) : undefined,
      tokensPerTask: { input: Math.round(sum((o) => o.input_tokens) / cases), output: Math.round(sum((o) => o.output_tokens) / cases) },
      // Every operation's cases pooled, for the LLM / decision model split.
      split: r.decision
        ? split(
            {
              ...done[0],
              total: cases,
              calls: sum((o) => o.calls),
              input_tokens: sum((o) => o.input_tokens),
              output_tokens: sum((o) => o.output_tokens),
              decision_calls: sum((o) => o.decision_calls),
              decision_input_tokens: sum((o) => o.decision_input_tokens),
              cost_usd: sum((o) => o.cost_usd ?? 0),
              decision_cost_usd: sum((o) => o.decision_cost_usd ?? 0),
            },
            r.decision.model,
          )
        : undefined,
    }]
  })
}

export const BOARDS: Board[] = [
  {
    id: 'all-operations',
    group: 'summary',
    about: 'Pick one LLM for all of Hindsight. Each row is an LLM, alone or with a decision model, run through every operation that uses an LLM; its score sums them up.',
    title: 'All operations',
    short: 'One LLM behind every Hindsight operation, summed up.',
    description: 'Most setups run one LLM for everything, sometimes with a decision model. This board sums up the four operation boards that use an LLM: quality is the mean over them, cost is what running each of them costs. No runs of its own.',
    swaps: ['llm'],
    fixed: ['embeddings', 'reranker'],
    categories: OPS.map(([op, label]) => ({ id: op, label, cases: CATS[op].reduce((t, c) => t + c.cases, 0), about: `The ${label} board`, caseIds: [] })),
    costUnit: 'per 1k cases of each',
    tabs: [{ id: 'all', label: 'Every LLM operation', description: 'A row needs a run of every operation. A row with a decision model takes its LLM\'s retain run, since retain does not use one.', rows: summaryRows() }],
  },
  {
    id: 'retain',
    group: 'operation',
    about: 'Pick the LLM that turns chats, documents and code into memories: complete, attributed, dated, and nothing invented.',
    title: 'Retain',
    short: 'Turns raw input into memories.',
    description: 'Which LLM writes the best memories from chats, documents and code: complete, attributed, dated, and nothing invented.',
    swaps: ['llm'],
    fixed: ['embeddings', 'reranker'],
    categories: cats('retain'),
    benchmarks: benchmarks('retain'),
    costUnit: 'per 1k cases',
    tabs: [
      { id: 'default', label: 'Fact extraction', description: 'Default retain: one extraction call per chunk.', rows: llmRows('retain') },
    ],
  },
  {
    id: 'consolidation',
    group: 'operation',
    about: 'Pick the LLM that keeps observations current as new facts arrive: merges restatements, applies updates, keeps different things apart.',
    title: 'Consolidation',
    short: 'Merges facts into observations.',
    description: 'Which LLM keeps observations current: merges restatements, applies updates, and keeps different things apart, even in large noisy banks.',
    swaps: ['llm'],
    fixed: ['embeddings', 'reranker'],
    categories: cats('consolidation'),
    benchmarks: benchmarks('consolidation'),
    costUnit: 'per 1k cases',
    tabs: [
      { id: 'llm', label: 'LLM only', description: 'Every candidate observation goes to the LLM. Rows with + decision model also run one; open one to see what each model did and cost.', rows: llmRows('consolidation') },
    ],
  },
  {
    id: 'reflect',
    group: 'operation',
    about: 'Pick the LLM that answers questions from memory, planning its own searches and deciding when it has enough.',
    title: 'Reflect',
    short: 'Answers questions from memory.',
    description: 'Which LLM answers best from memory: uses the latest value, aggregates correctly, and says so when the answer is not there.',
    swaps: ['llm'],
    fixed: ['embeddings', 'reranker'],
    categories: cats('reflect'),
    benchmarks: benchmarks('reflect'),
    costUnit: 'per 1k cases',
    tabs: [
      { id: 'agent', label: 'Agent', description: 'The LLM plans its own searches and decides when it has enough. Rows with + decision model also run one; open one to see what each model did and cost.', rows: llmRows('reflect') },
    ],
  },
  {
    id: 'refresh-pages',
    group: 'operation',
    about: 'Pick the LLM that keeps knowledge pages up to date as memories change.',
    title: 'Refresh pages',
    short: 'Keeps knowledge pages up to date.',
    description: 'Which LLM keeps a knowledge page right as new memories arrive: takes updates, keeps counts right, and leaves the page alone when nothing changed.',
    swaps: ['llm'],
    fixed: ['embeddings', 'reranker'],
    categories: cats('refresh_pages'),
    benchmarks: benchmarks('refresh_pages'),
    costUnit: 'per 1k cases',
    tabs: [
      { id: 'delta', label: 'Full delta', description: 'Every new memory is read by the LLM. Rows with + decision model also run one; open one to see what each model did and cost.', rows: llmRows('refresh_pages') },
    ],
  },
]

// ── combined score ─────────────────────────────────────────────────────────
// Quality dominates; cost is scored on a log scale against the tab's cheapest row (logScore below:
// it scores 100, twice its cost 90, ten times 67), so a setup that pays a little more for clearly
// better answers, like one with a decision model, ranks above the bare LLM. Speed is not measured yet.
export const SCORE = { quality: 0.7, cost: 0.3 }
// A log scale against the best row: it gets 100, twice its value 90, ten times 67, a thousand 0.
export const logScore = (best: number, v: number) => Math.max(0, Math.min(100, 100 * (1 - Math.log10(v / best) / 3)))

function addScores(rows: Row[]): Row[] {
  const cheapest = Math.min(...rows.filter((x) => x.cost > 0).map((x) => x.cost))
  return rows.map((r) => {
    // A tab where nothing costs anything is scored on quality alone.
    if (rows.every((x) => x.cost === 0)) return { ...r, score: r.quality }
    const cost = r.cost === 0 ? 100 : logScore(cheapest, r.cost)
    return { ...r, score: round(SCORE.quality * r.quality + SCORE.cost * cost, 1) }
  })
}

for (const b of BOARDS) for (const t of b.tabs) t.rows = addScores(t.rows)

export const boardById = (id: string) => BOARDS.find((b) => b.id === id)

// ── use cases: Hindsight in a real task, end to end, one row per setup ──
export type CodingRow = (typeof CODING.rows)[number]
export type UseCase = {
  id: string
  title: string
  short: string
  description: string
  dataset: string
  // Pulled from agent-memory-benchmark at a pinned commit: `python benchmark-runner/pull_amb_coding.py <amb clone>`.
  // One of the two: a coding table, or a conversations one.
  results?: { ambCommit: string; rows: CodingRow[] }
  conversations?: { ambCommit: string; rows: ConversationRow[] }
}
export type ConversationRow = (typeof CONVERSATIONS.rows)[number]

export const USE_CASES: UseCase[] = [
  {
    id: 'coding-agents',
    title: 'Coding agents',
    short: 'Fixing real bugs with Hindsight’s coding-agent plugin.',
    description: 'A coding agent fixes regressions in a real codebase, with Hindsight’s coding-agent plugin recalling what it learned from the repo’s history. Each row is one Hindsight setup, so you see what a setup changes in practice.',
    dataset: 'SDE-Bench, from the Agent Memory Benchmark',
    results: { ambCommit: CODING.amb_commit, rows: CODING.rows },
  },
  {
    id: 'conversations',
    title: 'Conversations',
    short: 'Answering questions about long chat histories.',
    description: 'Hindsight remembers months of conversation between two people, then answers questions about it: who did what, when, and what follows from it. Each row is one Hindsight setup.',
    dataset: 'LoCoMo, from the Agent Memory Benchmark',
    conversations: { ambCommit: CONVERSATIONS.amb_commit, rows: CONVERSATIONS.rows },
  },
]
