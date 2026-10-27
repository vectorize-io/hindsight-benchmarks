// Leaderboard v2 data. MOCK: every number below is generated, not measured.
// The shape is what hindsight-evals will write: one board per operation, one
// swapped part per row, everything else on the reference stack.

import STRUCTURE from './eval-structure.json'

export const MOCK = true

export const RUN = {
  hindsightCommit: STRUCTURE.hindsight_commit.slice(0, 7),
  hindsightVersion: '1.1',
  judge: 'gpt-5.6-terra (Codex)',
  repeats: 1,
  date: '2026-10-27',
}

export type Part = 'llm' | 'embeddings' | 'reranker' | 'decision'

export const REFERENCE: Record<Part, { model: string; provider: string }> = {
  llm: { model: 'gemma-4-26b-a4b', provider: 'Google' },
  embeddings: { model: 'bge-small-en-v1.5', provider: 'BAAI' },
  reranker: { model: 'ms-marco-MiniLM-L-6-v2', provider: 'cross-encoder' },
  decision: { model: 'jev-latest', provider: 'TypeSafe' },
}

export const PART_LABEL: Record<Part, string> = {
  llm: 'LLM',
  embeddings: 'Embeddings',
  reranker: 'Reranker',
  decision: 'Decision model',
}

export type CategoryResult = { passed: number; total: number; failed?: string[] } // failed: case ids

export type Row = {
  id: string
  model: string
  provider: string
  icon?: string
  openWeights?: boolean
  tag?: 'reference' | 'baseline' | 'hindsight'
  quality: number // % of cases passed
  score?: number // combined: see SCORE
  categories: Record<string, CategoryResult>
  cost: number // USD, all-in, in the board's costUnit
  callsPerTask: number // LLM calls
  decisionCallsPerTask?: number
  tokensPerTask: { input: number; output: number }
  speed?: { value: number; unit: string; source: string } // provider-stated
  extra?: Record<string, string> // board-specific columns
}

export type Tab = {
  id: string
  label: string
  description: string
  rows: Row[]
  extraColumns?: { key: string; label: string; hint?: string }[]
}

export type Board = {
  id: string
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

// ── deterministic noise ────────────────────────────────────────────────────
function hash(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619)
  return ((h >>> 0) % 10000) / 10000
}
const noise = (key: string, spread: number) => (hash(key) - 0.5) * 2 * spread
const clamp = (x: number) => Math.max(0, Math.min(1, x))
const round = (x: number, d = 2) => Math.round(x * 10 ** d) / 10 ** d

// ── categories: generated from the evals at the pinned commit ──────────────
// `hindsight-evals --export-structure visualizer/lib/v2/eval-structure.json`
// writes every operation, category and case id; only the difficulty below is mock.
type Cat = { id: string; label: string; cases: number; caseIds: string[]; difficulty: number; about: string }

const DIFFICULTY: Record<string, number> = {
  temporal: 0.14, currency: 0.4, multi_hop: 0.18, no_invention: 0.16, recency: 0.2,
  scale: 0.24, aggregation: 0.22, no_overreach: 0.14, source_authority: 0.12, update: 0.12,
}

const label = (id: string) => (id.charAt(0).toUpperCase() + id.slice(1)).replace(/_/g, ' ').replace('Multi hop', 'Multi-hop')

const CATS: Record<string, Cat[]> = Object.fromEntries(
  Object.entries(STRUCTURE.operations).map(([op, o]) => [
    op,
    o.categories.map((c) => ({
      id: c.id,
      label: label(c.id),
      cases: c.cases.length,
      caseIds: c.cases,
      // recall's date and recency cases are hard on every embedder (README); elsewhere they are moderate
      difficulty: op === 'recall' && c.id === 'temporal' ? 0.35 : (DIFFICULTY[c.id] ?? 0.08),
      about: c.about,
    })),
  ]),
)

function categoryResults(cats: Cat[], strength: number, key: string) {
  const out: Record<string, CategoryResult> = {}
  let passed = 0
  let total = 0
  for (const c of cats) {
    const p = clamp(strength - c.difficulty + noise(`${key}:${c.id}`, 0.05))
    // Pass the cases in a fixed per-model order, so `failed` names real case ids.
    const order = [...c.caseIds].sort((x, y) => hash(`${key}:${x}`) - hash(`${key}:${y}`))
    const n = Math.round(c.cases * p)
    out[c.id] = { passed: n, total: c.cases, failed: order.slice(n) }
    passed += n
    total += c.cases
  }
  return { categories: out, quality: round((100 * passed) / total, 1) }
}

// ── LLMs ───────────────────────────────────────────────────────────────────
type Llm = {
  model: string
  provider: string
  icon?: string
  strength: number
  price: [number, number] // $ per 1M input, output
  speed: number // provider-stated output tok/s
  source: string
  verbosity: number // output-length multiplier
  openWeights?: boolean
  tag?: Row['tag']
}

const LLMS: Llm[] = [
  { model: 'gpt-6.1-sol', provider: 'OpenAI', icon: '/openai.png', strength: 0.97, price: [2.5, 15], speed: 92, source: 'OpenRouter', verbosity: 1.3 },
  { model: 'gpt-6-luna', provider: 'OpenAI', icon: '/openai.png', strength: 0.93, price: [0.4, 2.4], speed: 165, source: 'OpenRouter', verbosity: 1.1 },
  { model: 'claude-haiku-5.5', provider: 'Anthropic', strength: 0.92, price: [1, 5], speed: 140, source: 'OpenRouter', verbosity: 1.0 },
  { model: 'gemini-3.7-flash', provider: 'Google', icon: '/gemini.png', strength: 0.91, price: [0.75, 3.75], speed: 215, source: 'OpenRouter', verbosity: 1.2 },
  { model: 'deepseek-v4-flash', provider: 'DeepSeek', strength: 0.88, price: [0.14, 0.56], speed: 72, source: 'OpenRouter', verbosity: 1.8, openWeights: true },
  { model: 'gemini-3.1-flash-lite', provider: 'Google', icon: '/gemini.png', strength: 0.86, price: [0.1, 0.4], speed: 320, source: 'OpenRouter', verbosity: 0.9 },
  { model: 'qwen3.8-27b', provider: 'Qwen', icon: '/huggingface.png', strength: 0.85, price: [0.12, 0.6], speed: 96, source: 'OpenRouter', verbosity: 1.1, openWeights: true },
  { model: 'nemotron-3.5-super', provider: 'NVIDIA', strength: 0.84, price: [0.2, 0.8], speed: 110, source: 'OpenRouter', verbosity: 1.0, openWeights: true },
  { model: 'gemma-4-26b-a4b', provider: 'Google', icon: '/gemini.png', strength: 0.83, price: [0.09, 0.3], speed: 130, source: 'OpenRouter', verbosity: 1.0, openWeights: true, tag: 'reference' },
  { model: 'gpt-oss-20b', provider: 'OpenAI', icon: '/openai.png', strength: 0.74, price: [0.05, 0.2], speed: 250, source: 'OpenRouter', verbosity: 0.6, openWeights: true },
  { model: 'granite-4.2-8b', provider: 'IBM', icon: '/huggingface.png', strength: 0.6, price: [0.03, 0.12], speed: 300, source: 'OpenRouter', verbosity: 0.9, openWeights: true },
  { model: 'minicpm5-2b', provider: 'OpenBMB', icon: '/huggingface.png', strength: 0.45, price: [0.02, 0.05], speed: 410, source: 'OpenRouter', verbosity: 1.4, openWeights: true },
]

const DECISION_PRICE = 0.04 // $ per 1M input, jev-latest

type Workload = { input: number; output: number; calls: number; decisionCalls?: number; decisionInput?: number; qualityShift?: number }

function llmRows(op: string, tab: string, w: Workload): Row[] {
  return LLMS.map((m) => {
    const { categories, quality } = categoryResults(CATS[op], m.strength + (w.qualityShift ?? 0), `${op}:${tab}:${m.model}`)
    const input = Math.round(w.input * (1 + noise(`${m.model}:${op}:in`, 0.08)))
    const output = Math.round(w.output * m.verbosity)
    const decisionCost = ((w.decisionInput ?? 0) * DECISION_PRICE) / 1e6
    const cost = (input * m.price[0] + output * m.price[1]) / 1e6 + decisionCost
    return {
      id: m.model,
      model: m.model,
      provider: m.provider,
      icon: m.icon,
      openWeights: m.openWeights,
      tag: m.tag,
      quality,
      categories,
      cost: round(cost * 1000, 2),
      callsPerTask: round(w.calls * (1 + noise(`${m.model}:${op}:calls`, 0.1)), 1),
      decisionCallsPerTask: w.decisionCalls,
      tokensPerTask: { input, output },
      speed: { value: m.speed, unit: 'tok/s', source: m.source },
    }
  }).sort((a, b) => b.quality - a.quality || a.cost - b.cost)
}

// ── embeddings & rerankers ─────────────────────────────────────────────────
type Retriever = { model: string; provider: string; icon?: string; strength: number; price: number; ndcg: number; size: string; dims?: number; tag?: Row['tag']; openWeights?: boolean; speed?: number }

const EMBEDDERS: Retriever[] = [
  { model: 'gemini-embedding-2', provider: 'Google', icon: '/gemini.png', strength: 0.88, price: 0.15, ndcg: 0.761, size: '—', dims: 3072 },
  { model: 'text-embedding-3-large', provider: 'OpenAI', icon: '/openai.png', strength: 0.85, price: 0.13, ndcg: 0.738, size: '—', dims: 3072 },
  { model: 'qwen3-embedding-4b', provider: 'Qwen', icon: '/huggingface.png', strength: 0.87, price: 0.02, ndcg: 0.755, size: '4B', dims: 2560, openWeights: true },
  { model: 'nemotron-3.5-embed-1b', provider: 'NVIDIA', strength: 0.84, price: 0.02, ndcg: 0.731, size: '1B', dims: 2048, openWeights: true },
  { model: 'voyage-4-lite', provider: 'Voyage AI', icon: '/voyageai.png', strength: 0.83, price: 0.02, ndcg: 0.727, size: '—', dims: 1024 },
  { model: 'cohere-embed-v4', provider: 'Cohere', icon: '/cohere.png', strength: 0.82, price: 0.12, ndcg: 0.72, size: '—', dims: 1536 },
  { model: 'qwen3-embedding-0.6b', provider: 'Qwen', icon: '/huggingface.png', strength: 0.8, price: 0.01, ndcg: 0.712, size: '0.6B', dims: 1024, openWeights: true },
  { model: 'bge-small-en-v1.5', provider: 'BAAI', icon: '/huggingface.png', strength: 0.72, price: 0, ndcg: 0.664, size: '33M', dims: 384, tag: 'reference', openWeights: true },
  { model: 'all-MiniLM-L6-v2', provider: 'sentence-transformers', icon: '/huggingface.png', strength: 0.64, price: 0, ndcg: 0.621, size: '22M', dims: 384, openWeights: true },
]

const RERANKERS: Retriever[] = [
  { model: 'jev-latest', provider: 'TypeSafe · decision model as reranker', strength: 0.87, price: 0.04, ndcg: 0.776, size: '—', speed: 85 },
  { model: 'zerank-2', provider: 'ZeroEntropy', icon: '/zeroentropy.png', strength: 0.89, price: 0.05, ndcg: 0.788, size: '4B', speed: 55 },
  { model: 'rerank-3.5', provider: 'Cohere', icon: '/cohere.png', strength: 0.86, price: 2.0, ndcg: 0.772, size: '—', speed: 120 },
  { model: 'jina-reranker-v3', provider: 'Jina AI', icon: '/jina.png', strength: 0.85, price: 0.02, ndcg: 0.768, size: '0.6B', openWeights: true, speed: 70 },
  { model: 'voyage-rerank-2.5-lite', provider: 'Voyage AI', icon: '/voyageai.png', strength: 0.84, price: 0.02, ndcg: 0.764, size: '—', speed: 60 },
  { model: 'qwen3-reranker-0.6b', provider: 'Qwen', icon: '/huggingface.png', strength: 0.82, price: 0.01, ndcg: 0.752, size: '0.6B', openWeights: true, speed: 90 },
  { model: 'ms-marco-MiniLM-L-6-v2', provider: 'cross-encoder', icon: '/huggingface.png', strength: 0.74, price: 0, ndcg: 0.709, size: '22M', tag: 'reference', openWeights: true, speed: 12 },
  { model: 'none', provider: 'embeddings only', strength: 0.66, price: 0, ndcg: 0.668, size: '—', tag: 'baseline' },
]

function retrieverRows(list: Retriever[], kind: 'embeddings' | 'reranker'): Row[] {
  return list.map((r) => {
    const { categories, quality } = categoryResults(CATS.recall, r.strength, `recall:${kind}:${r.model}`)
    const tokens = kind === 'embeddings' ? 60 : 2400 // per query: the query, or the candidates
    return {
      id: r.model,
      model: r.model,
      provider: r.provider,
      icon: r.icon,
      openWeights: r.openWeights,
      tag: r.tag,
      quality,
      categories,
      cost: round(tokens * r.price, 2), // per 1M queries
      callsPerTask: 0,
      tokensPerTask: { input: tokens, output: 0 },
      speed: r.speed ? { value: r.speed, unit: 'ms p50', source: 'provider' } : undefined,
      extra: {
        ndcg: r.ndcg.toFixed(3),
        size: r.size,
        ...(r.dims ? { dims: String(r.dims) } : {}),
        price: r.price === 0 ? 'local' : `$${r.price}/1M`,
      },
    }
  }).sort((a, b) => b.quality - a.quality || a.cost - b.cost)
}

// Best pairs: a small grid over the top embedders x top rerankers.
export const PAIRS = (() => {
  const emb = EMBEDDERS.slice(0, 4)
  const rer = RERANKERS.slice(0, 4)
  return {
    embedders: emb.map((e) => e.model),
    rerankers: rer.map((r) => r.model),
    quality: emb.map((e) =>
      rer.map((r) => round(100 * clamp((e.strength + r.strength) / 2 + 0.03 + noise(`pair:${e.model}:${r.model}`, 0.03)), 1)),
    ),
  }
})()

// ── decision models ────────────────────────────────────────────────────────
type Decider = { model: string; provider: string; icon?: string; shift: number; callsSaved: number; price: number; latency?: number; tag?: Row['tag'] }

const DECIDERS: Decider[] = [
  { model: 'none', provider: 'LLM decides everything', shift: 0, callsSaved: 0, price: 0, tag: 'baseline' },
  { model: 'jev-latest', provider: 'TypeSafe', shift: -0.01, callsSaved: 0.6, price: 0.04, latency: 120, tag: 'reference' },
  { model: 'clef', provider: 'Cloudflare', shift: -0.02, callsSaved: 0.58, price: 0.042, latency: 150 },
  { model: 'gemini-3.1-flash-lite', provider: 'Google (as decision model)', icon: '/gemini.png', shift: -0.04, callsSaved: 0.4, price: 0.1, latency: 900 },
]

function decisionRows(): Row[] {
  const gemma = LLMS.find((m) => m.tag === 'reference')!
  return DECIDERS.map((d) => {
    const ops = ['recall', 'reflect', 'consolidation', 'refresh_pages']
    const per = ops.map((op) => categoryResults(CATS[op], gemma.strength + d.shift, `decision:${op}:${d.model}`))
    const categories: Record<string, CategoryResult> = {}
    ops.forEach((op, i) => {
      const passed = Object.values(per[i].categories).reduce((s, c) => s + c.passed, 0)
      const total = Object.values(per[i].categories).reduce((s, c) => s + c.total, 0)
      categories[op] = { passed, total }
    })
    const passed = Object.values(categories).reduce((s, c) => s + c.passed, 0)
    const total = Object.values(categories).reduce((s, c) => s + c.total, 0)
    const llmCalls = 4.2 * (1 - d.callsSaved)
    const llmCost = (llmCalls * 9000 * gemma.price[0] + llmCalls * 250 * gemma.price[1]) / 1e6
    const decisionCost = d.callsSaved ? (3 * 1200 * d.price) / 1e6 : 0
    return {
      id: d.model,
      model: d.model,
      provider: d.provider,
      icon: d.icon,
      tag: d.tag,
      quality: round((100 * passed) / total, 1),
      categories,
      cost: round((llmCost + decisionCost) * 1000, 2),
      callsPerTask: round(llmCalls, 1),
      decisionCallsPerTask: d.callsSaved ? 3 : 0,
      tokensPerTask: { input: Math.round(llmCalls * 9000), output: Math.round(llmCalls * 250) },
      speed: d.latency ? { value: d.latency, unit: 'ms p50', source: 'provider' } : undefined,
      extra: { saved: d.callsSaved ? `−${Math.round(d.callsSaved * 100)}%` : '—' },
    }
  })
}

// ── boards ─────────────────────────────────────────────────────────────────
const cats = (op: string) => CATS[op].map(({ id, label, cases, about, caseIds }) => ({ id, label, cases, about, caseIds }))
const benchmarks = (op: string) => STRUCTURE.operations[op as keyof typeof STRUCTURE.operations]?.benchmarks ?? []

export const BOARDS: Board[] = [
  {
    id: 'retain',
    title: 'Retain',
    short: 'Turns raw input into memories.',
    description: 'Which LLM writes the best memories from chats, documents and code: complete, attributed, dated, and nothing invented.',
    swaps: ['llm'],
    fixed: ['embeddings', 'reranker'],
    categories: cats('retain'),
    benchmarks: benchmarks('retain'),
    costUnit: 'per 1k documents',
    tabs: [
      { id: 'default', label: 'Fact extraction', description: 'Default retain: one extraction call per chunk.', rows: llmRows('retain', 'default', { input: 3400, output: 420, calls: 1 }) },
    ],
  },
  {
    id: 'recall',
    title: 'Recall',
    short: 'Finds the right memories for a query.',
    description: 'Recall uses no LLM. Its quality comes from the embedder and the reranker, so each gets its own board with the other held at the reference.',
    swaps: ['embeddings', 'reranker'],
    fixed: ['llm'],
    categories: cats('recall'),
    benchmarks: benchmarks('recall'),
    costUnit: 'per 1M queries',
    tabs: [
      {
        id: 'embeddings',
        label: 'Embeddings',
        description: 'Embedder swapped, reranker fixed to ms-marco-MiniLM-L-6-v2.',
        rows: retrieverRows(EMBEDDERS, 'embeddings'),
        extraColumns: [
          { key: 'ndcg', label: 'nDCG@10', hint: 'BEIR SciFact slice, 500 docs' },
          { key: 'dims', label: 'Dims' },
          { key: 'price', label: 'Price' },
        ],
      },
      {
        id: 'reranker',
        label: 'Rerankers',
        description: 'Reranker swapped, embedder fixed to bge-small-en-v1.5.',
        rows: retrieverRows(RERANKERS, 'reranker'),
        extraColumns: [
          { key: 'ndcg', label: 'nDCG@10', hint: 'BEIR SciFact slice, 500 docs' },
          { key: 'size', label: 'Size' },
          { key: 'price', label: 'Price' },
        ],
      },
      { id: 'pairs', label: 'Best pairs', description: 'Top 4 embedders × top 4 rerankers, to catch pairs that work better together than apart.', rows: [] },
    ],
  },
  {
    id: 'consolidation',
    title: 'Consolidation',
    short: 'Merges facts into observations.',
    description: 'Which LLM keeps observations current: merges restatements, applies updates, and keeps different things apart, even in large noisy banks.',
    swaps: ['llm'],
    fixed: ['embeddings', 'reranker', 'decision'],
    categories: cats('consolidation'),
    benchmarks: benchmarks('consolidation'),
    costUnit: 'per 1k facts',
    tabs: [
      { id: 'llm', label: 'LLM only', description: 'Every candidate observation goes to the LLM.', rows: llmRows('consolidation', 'llm', { input: 7600, output: 300, calls: 1.4 }) },
      { id: 'triage', label: 'With decision triage', description: 'The decision model skips facts that change nothing; the LLM sees the rest.', rows: llmRows('consolidation', 'triage', { input: 3100, output: 160, calls: 0.6, decisionCalls: 1, decisionInput: 900, qualityShift: -0.02 }) },
    ],
  },
  {
    id: 'reflect',
    title: 'Reflect',
    short: 'Answers questions from memory.',
    description: 'Which LLM answers best from memory: uses the latest value, aggregates correctly, and says so when the answer is not there.',
    swaps: ['llm'],
    fixed: ['embeddings', 'reranker', 'decision'],
    categories: cats('reflect'),
    benchmarks: benchmarks('reflect'),
    costUnit: 'per 1k questions',
    tabs: [
      { id: 'agent', label: 'Agent', description: 'The LLM plans its own searches and decides when it has enough.', rows: llmRows('reflect', 'agent', { input: 9200, output: 650, calls: 4.1 }) },
      { id: 'decision', label: 'With decision model', description: 'The decision model plans the searches; the LLM writes the answer.', rows: llmRows('reflect', 'decision', { input: 3300, output: 380, calls: 1.3, decisionCalls: 3, decisionInput: 3600, qualityShift: -0.025 }) },
    ],
  },
  {
    id: 'refresh-pages',
    title: 'Refresh pages',
    short: 'Keeps knowledge pages up to date.',
    description: 'Which LLM keeps a knowledge page right as new memories arrive: takes updates, keeps counts right, and leaves the page alone when nothing changed.',
    swaps: ['llm'],
    fixed: ['embeddings', 'reranker', 'decision'],
    categories: cats('refresh_pages'),
    benchmarks: benchmarks('refresh_pages'),
    costUnit: 'per 1k refreshes',
    tabs: [
      { id: 'delta', label: 'Full delta', description: 'Every new memory is read by the LLM.', rows: llmRows('refresh_pages', 'delta', { input: 8300, output: 240, calls: 1.2 }) },
      { id: 'gate', label: 'With decision gate', description: 'The decision model picks which new memories could change the page.', rows: llmRows('refresh_pages', 'gate', { input: 4100, output: 210, calls: 0.7, decisionCalls: 5, decisionInput: 6000, qualityShift: -0.03 }) },
    ],
  },
  {
    id: 'decision',
    title: 'Decision model',
    short: 'Cheap yes/no calls that save LLM calls.',
    description: 'Is a decision model worth it? LLM fixed to gemma-4-26b-a4b; the decision model is swapped. "none" is the baseline: same tasks, same grading, with cost and calls side by side.',
    swaps: ['decision'],
    fixed: ['llm', 'embeddings', 'reranker'],
    categories: [
      { id: 'recall', label: 'Recall', cases: CATS.recall.reduce((s, c) => s + c.cases, 0), about: 'Every recall case, the decision model reranking and cutting results.', caseIds: [] },
      { id: 'reflect', label: 'Reflect', cases: CATS.reflect.reduce((s, c) => s + c.cases, 0), about: 'Every reflect case, the decision model planning searches.', caseIds: [] },
      { id: 'consolidation', label: 'Consolidation', cases: CATS.consolidation.reduce((s, c) => s + c.cases, 0), about: 'Every consolidation case, the decision model triaging facts.', caseIds: [] },
      { id: 'refresh_pages', label: 'Refresh pages', cases: CATS.refresh_pages.reduce((s, c) => s + c.cases, 0), about: 'Every page case, the decision model gating new memories.', caseIds: [] },
    ],
    costUnit: 'per 1k tasks',
    tabs: [
      {
        id: 'default',
        label: 'Decision models',
        description: 'Quality across the four operations that use a decision model.',
        rows: decisionRows(),
        extraColumns: [{ key: 'saved', label: 'LLM calls saved' }],
      },
    ],
  },
]

// ── combined score ─────────────────────────────────────────────────────────
// Quality dominates; cost and speed are scored against the tab's reference row
// (the reference scores 50 on each, half the cost or twice the speed ~67).
export const SCORE = { quality: 0.7, cost: 0.15, speed: 0.15 }

function addScores(rows: Row[]): Row[] {
  const ref = rows.find((r) => r.tag === 'reference') ?? rows[Math.floor(rows.length / 2)]
  return rows.map((r) => {
    const cost = r.cost === 0 ? 100 : ref.cost === 0 ? 0 : (100 * ref.cost) / (ref.cost + r.cost)
    const v = r.speed?.value
    const rv = ref.speed?.value
    const lowerIsBetter = r.speed?.unit.startsWith('ms')
    const speed = !v || !rv ? 50 : lowerIsBetter ? (100 * rv) / (rv + v) : (100 * v) / (v + rv)
    return { ...r, score: round(SCORE.quality * r.quality + SCORE.cost * cost + SCORE.speed * speed, 1) }
  })
}

for (const b of BOARDS) for (const t of b.tabs) t.rows = addScores(t.rows)

export const boardById = (id: string) => BOARDS.find((b) => b.id === id)
