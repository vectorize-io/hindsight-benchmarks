// The Benchmarks page's cards: per Agent Memory Benchmark dataset, Hindsight next to every other
// system, ranked. Data: lib/benchmarks.json (benchmark-runner/pull_amb_benchmarks.py), which holds
// the runs in AMB's outputs/ and the numbers other systems published (external_results.json).
import DATA from './benchmarks.json'
import CODING from './coding-agents.json'

export const AMB_SITE = 'https://agentmemorybenchmark.ai'
export const AMB_REPO = 'https://github.com/vectorize-io/agent-memory-benchmark'

type SplitScore = { score: number | null; own_language?: number; path?: string; avg_query_usd?: number; ingest_usd?: number }
type Run = { memory_provider: string | null; answer_llm: string | null; judge_llm: string | null; description: string | null; splits: Record<string, SplitScore> }
type Published = { memory: string; accuracy: number | null; comment: string | null; source_url: string | null }
type Dataset = { description: string; score_type: string; note: string | null; runs: Record<string, Run>; published: Record<string, Published[]> }
const DATASETS = DATA.datasets as unknown as Record<string, Dataset>
export const AMB_COMMIT = DATA.amb_commit

// baseline: the same agent with no memory, there to show the gain, not a system to rank against.
export type Row = { label: string; value: number; ours: boolean; published: boolean; baseline?: boolean; href?: string; title?: string; price?: string }
export type Panel = { label?: string; rows: Row[] }
export type Card = { key: string; bench: string; title: string; about: string; metric: string; panels: Panel[]; note: string }
export type Section = { key: string; title: string; about: string; cards: Card[] }

type CardSpec = {
  key: string
  bench: string
  title: string
  about: string
  splits: [string, string?][] // [split, panel label]
  ownLanguage?: boolean // ViDoRe: each set scored on questions in its own language
  runLabels?: Record<string, string> // several Hindsight runs on one card (SupportBench's setups)
}

// Other systems AMB ran itself, by run name.
const RUN_NAMES: Record<string, string> = { 'hybrid-search': 'Hybrid search (AMB baseline)', cognee: 'Cognee' }
const ALL_LANGUAGES = ' (all 6 query languages)' // a second ViDoRe setting, scored differently

const MEMORY: CardSpec[] = [
  { key: 'longmemeval', bench: 'LongMemEval S', title: 'Long chat histories', about: '500 questions over chat histories of about 115K tokens each.', splits: [['s']] },
  { key: 'locomo', bench: 'LoCoMo', title: 'Long conversations', about: '1,540 questions over 10 long conversations held across many sessions.', splits: [['locomo10']] },
  { key: 'personamem', bench: 'PersonaMem 32K', title: 'What a user likes', about: '589 questions on a user’s preferences as they change over time.', splits: [['32k']] },
  { key: 'lifebench', bench: 'LifeBench EN', title: 'A person’s life', about: '2,003 questions over a simulated year of one person’s life.', splits: [['en']] },
  {
    key: 'beam', bench: 'BEAM', title: 'Memory as it grows', about: '2,000 questions over conversations from 100K to 10M tokens.',
    splits: [['100k', '100K tokens'], ['500k', '500K tokens'], ['1m', '1M tokens'], ['10m', '10M tokens']],
  },
  { key: 'precisionmembench', bench: 'PrecisionMemBench', title: 'Recalling exactly the right memories', about: '77 cases that say which memories must come back and which must not.', splits: [['single-turn']] },
]
const KNOWLEDGE: CardSpec[] = [
  { key: 'vidore-v3-pdf', bench: 'ViDoRe V3 · raw PDFs', title: 'Enterprise PDFs', about: '8 public sets of real company PDFs, English and French. The server reads the files itself.', splits: [['average']], ownLanguage: true },
  { key: 'vidore-v3-markdown', bench: 'ViDoRe V3 · page text', title: 'Enterprise documents as text', about: 'The same 8 sets, as the page text the dataset ships.', splits: [['average']], ownLanguage: true },
  {
    key: 'legalbench-rag', bench: 'LegalBench-RAG mini', title: 'Contracts',
    about: '776 questions over 714 contracts, scored on the exact text lawyers marked. The other systems ran with no file names.',
    splits: [['mini']],
    runLabels: {
      'hindsight-no-filenames': 'Hindsight, no file names',
      'hindsight-filenames': 'Hindsight, with file names',
    },
  },
  { key: 'atm-bench-retrieval', bench: 'ATM-Bench', title: 'Photos, videos and emails', about: '1,013 questions over four years of one person’s photos, videos and emails.', splits: [['atm-bench']] },
]
const HYBRID: CardSpec[] = [
  {
    key: 'supportbench', bench: 'SupportBench', title: 'A support agent: product docs and customer history',
    about: '114 support questions over the real Nextcloud manuals (837 PDF pages), each customer’s past chats and approved exceptions. The same documents, stored three ways.',
    splits: [['nextcloud']],
    runLabels: {
      'support-linked': 'Memory + knowledge bank',
      'support-memory': 'Memory bank only',
      'support-knowledge': 'Knowledge bank only',
    },
  },
]

// Search price per 1,000 queries, as each hosted service lists it: Hindsight Cloud's KB Search
// ($0.002 per query, docs.hindsight.vectorize.io/cloud/billing) and Mixedbread's list price with its reranker.
const SEARCH_PRICE: Record<string, number> = { Hindsight: 2, 'Mixedbread + reranker': 7.5 }
const STORE: Record<string, string> = { 'vidore-v3-pdf': 'its page-image store', 'vidore-v3-markdown': 'its page-text store' }

const isOurs = (r: Run) => (r.memory_provider ?? '').startsWith('hindsight')
const model = (m: string | null) => (m ?? '').replace(/^[a-z]+:/, '').replace('gemini-3.1-pro-preview', 'Gemini 3.1 Pro').replace('gemini-2.5-flash-lite', 'Gemini 2.5 Flash-Lite')

function card(spec: CardSpec, section: string): Card | null {
  const d = DATASETS[spec.key]
  if (!d) return null
  const runs = Object.entries(d.runs)
  const panels = spec.splits.map(([split, label]) => {
    const rows: Row[] = []
    for (const [name, r] of runs) {
      const s = r.splits[split]
      const value = s && (spec.ownLanguage ? s.own_language : s.score)
      if (value == null) continue
      if (spec.runLabels && !spec.runLabels[name]) continue
      rows.push({
        label: spec.runLabels?.[name] ?? (isOurs(r) ? 'Hindsight' : RUN_NAMES[name] ?? name),
        value,
        ours: isOurs(r),
        published: false,
        href: s.path ? `${AMB_SITE}/run/${s.path}` : `${AMB_SITE}/dataset/${spec.key}`,
        title: r.description ?? undefined,
      })
    }
    for (const p of d.published[split] ?? []) {
      if (p.accuracy == null || p.memory.endsWith(ALL_LANGUAGES)) continue
      // No links out to other systems: a published row points at the dataset's page on AMB, which cites the source.
      rows.push({ label: p.memory, value: p.accuracy, ours: false, published: true, href: `${AMB_SITE}/dataset/${spec.key}`, title: p.comment ?? undefined })
    }
    // A system that published several numbers under one name (MemMachine on LoCoMo) shows its best.
    const best = rows.sort((a, b) => b.value - a.value).filter((r, i, all) => all.findIndex((o) => o.label === r.label) === i)
    if (STORE[spec.key]) for (const r of best) if (SEARCH_PRICE[r.label] != null) r.price = `$${SEARCH_PRICE[r.label].toFixed(2)}`
    return { label, rows: best }
  })
  if (!panels.some((p) => p.rows.some((r) => r.ours))) return null
  const ran = runs.filter(([, r]) => r.answer_llm)
  const others = [...new Set(runs.filter(([, r]) => !isOurs(r)).map(([n]) => RUN_NAMES[n] ?? n))]
  const note =
    section === 'knowledge'
      ? 'Search only: no LLM writes or grades anything. Hindsight embeds with EmbeddingGemma 2 and reranks with hs-decide-1-flash-mm. The other rows are the numbers each system or leaderboard published.'
      : section === 'hybrid'
        ? 'One Hindsight agent (reflect, Gemma 4 26B) answers in each setup; only where the documents are stored changes. No other system has run SupportBench yet.'
        : `Run on AMB: Hindsight${others.length ? `, ${others.join(', ')}` : ''}${ran.length ? `, answers by ${model(ran[0][1].answer_llm)}, graded by ${model(ran[0][1].judge_llm)}` : ''}. Striped rows are the numbers each system published, with its own models and setup.`
  const metric = d.score_type === 'accuracy' ? 'Accuracy' : d.score_type.replace(/^ndcg/, 'NDCG').replace(/^recall/, 'Recall')
  return { key: spec.key, bench: spec.bench, title: spec.title, about: spec.about, metric, panels,
    note: STORE[spec.key] ? `${note} Prices are per 1,000 searches, as each service lists them: $2.00 is Hindsight Cloud’s knowledge-bank search, a flat fee per query with reranking included (loading and storing documents are billed separately); $7.50 is Mixedbread’s list price with its reranker, on ${STORE[spec.key]}.` : note }
}

// SDE-Bench, from the coding leaderboard's data (pull_amb_coding.py): per coding agent, how often the
// first fix passes the hidden test, with Hindsight's plugin and with no memory. No other memory system has run it.
const AGENTS: Record<string, string> = { 'claude-code': 'Claude Code · Sonnet 5', codex: 'Codex · GPT-5.4 mini', opencode: 'OpenCode · Gemini 3.5 Flash' }
const MEMORY_LLM: Record<string, string> = { hindsight: 'Hindsight (Flash-Lite)', 'hindsight-gemma': 'Hindsight (Gemma 4)', vanilla: 'No memory' }
function sdeCard(): Card {
  const rows = CODING.rows as { agent: string; memory: string; first_try: number; tasks: number; repeats: number }[]
  const panels = [...new Set(rows.map((r) => r.agent))].map((agent) => ({
    label: AGENTS[agent] ?? agent,
    rows: rows
      .filter((r) => r.agent === agent)
      .map((r) => ({
        label: MEMORY_LLM[r.memory] ?? r.memory,
        value: r.first_try / 100,
        ours: r.memory !== 'vanilla',
        published: false,
        baseline: r.memory === 'vanilla',
        href: `${AMB_SITE}/dataset/sdebench`,
      }))
      .sort((a, b) => b.value - a.value),
  }))
  return {
    key: 'sdebench', bench: 'SDE-Bench', title: 'Coding agents that learn from past fixes', metric: 'First-try fixes',
    about: '61 bug-fix tasks whose obvious fix fails a hidden test. How often the agent’s first fix passes, with memory of past sessions and without.',
    panels,
    note: `Each agent and setup ran every task ${rows[0].repeats} times on AMB; the score is the share fixed with no correction. Hindsight (Flash-Lite) and Hindsight (Gemma 4) are the coding-agent plugin with the server’s LLM set to Gemini 2.5 Flash-Lite or Gemma 4 26B. No other memory system has run SDE-Bench yet.`,
  }
}

export const SECTIONS: Section[] = [
  { key: 'memory', title: 'Memory', about: 'An agent that remembers users and conversations. It answers from what Hindsight recalls from a memory bank, and an LLM grades the answer.', cards: [...MEMORY.map((s) => card(s, 'memory')), sdeCard()] },
  { key: 'knowledge', title: 'Knowledge', about: 'Search over documents in a knowledge bank: PDFs, text, images and video. Scored on the ranking alone, against each dataset’s own labels.', cards: KNOWLEDGE.map((s) => card(s, 'knowledge')) },
  { key: 'hybrid', title: 'Memory + knowledge', about: 'An agent that needs both: shared documents in a knowledge bank, and what it knows about each user in a memory bank.', cards: HYBRID.map((s) => card(s, 'hybrid')) },
].map((s) => ({ ...s, cards: s.cards.filter((c): c is Card => c !== null) }))

// Hindsight's best place in a panel, and how many systems it holds (null: Hindsight is the only one).
export function rank(p: Panel): { place: number; of: number } | null {
  if (p.rows.every((r) => r.ours || r.baseline)) return null
  const field = p.rows.filter((r) => !r.baseline)
  return { place: field.findIndex((r) => r.ours) + 1, of: field.filter((r) => !r.ours).length + 1 }
}

