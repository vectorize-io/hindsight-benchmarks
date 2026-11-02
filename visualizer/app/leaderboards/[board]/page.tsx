import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import { BOARDS, SCORE, USE_CASES, boardById, logScore, type CodingRow, type ConversationRow, type Row, type UseCase } from '@/lib/data'
import { BoardView, CostBar, ModelScatter, QualityBar } from '@/components/BoardView'
import { BoardNav, PageHero, SetupStrip } from '@/components/Page'
import { SortTable } from '@/components/SortTable'

export function generateStaticParams() {
  return [...BOARDS, ...USE_CASES].map((b) => ({ board: b.id }))
}

const useCaseById = (id: string) => USE_CASES.find((u) => u.id === id)

export async function generateMetadata({ params }: { params: Promise<{ board: string }> }): Promise<Metadata> {
  const id = (await params).board
  const board = boardById(id) ?? useCaseById(id)
  return board ? { title: `${board.title} leaderboard`, description: board.description } : {}
}

export default async function BoardPage({ params }: { params: Promise<{ board: string }> }) {
  const id = (await params).board
  const useCase = useCaseById(id)
  if (useCase) return <UseCasePage u={useCase} />
  const board = boardById(id)
  if (!board) notFound()
  const cases = board.categories.reduce((s, c) => s + c.cases, 0)

  return (
    <>
      <PageHero title={`${board.title} leaderboard`} subtitle={board.description}></PageHero>
      <BoardNav current={board.id} />
      <main className="max-w-6xl mx-auto px-4 pt-8 space-y-8">
        <SetupStrip board={board} />
        <div className="flex items-baseline justify-between gap-4 flex-wrap">
          <h2 className="font-heading font-bold text-2xl tracking-tight">Results</h2>
          <span className="text-sm text-muted-foreground">
            {cases > 0 && `${cases} cases in ${board.categories.length} ${board.group === 'summary' ? 'operations' : 'categories'}`}
          </span>
        </div>
        <BoardView board={board} />
      </main>
    </>
  )
}

// A use case: Hindsight in a real task, one row per setup.
function UseCasePage({ u }: { u: UseCase }) {
  return (
    <>
      <PageHero title={`${u.title} leaderboard`} subtitle={u.description} />
      <BoardNav current={u.id} />
      <main className="max-w-6xl mx-auto px-4 pt-8 space-y-8">
        {u.results && <CodingResults ambCommit={u.results.ambCommit} rows={u.results.rows} />}
        {u.conversations && <ConversationResults ambCommit={u.conversations.ambCommit} rows={u.conversations.rows} />}
      </main>
    </>
  )
}

const AMB_REPO = 'https://github.com/vectorize-io/agent-memory-benchmark'

const AGENT_ICON: Record<string, string> = {
  'claude-code': '/logos/harness/claude-code.png',
  codex: '/logos/harness/codex.svg',
  opencode: '/logos/harness/opencode.png',
}

// A model's logo by its name, for the coding and conversation tables.
const modelIcon = (m: string) =>
  m.startsWith('hs-') ? '/favicon.png' : /gemini|gemma/.test(m) ? '/gemini.png' : m.includes('claude') ? '/logos/anthropic.webp' : m.includes('gpt') ? '/openai.png' : null

function ModelName({ name, className }: { name: string; className?: string }) {
  const icon = modelIcon(name)
  return (
    <span className={`inline-flex items-center gap-1 whitespace-nowrap ${className ?? ''}`}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      {icon && <img src={icon} alt="" className="w-3.5 h-3.5 object-contain shrink-0" />}
      {name}
    </span>
  )
}

// The memory column: Hindsight with its logo, then the models its server ran, on one quiet line.
function MemoryCell({ models }: { models: string[] }) {
  if (!models.length) return <span className="text-muted-foreground whitespace-nowrap">No memory</span>
  const rest = models.filter((m) => m !== 'Hindsight')
  return (
    <div>
      <div className="flex items-center gap-1.5 font-medium">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/favicon.png" alt="" className="w-4 h-4" />
        Hindsight
      </div>
      {rest.length > 0 && (
        <div className="font-mono text-xs text-muted-foreground mt-0.5 whitespace-nowrap">
          {rest.map((m, i) => (
            <span key={m}>
              {i > 0 && ' + '}
              <ModelName name={m} className={m.startsWith('hs-') ? 'text-[hsl(var(--primary))]' : ''} />
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function InfoBox({ about, items, ambCommit, path }: { about: string; items: [string, string][]; ambCommit: string; path: string }) {
  return (
    <div className="rounded-xl border border-border p-4 sm:p-5 grid sm:grid-cols-2 gap-x-8 gap-y-1 text-sm">
      <p className="sm:col-span-2 text-muted-foreground mb-3 pb-3 border-b border-border">{about}</p>
      {[['Results from', 'Agent Memory Benchmark'], ['At commit', ambCommit.slice(0, 7)], ...items].map(([k, v]) => (
        <div key={k}>
          <span className="text-muted-foreground">{k}:</span>{' '}
          {k === 'At commit' ? (
            <a href={`${AMB_REPO}/tree/${ambCommit}/outputs/${path}`} className="font-mono text-[13px] text-[hsl(var(--primary))] hover:underline">
              {v}
            </a>
          ) : (
            <span className="font-mono text-[13px]">{v}</span>
          )}
        </div>
      ))}
    </div>
  )
}

// One row per Hindsight setup: share of questions answered right, overall and by question type.
function ConversationResults({ ambCommit, rows }: { ambCommit: string; rows: ConversationRow[] }) {
  const types = Object.keys(rows[0].by_type) as (keyof ConversationRow['by_type'])[]
  return (
    <>
      <InfoBox
        about="Hindsight retains every session of a long conversation, then answers each question with reflect. A judge compares each answer with the expected one. Questions LoCoMo gets wrong (an answer the conversation never states, or the wrong person named) are removed."
        ambCommit={ambCommit}
        path="locomo-fixed"
        items={[
          ['Dataset', `LoCoMo, 10 conversations, ${rows[0].questions} questions`],
          ['Judge', rows[0].judge],
        ]}
      />
      <SortTable
        initial={1}
        cols={[
          { label: 'Memory', sortable: false },
          { label: 'Answered right' },
          ...types.map((t) => ({ label: t.replace('-', ' ').replace(/^./, (c) => c.toUpperCase()), right: true })),
          { label: 'Time per answer', tip: 'Average time to answer one question', right: true, low: true },
        ]}
        rows={rows.map((r) => ({
          key: r.run,
          values: [null, r.accuracy, ...types.map((t) => r.by_type[t]), r.answer_s],
          cells: [
            <MemoryCell key="m" models={r.memory_models} />,
            <QualityBar key="q" value={r.accuracy} />,
            ...types.map((t) => (
              <span key={t} className="font-mono tabular-nums text-muted-foreground">
                {r.by_type[t].toFixed(1)}%
              </span>
            )),
            <span key="s" className="font-mono tabular-nums text-muted-foreground">
              {r.answer_s.toFixed(1)} s
            </span>,
          ],
        }))}
      />
    </>
  )
}

// One row per agent and setup, averaged over its repeats, best score first. The score is the
// operation boards' formula with fixed first try as the quality.
function CodingResults({ ambCommit, rows }: { ambCommit: string; rows: CodingRow[] }) {
  // Cost and wall time share the 30% the operation boards give to cost, each on the same log scale.
  const cheapest = Math.min(...rows.map((r) => r.cost_usd))
  const fastest = Math.min(...rows.map((r) => r.wall_s))
  const score = (r: CodingRow) =>
    SCORE.quality * r.first_try + (SCORE.cost / 2) * (logScore(cheapest, r.cost_usd) + logScore(fastest, r.wall_s))
  const priciest = Math.max(...rows.map((r) => r.cost_usd))
  // The chart: per agent, no memory, then Hindsight (solid arrow), then Hindsight with the decision model
  // (dashed, from Hindsight). The older gemini-2.5-flash-lite runs stay in the table only.
  const CHAIN = ['vanilla', 'hindsight-gemma', 'hindsight-gemma-decide']
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
    from: r.memory === 'hindsight-gemma-decide' ? `${r.agent}hindsight-gemma` : undefined,
    extra: { Corrections: `${r.corrections.toFixed(2)} per bug`, Time: `${Math.round(r.wall_s)} s per bug` },
  }))
  return (
    <>
      <div className="rounded-xl border border-border p-4 sm:p-5 grid sm:grid-cols-2 gap-x-8 gap-y-1 text-sm">
        <p className="sm:col-span-2 text-muted-foreground mb-3 pb-3 border-b border-border">
          See what Hindsight changes for a coding agent. Each agent fixes the same bugs with no memory and with Hindsight’s coding-agent plugin; fixed first try shows how often
          memory spares it a round of failing tests.
        </p>
        {[
          ['Results from', 'Agent Memory Benchmark'],
          ['At commit', ambCommit.slice(0, 7)],
          ['Dataset', `SDE-Bench boltons, ${rows[0].tasks} bugs`],
          ['Repeats', `${rows[0].repeats} per row, averaged`],
        ].map(([k, v]) => (
          <div key={k}>
            <span className="text-muted-foreground">{k}:</span>{' '}
            {k === 'At commit' ? (
              <a href={`${AMB_REPO}/tree/${ambCommit}/outputs/sdebench`} className="font-mono text-[13px] text-[hsl(var(--primary))] hover:underline">
                {v}
              </a>
            ) : (
              <span className="font-mono text-[13px]">{v}</span>
            )}
          </div>
        ))}
      </div>
      <SortTable
        initial={2}
        cols={[
          { label: 'Agent' },
          { label: 'Memory', sortable: false },
          { label: 'Score', right: true, tip: '70% fixed first try, 15% cost per bug, 15% time per bug. Cost and time on a log scale: the best row gets 100, twice its value 90, ten times 67' },
          { label: 'Fixed first try', tip: 'Bugs fixed on the first attempt, before any failing test was fed back' },
          { label: 'Cost per bug', low: true, tip: "Agent tokens plus Hindsight's own memory bill, per bug. Runs before memory cost was measured show the agent only (marked *)" },
          { label: 'Corrections', right: true, low: true, tip: 'Times a failing test was fed back to the agent, per bug' },
          { label: 'Time per bug', right: true, low: true, tip: 'Wall-clock time per bug, averaged over the repeats' },
        ]}
        rows={rows.map((r) => ({
          key: r.agent + r.memory,
          className: r.memory === 'vanilla' ? '' : 'bg-[hsl(var(--primary)/0.04)]',
          values: [r.agent, null, score(r), r.first_try, r.cost_usd, r.corrections, r.wall_s],
          cells: [
            <div key="a" className="flex items-center gap-2.5">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              {AGENT_ICON[r.agent] && <img src={AGENT_ICON[r.agent]} alt="" className="w-5 h-5 object-contain shrink-0 rounded-sm" />}
              <div>
                <div className="font-mono text-[13px] font-medium">{r.agent}</div>
                <ModelName name={r.model} className="text-xs text-muted-foreground" />
              </div>
            </div>,
            <MemoryCell key="m" models={r.memory_models} />,
            <span key="s" className="font-mono font-semibold tabular-nums">{score(r).toFixed(1)}</span>,
            <QualityBar key="q" value={r.first_try} />,
            <span
              key="c"
              className="flex items-center gap-1"
              title={r.memory_cost_usd != null ? `agent $${r.agent_cost_usd.toFixed(3)} + memory $${r.memory_cost_usd.toFixed(3)}` : r.memory === 'vanilla' ? 'agent only, no memory' : 'agent only: memory cost not measured in this run'}
            >
              <CostBar value={r.cost_usd} priciest={priciest} />
              {r.memory_cost_usd == null && r.memory !== 'vanilla' && <span className="text-muted-foreground">*</span>}
            </span>,
            <span key="r" className="font-mono tabular-nums">{r.corrections.toFixed(2)}</span>,
            <span key="t" className="font-mono tabular-nums text-muted-foreground">{Math.round(r.wall_s)} s</span>,
          ],
        }))}
      />
      <ModelScatter
        rows={points}
        task="SDE-Bench"
        text={{
          noun: 'agents',
          x: 'Cost per bug',
          unit: 'bug',
          quality: 'Fixed first try',
          pair: 'The same agent with Hindsight (solid arrow), then adding hs-decide-1-flash (dashed)',
          costNote: 'Hindsight runs on gemma-4-26b-a4b here. Cost per bug is the agent’s tokens plus Hindsight’s own memory bill, averaged over the repeats.',
        }}
      />
    </>
  )
}
