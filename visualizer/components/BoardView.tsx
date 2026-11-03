'use client'

import { Fragment, useEffect, useRef, useState } from 'react'
import { ChevronDown, ArrowDown, Info } from 'lucide-react'
import type { Board, Row, Source, Tab } from '@/lib/data'
import { SCORE } from '@/lib/data'
import { cn } from '@/lib/utils'
import { ModelCell, PROVIDER_ICON, fmtCost, fmtTokens } from './ModelCell'

export function QualityBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <span className="font-mono text-sm font-semibold tabular-nums w-11 text-right">{value.toFixed(1)}</span>
      <div className="h-1.5 flex-1 rounded-full bg-secondary overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${value}%`, background: 'linear-gradient(90deg, #0074d9, #009296)' }} />
      </div>
    </div>
  )
}

// Longer is more expensive, the opposite of quality: amber so the two never read alike.
export function CostBar({ value, priciest }: { value: number; priciest: number }) {
  return (
    <div className="flex items-center gap-2 min-w-[130px]">
      <span className="font-mono text-sm font-semibold tabular-nums w-12 text-right">{fmtCost(value)}</span>
      <div className="h-1.5 flex-1 rounded-full bg-secondary overflow-hidden">
        <div className="h-full rounded-full bg-amber-500/80" style={{ width: `${priciest ? (100 * value) / priciest : 0}%` }} />
      </div>
    </div>
  )
}

type SortKey = 'score' | 'quality' | 'cost'

const SCORE_HINT = `Combined score: ${SCORE.quality * 100}% quality, ${SCORE.cost * 100}% cost. The cheapest row gets 100 on cost, a row that costs twice as much 90, ten times as much 67.`

export function InfoTip({ text }: { text: string }) {
  return (
    <span title={text} aria-label={text} className="inline-flex cursor-help text-muted-foreground hover:text-foreground">
      <Info className="w-3 h-3" />
    </span>
  )
}

export function Tabs({ tabs, current, onChange }: { tabs: { id: string; label: string }[]; current: string; onChange: (id: string) => void }) {
  return (
    <div className="flex flex-wrap gap-1 p-1 rounded-xl bg-secondary w-fit mb-3" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.id}
          role="tab"
          aria-selected={t.id === current}
          onClick={() => onChange(t.id)}
          className={cn(
            'px-3.5 h-8 rounded-lg text-sm font-heading font-bold transition-colors',
            t.id === current ? 'bg-background shadow-sm text-foreground' : 'text-muted-foreground hover:text-foreground',
          )}
        >
          {t.label}
        </button>
      ))}
    </div>
  )
}

export function BoardView({ board }: { board: Board }) {
  const [tabId, setTabId] = useState(board.tabs[0].id)
  const tab = board.tabs.find((t) => t.id === tabId)!

  return (
    <div>
      {board.tabs.length > 1 && <Tabs tabs={board.tabs} current={tabId} onChange={setTabId} />}
      <p className="text-sm text-muted-foreground mb-6">
        {tab.description}
        {board.tabs.length > 1 && ' Rows are ranked within this tab only.'}
      </p>

      {tab.rows.length === 0 ? (
        <p className="text-sm text-muted-foreground border border-dashed border-border rounded-lg px-4 py-8 text-center">No results yet.</p>
      ) : (
        <TabView board={board} tab={tab} key={tab.id} />
      )}
    </div>
  )
}

function TabView({ board: boardIn, tab }: { board: Board; tab: Tab }) {
  // A tab scored on another set brings its own categories.
  const board = tab.categories ? { ...boardIn, categories: tab.categories, benchmarks: undefined } : boardIn
  const [sort, setSort] = useState<SortKey>('score')
  const [open, setOpen] = useState<string | null>(null)
  const hasDecision = tab.rows.some((r) => r.decisionCallsPerTask)

  const rows = [...tab.rows].sort((a, b) =>
    sort === 'score'
      ? (b.score ?? 0) - (a.score ?? 0)
      : sort === 'quality'
      ? b.quality - a.quality || a.cost - b.cost
      : a.cost - b.cost,
  )

  const SortHead = ({ k, children, className, tip }: { k: SortKey; children: React.ReactNode; className?: string; tip?: string }) => (
    <th className={cn('px-3 py-2.5 font-medium whitespace-nowrap', className)}>
      <span className="inline-flex items-center gap-1">
        <button onClick={() => setSort(k)} className={cn('inline-flex items-center gap-1 hover:text-foreground', sort === k && 'text-foreground')}>
          {children}
          {sort === k && <ArrowDown className="w-3 h-3" />}
        </button>
        {tip && <InfoTip text={tip} />}
      </span>
    </th>
  )
  const costTip = 'Tokens each model really used in these evals, times its list price on OpenRouter. Includes decision model calls.'

  const priciest = Math.max(0, ...tab.rows.map((r) => r.cost))
  const best = Math.max(...tab.rows.map((r) => r.score ?? -Infinity))

  return (
    <div className="space-y-10">
      <div className="rounded-xl border border-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-xs text-muted-foreground text-left bg-[hsl(var(--band))]">
              <th className="px-3 py-2.5 font-medium w-10">#</th>
              <th className="px-3 py-2.5 font-medium">Model</th>
              <SortHead k="score" tip={SCORE_HINT}>
                Score
              </SortHead>
              <SortHead k="quality">Quality</SortHead>
              <SortHead k="cost" tip={`${costTip} The bar is longer when it costs more.`}>
                Cost <span className="font-normal">{board.costUnit}</span>
              </SortHead>
              <th className="px-3 py-2.5 font-medium">Cases</th>
              <th className="px-3 py-2.5 font-medium text-right whitespace-nowrap">LLM calls</th>
              {hasDecision && <th className="px-3 py-2.5 font-medium text-right whitespace-nowrap">Decision calls</th>}
              <th className="px-3 py-2.5 font-medium text-right whitespace-nowrap">Tokens in / out</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const total = Object.values(r.categories).reduce((s, c) => s + c.total, 0)
              const passed = Object.values(r.categories).reduce((s, c) => s + c.passed, 0)
              const expanded = open === r.id
              return (
                <Fragment key={r.id}>
                  <tr
                    onClick={() => setOpen(expanded ? null : r.id)}
                    className={cn(
                      'border-b border-border last:border-0 transition-colors cursor-pointer hover:bg-secondary/60',
                      r.tag === 'reference' && 'bg-[hsl(var(--teal))]/[0.05]',
                    )}
                  >
                    <td className="px-3 py-3 font-mono text-xs text-muted-foreground">{sort !== 'cost' ? i + 1 : ''}</td>
                    <td className="px-3 py-3 max-w-[260px]">
                      <div className="flex items-center gap-2">
                        <ChevronDown className={cn('w-3.5 h-3.5 shrink-0 text-muted-foreground transition-transform', expanded && 'rotate-180')} />
                        <ModelCell row={r} />
                      </div>
                    </td>
                    <td className="px-3 py-3 font-mono text-sm font-semibold tabular-nums">
                        {r.score === best ? (
                          // The winner: the brand gradient drawn as a thick underline.
                          <span className="pb-1 bg-[linear-gradient(90deg,#0074d9,#009296)] bg-[length:100%_3px] bg-bottom bg-no-repeat" title="Best score in this tab">
                            {r.score?.toFixed(1)}
                          </span>
                        ) : (
                          r.score?.toFixed(1)
                        )}
                      </td>
                    <td className="px-3 py-3">
                      <QualityBar value={r.quality} />
                    </td>
                    <td className="px-3 py-3">
                      <CostBar value={r.cost} priciest={priciest} />
                    </td>
                    <td className="px-3 py-3 font-mono text-xs text-muted-foreground tabular-nums whitespace-nowrap">
                      {passed}/{total}
                    </td>
                    <td className="px-3 py-3 font-mono text-right tabular-nums">{r.callsPerTask.toFixed(1)}</td>
                    {hasDecision && <td className="px-3 py-3 font-mono text-right tabular-nums text-muted-foreground">{r.decisionCallsPerTask ?? '—'}</td>}
                    <td className="px-3 py-3 font-mono text-right tabular-nums text-muted-foreground whitespace-nowrap">
                      {fmtTokens(r.tokensPerTask.input)} / {fmtTokens(r.tokensPerTask.output)}
                    </td>
                  </tr>
                  {expanded && (
                    <tr className="border-b border-border bg-[hsl(var(--band))]">
                      <td colSpan={99} className="px-6 py-5">
                        <CategoryBreakdown board={board} row={r} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
      <>
          <p className="-mt-7 text-xs text-muted-foreground">
            Click a row for its categories and failed cases. Hover <Info className="w-3 h-3 inline" /> for how a column is computed.
          </p>

          <ModelScatter rows={tab.rows} task={board.title} />

          <section>
            <h3 className="font-heading font-bold text-base mb-1">Cases passed by category</h3>
            <p className="text-xs text-muted-foreground mb-4">Where each model wins and loses. The best in each category is outlined.</p>
            <Heatmap board={board} rows={rows} />
          </section>

          {board.group === 'operation' && <section className="rounded-xl border border-border p-5">
            <h3 className="font-heading font-bold text-base mb-1">Categories</h3>
            <p className="text-xs text-muted-foreground mb-4">From the evals in this version. Click one for its cases.</p>
            <ul className="grid md:grid-cols-2 gap-x-10 gap-y-1">
              {board.categories.map((c) => (
                <li key={c.id}>
                  <details className="group">
                    <summary className="text-sm flex gap-3 cursor-pointer list-none py-1 hover:text-foreground">
                      <span className="font-medium w-32 shrink-0">{c.label}</span>
                      <span className="text-muted-foreground flex-1">
                        {c.about} <span className="font-mono text-xs">({c.cases})</span>
                      </span>
                    </summary>
                    {c.caseIds.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 pl-[8.75rem] pb-2 pt-1">
                        {c.caseIds.map((id) => (
                          <span key={id} className="font-mono text-[11px] px-2 py-0.5 rounded bg-secondary text-muted-foreground">{id}</span>
                        ))}
                      </div>
                    )}
                  </details>
                </li>
              ))}
            </ul>
            {board.benchmarks && board.benchmarks.length > 0 && (
              <p className="text-xs text-muted-foreground mt-4 pt-3 border-t border-border">
                Also in this version, measured but not graded as cases:{' '}
                {board.benchmarks.map((b) => <span key={b} className="font-mono">{b} </span>)}
              </p>
            )}
          </section>}
      </>
    </div>
  )
}

// The best cell in a column, outlined as on the Hindsight Models tables: blue when it is ours.
const winCell = (ours: boolean) =>
  ours ? 'bg-[hsl(var(--primary)/0.14)] shadow-[inset_0_0_0_1.5px_hsl(var(--primary))]' : 'bg-foreground/[0.07] shadow-[inset_0_0_0_1.5px_hsl(var(--foreground)/0.35)]'

function CategoryBreakdown({ board, row }: { board: Board; row: Row }) {
  const failed = board.categories.flatMap((c) => (row.categories[c.id].failed ?? []).map((id) => ({ cat: c.label, id })))
  return (
    <div className="space-y-5">
      {row.split && <WorkSplit row={row} split={row.split} />}
      {row.source && <SourceDetails row={row} source={row.source} />}
      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-2.5">
        {board.categories.map((c) => {
          const r = row.categories[c.id]
          const ratio = r.passed / r.total
          return (
            <div key={c.id} className="flex items-center gap-3 text-sm" title={c.about}>
              <span className="w-32 shrink-0 truncate">{c.label}</span>
              <div className="h-1.5 flex-1 rounded-full bg-secondary overflow-hidden">
                <div className="h-full rounded-full" style={{ width: `${ratio * 100}%`, background: ratio < 0.5 ? 'hsl(var(--bad))' : 'linear-gradient(90deg, #0074d9, #009296)' }} />
              </div>
              <span className="font-mono text-xs tabular-nums text-muted-foreground w-9 text-right">
                {r.passed}/{r.total}
              </span>
            </div>
          )
        })}
      </div>
      {failed.length > 0 && (
        <div>
          <div className="text-xs text-muted-foreground mb-2">Failed cases</div>
          <div className="flex flex-wrap gap-1.5">
            {failed.map((f) => (
              <span key={`${f.cat}/${f.id}`} className="font-mono text-[11px] px-2 py-0.5 rounded bg-[hsl(var(--bad))]/10 text-[hsl(var(--bad))]">
                {f.cat.toLowerCase().replace(/ /g, '_')}/{f.id}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// For a row that also runs a decision model: per case, what the LLM and the
// decision model each did and cost, as a share bar per measure.
function WorkSplit({ row, split }: { row: Row; split: NonNullable<Row['split']> }) {
  const llmName = row.model
  const measures = [
    { label: 'Calls', llm: split.llm.calls, dec: split.decision.calls, fmt: (v: number) => v.toFixed(1) },
    { label: 'Input tokens', llm: split.llm.input, dec: split.decision.input, fmt: (v: number) => fmtTokens(Math.round(v)) },
    { label: 'Output tokens', llm: split.llm.output, dec: split.decision.output, fmt: (v: number) => fmtTokens(Math.round(v)) },
    { label: 'Cost per 1k cases', llm: split.llm.cost, dec: split.decision.cost, fmt: (v: number) => fmtCost(v) },
  ]
  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground mb-3">
        <span>Where the work goes, per case</span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-[hsl(var(--primary))]" />
          <span className="font-mono">{llmName}</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-sm bg-[hsl(var(--teal))]" />
          <span className="font-mono">{split.decisionModel}</span>
        </span>
      </div>
      <div className="grid sm:grid-cols-2 gap-x-8 gap-y-2.5">
        {measures.map((m) => {
          const total = m.llm + m.dec
          const share = total ? m.dec / total : 0
          return (
            <div key={m.label} className="flex items-center gap-3 text-sm">
              <span className="w-32 shrink-0">{m.label}</span>
              <div className="h-2 flex-1 rounded-full bg-secondary overflow-hidden flex" title={`${Math.round(share * 100)}% ${split.decisionModel}`}>
                <div className="h-full bg-[hsl(var(--primary))]" style={{ width: `${(1 - share) * 100}%` }} />
                <div className="h-full bg-[hsl(var(--teal))]" style={{ width: `${share * 100}%` }} />
              </div>
              <span className="font-mono text-xs tabular-nums text-muted-foreground w-36 text-right whitespace-nowrap">
                {m.fmt(m.llm)} <span className="opacity-50">+</span> <span className="text-[hsl(var(--teal))]">{m.fmt(m.dec)}</span>
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function Heatmap({ board, rows }: { board: Board; rows: Row[] }) {
  return (
    <div className="rounded-xl border border-border overflow-x-auto">
      <table className="text-xs w-full">
        <thead>
          <tr className="border-b border-border bg-[hsl(var(--band))]">
            <th className="px-3 py-2 text-left font-medium text-muted-foreground sticky left-0 bg-[hsl(var(--band))]">Model</th>
            {board.categories.map((c) => (
              <th key={c.id} className="px-1.5 py-2 font-medium text-muted-foreground whitespace-nowrap" title={c.about}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-b border-border last:border-0">
              <td className="px-3 py-1.5 font-mono whitespace-nowrap sticky left-0 bg-background">
                <span className="inline-flex items-center gap-2">
                  {(r.icon ?? PROVIDER_ICON[r.provider]) ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={r.icon ?? PROVIDER_ICON[r.provider]} alt="" className="w-4 h-4 object-contain shrink-0" />
                  ) : (
                    <span className="w-4 h-4 shrink-0 rounded-sm bg-secondary text-[9px] font-bold flex items-center justify-center text-muted-foreground">{r.provider[0]}</span>
                  )}
                  {r.split ? r.model : (r.label ?? r.model)}
                  {r.split && (
                    <span className="inline-flex items-center gap-1 text-muted-foreground">
                      +
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src="/favicon.png" alt="" className="w-3.5 h-3.5" />
                      <span className="text-[hsl(var(--primary))]">{r.split.decisionModel}</span>
                    </span>
                  )}
                </span>
              </td>
              {board.categories.map((c) => {
                const v = r.categories[c.id]
                const best = Math.max(...rows.map((x) => x.categories[c.id].passed / x.categories[c.id].total))
                return (
                  <td key={c.id} className="px-1 py-1">
                    <div
                      className={cn('rounded font-mono tabular-nums text-center py-1 min-w-[44px]', rows.length > 1 && v.passed / v.total === best && winCell(r.tag === 'hindsight'))}
                      title={`${r.label ?? r.model} · ${c.label}: ${v.passed}/${v.total}`}
                    >
                      {v.passed}/{v.total}
                    </div>
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Quality against cost, drawn at the container's real width so text stays at its true size.
// The plain models are the chart, each once as its logo, with the best-value region drawn from
// them alone. A run with the decision model is a small Hindsight logo, arrowed from its LLM.
// bare: the chart alone, without title and legend, for the home page card.
// The coding board reuses it with agents for models and Hindsight setups for decision-model runs, and the
// Hindsight Models pages with latency for cost: `text` renames the parts, `ms` reads the cost as milliseconds.
export type ScatterText = { title?: string; noun: string; x: string; unit: string; quality: string; speed?: string; pair?: string; costNote?: string }
export function ModelScatter({ rows, task, bare, text }: { rows: Row[]; task: string; bare?: boolean; text?: ScatterText }) {
  const box = useRef<HTMLDivElement>(null)
  const [W, setW] = useState(760)
  const [hover, setHover] = useState<string | null>(null)
  useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const plain = rows.filter((r) => r.cost > 0 && !r.split)
  const pairs = rows
    .filter((d) => d.cost > 0 && d.split)
    .flatMap((d) => {
      const base = plain.find((b) => b.model === d.model)
      return base ? [{ base, d }] : []
    })
  if (plain.length < 2) return null
  const dm = pairs[0]?.d.split?.decisionModel
  const tx: ScatterText = text ?? {
    noun: 'models',
    x: 'Cost per 1k cases',
    unit: '1k cases',
    quality: 'Quality',
    pair: dm && `The same model with ${dm}, at the arrow's head`,
    costNote: dm && `Cost per 1k cases includes the ${dm} calls for the runs that use it.`,
  }

  const shown = [...plain, ...pairs.map(({ d }) => d)]
  const H = Math.round(Math.min(520, Math.max(320, W * 0.5)))
  const pad = { l: 56, r: 20, t: 16, b: 52 }
  const all = shown
  const xs = all.map((r) => Math.log10(r.cost))
  const x0 = Math.min(...xs) - 0.1
  const x1 = Math.max(...xs) + 0.1
  const qs = all.map((r) => r.quality)
  const span = Math.max(...qs) - Math.min(...qs)
  const yStep = span > 40 ? 10 : span > 18 ? 5 : 2
  const y0 = Math.max(0, Math.floor((Math.min(...qs) - 1) / yStep) * yStep)
  const y1 = Math.min(100, Math.ceil((Math.max(...qs) + 1) / yStep) * yStep)
  const r1 = (v: number) => Math.round(v * 10) / 10
  const sx = (c: number) => r1(pad.l + ((Math.log10(c) - x0) / (x1 - x0)) * (W - pad.l - pad.r))
  const sy = (q: number) => r1(pad.t + (1 - (q - y0) / (y1 - y0)) * (H - pad.t - pad.b))

  // Best value among the plain models only.
  const front: Row[] = []
  for (const p of [...plain].sort((a, b) => a.cost - b.cost || b.quality - a.quality))
    if (!front.length || p.quality > front[front.length - 1].quality) front.push(p)
  const onFront = new Set(front.map((p) => p.id))
  const region = [`${pad.l},${pad.t}`, `${pad.l},${sy(front[0].quality)}`]
  front.forEach((f, i) => {
    if (i) region.push(`${sx(f.cost)},${sy(front[i - 1].quality)}`)
    region.push(`${sx(f.cost)},${sy(f.quality)}`)
  })
  region.push(`${sx(front[front.length - 1].cost)},${pad.t}`)


  // 1-2-5 ticks, or finer ones when the costs sit close together.
  const ticksFor = (ms: number[]) => {
    const out: number[] = []
    for (let e = Math.floor(x0); e <= Math.ceil(x1); e++)
      for (const m of ms) {
        const v = Math.round(m * 10 ** e * 1000) / 1000
        if (Math.log10(v) >= x0 && Math.log10(v) <= x1) out.push(v)
      }
    return out
  }
  let xTicks = ticksFor([1, 2, 5])
  if (xTicks.length < 3) xTicks = ticksFor([1, 1.5, 2, 3, 4, 5, 6, 7, 8])
  const yTicks: number[] = []
  for (let q = y0; q <= y1; q += yStep) yTicks.push(q)

  // Labels always right of the point (left only past the edge), nudged up or down to clear.
  const CHAR = 6.7
  const L = 8 // half a logo
  const icon = (p: Row) => p.icon ?? PROVIDER_ICON[p.provider]
  const placed = shown.map((p) => ({ x: sx(p.cost) - L, y: sy(p.quality), w: 2 * L }))
  // The home card is narrow: only the best-value models get a name there, hover shows the rest.
  const labels = plain
    .filter((p) => !bare || onFront.has(p.id))
    .sort((a, b) => sy(a.quality) - sy(b.quality))
    .map((p) => {
      const cx = sx(p.cost)
      const cy = sy(p.quality)
      const w = p.model.length * CHAR
      const x = cx + L + 5 + w > W - pad.r ? cx - L - 5 - w : cx + L + 5
      let y = cy
      for (const dy of [0, -13, 13, -26, 26, -39, 39]) {
        if (placed.every((b) => (b.x === cx - L && b.y === cy) || Math.abs(b.y - (cy + dy)) >= 13 || x + w < b.x || b.x + b.w < x)) {
          y = cy + dy
          break
        }
      }
      placed.push({ x, y, w })
      return { p, cx, cy, x, y, w }
    })

  const muted = 'hsl(var(--muted-foreground))'
  const accent = 'hsl(var(--primary))'
  const hp = shown.find((p) => p.id === hover)
  const chart = (
      <div ref={box} className="relative">
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block" role="img" aria-label="Quality vs. cost">
          <defs>
            <marker id="ghost-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M0 0 L10 5 L0 10 z" fill={accent} opacity="0.7" />
            </marker>
          </defs>
          <polygon points={region.join(' ')} fill={accent} opacity="0.07" />
          <text x={pad.l + 8} y={pad.t + 14} fontSize="11" fill={accent} opacity="0.8">
            Best value
          </text>
          {yTicks.map((q) => (
            <g key={q}>
              <line x1={pad.l} x2={W - pad.r} y1={sy(q)} y2={sy(q)} stroke="hsl(var(--border))" />
              <text x={pad.l - 10} y={sy(q) + 4} textAnchor="end" fontSize="11" fill={muted} className="font-mono">
                {q}%
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={pad.t} y2={H - pad.b} stroke="hsl(var(--border))" strokeDasharray="2 4" />
              <text x={sx(t)} y={H - pad.b + 18} textAnchor="middle" fontSize="11" fill={muted} className="font-mono">
                {tx.speed ? `${t} ${tx.speed}` : `$${t}`}
              </text>
            </g>
          ))}
          <text x={pad.l + (W - pad.l - pad.r) / 2} y={H - 10} textAnchor="middle" fontSize="11" fill={muted}>
            {tx.x} (log scale) →
          </text>
          <text transform={`translate(14 ${pad.t + (H - pad.t - pad.b) / 2}) rotate(-90)`} textAnchor="middle" fontSize="11" fill={muted}>
            {tx.quality} →
          </text>

          {pairs.map(({ base, d }) => {
              const [ax, ay, bx, by] = [sx(base.cost), sy(base.quality), sx(d.cost), sy(d.quality)]
              const len = Math.hypot(bx - ax, by - ay) || 1
              const k = Math.max(0, (len - 9) / len)
              const j = Math.min(1, (L + 2) / len)
              return (
                <g key={d.id} onMouseEnter={() => setHover(d.id)} onMouseLeave={() => setHover(null)}>
                  <line
                    x1={r1(ax + (bx - ax) * j)}
                    y1={r1(ay + (by - ay) * j)}
                    x2={r1(ax + (bx - ax) * k)}
                    y2={r1(ay + (by - ay) * k)}
                    stroke={accent}
                    strokeWidth="1.25"
                    opacity="0.5"
                    markerEnd="url(#ghost-arrow)"
                  />
                  <rect x={bx - 7} y={by - 7} width={14} height={14} rx={3} fill="hsl(var(--background))" stroke={accent} strokeWidth="1.25" />
                  <image href="/favicon.png" x={bx - 5.5} y={by - 5.5} width={11} height={11} />
                  <circle cx={bx} cy={by} r={11} fill="transparent" />
                </g>
              )
            })}
          {plain.map((p) => (
            <g key={p.id} onMouseEnter={() => setHover(p.id)} onMouseLeave={() => setHover(null)}>
              <circle cx={sx(p.cost)} cy={sy(p.quality)} r={12} fill="transparent" />
              <rect x={sx(p.cost) - L} y={sy(p.quality) - L} width={2 * L} height={2 * L} rx={3} fill="hsl(var(--background))" stroke="hsl(var(--border))" />
              {icon(p) ? (
                <image href={icon(p)} x={sx(p.cost) - L + 2} y={sy(p.quality) - L + 2} width={2 * L - 4} height={2 * L - 4} preserveAspectRatio="xMidYMid meet" />
              ) : (
                <text x={sx(p.cost)} y={sy(p.quality) + 3.5} textAnchor="middle" fontSize="10" fontWeight={700} fill={muted}>
                  {p.provider[0]}
                </text>
              )}
            </g>
          ))}
          {labels.map(({ p, cx, cy, x, y, w }) => (
            <g key={p.id}>
              {y !== cy && <line x1={cx} y1={cy} x2={x > cx ? x - 2 : x + w + 2} y2={y - 4} stroke="hsl(var(--border))" />}
              <text
                x={x}
                y={y + 4}
                fontSize="11"
                fill={onFront.has(p.id) ? 'hsl(var(--foreground))' : muted}
                fontWeight={onFront.has(p.id) ? 600 : 400}
                className="font-mono"
              >
                {p.model}
              </text>
            </g>
          ))}
        </svg>
        {hp && (() => {
          // A point with several runs on top (an agent's Hindsight setups) compares from those runs only.
          const mine = pairs.filter(({ base, d }) => base.id === hp.id || d.id === hp.id)
          const other = mine.length === 1 ? (mine[0].d.id === hp.id ? mine[0].base : mine[0].d) : undefined
          const sign = (v: number, digits = 1) => `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`
          return (
            <div
              className="pointer-events-none absolute z-10 w-64 rounded-lg border border-border bg-popover px-3 py-2.5 text-xs shadow-lg"
              style={{ left: sx(hp.cost) + 270 > W ? sx(hp.cost) - 270 : sx(hp.cost) + 16, top: Math.max(0, sy(hp.quality) - 60) }}
            >
              <div className="flex items-center gap-2 mb-2">
                {icon(hp) && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={icon(hp)} alt="" className="w-5 h-5 object-contain" />
                )}
                <div className="min-w-0">
                  <div className="font-mono font-semibold text-foreground truncate">{hp.model}</div>
                  <div className="text-muted-foreground truncate">
                    {hp.provider}
                    {hp.split && <span className="text-[hsl(var(--primary))]"> + {hp.split.decisionModel}</span>}
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono tabular-nums">
                <span className="text-muted-foreground font-sans">Quality</span>
                <span>{hp.quality.toFixed(1)}%</span>
                {hp.score != null && (
                  <>
                    <span className="text-muted-foreground font-sans">Score</span>
                    <span>{hp.score.toFixed(1)}</span>
                  </>
                )}
                <span className="text-muted-foreground font-sans">{tx.speed ? 'Latency' : 'Cost'}</span>
                <span>
                  {tx.speed ? `${hp.cost.toLocaleString('en-US', { maximumFractionDigits: 1 })} ${tx.speed}, median` : `${fmtCost(hp.cost)} per ${tx.unit}`}
                </span>
                {hp.callsPerTask > 0 && (
                  <>
                    <span className="text-muted-foreground font-sans">Calls</span>
                    <span>
                      {hp.callsPerTask.toFixed(1)} per case
                      {hp.decisionCallsPerTask ? ` + ${hp.decisionCallsPerTask.toFixed(1)} decide` : ''}
                    </span>
                  </>
                )}
                {Object.entries(hp.extra ?? {}).map(([k, v]) => (
                  <Fragment key={k}>
                    <span className="text-muted-foreground font-sans">{k}</span>
                    <span>{v}</span>
                  </Fragment>
                ))}
              </div>
              {other && (() => {
                const [w, wo] = hp.split ? [hp, other] : [other, hp]
                const dq = w.quality - wo.quality
                const dc = (w.cost / wo.cost - 1) * 100
                return (
                  <div className="mt-2 pt-2 border-t border-border font-mono tabular-nums">
                    <span className="text-muted-foreground font-sans">{hp.split ? (tx.noun === 'models' ? 'Against the model alone: ' : 'Against no memory: ') : `With ${w.split?.decisionModel}: `}</span>
                    <span className={dq >= 0 ? 'text-emerald-500' : 'text-red-500'}>{sign(dq)} quality</span>,{' '}
                    <span className={dc <= 0 ? 'text-emerald-500' : 'text-red-500'}>{sign(dc, 0)}% cost</span>
                  </div>
                )
              })()}
              {onFront.has(hp.id) && <div className="text-[hsl(var(--primary))] mt-1.5">Best value at this cost</div>}
            </div>
          )
        })()}
      </div>
  )
  if (bare) return chart
  return (
    <section className="rounded-xl border border-border p-5">
      <h3 className="font-heading font-bold text-base">
        {tx.title ?? `Quality vs. cost across ${plain.length} ${tx.noun} on ${task}`}
      </h3>
      <p className="text-xs text-muted-foreground mb-3">Top left is best value. Hover a point for details.</p>
      {chart}
      <div className="flex flex-wrap gap-x-5 gap-y-1 mt-2 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <svg width="12" height="12"><rect x="0.5" y="0.5" width="11" height="11" rx="2.5" fill="none" stroke="hsl(var(--border))" /></svg>
          {tx.noun === 'models' ? 'Model, as its provider’s logo' : 'Agent with no memory'}
        </span>
        {tx.pair && (
          <span className="inline-flex items-center gap-1.5">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/favicon.png" alt="" className="w-3.5 h-3.5 rounded-[3px] border border-[hsl(var(--primary))] p-[1px]" />
            {tx.pair}
          </span>
        )}
        <span className="inline-flex items-center gap-1.5">
          <svg width="12" height="10"><rect width="12" height="10" fill={accent} opacity="0.12" /></svg>
          Best value: no other {tx.noun === 'agents' ? 'agent' : 'model'} is both {tx.speed ? 'faster' : 'cheaper'} and better
        </span>
      </div>
      {tx.costNote && <p className="text-xs text-muted-foreground mt-1">{tx.costNote}</p>}
    </section>
  )
}

// Where a row's numbers come from, so a reader can check the price against the list it was taken from.
function SourceDetails({ row, source }: { row: Row; source: Source }) {
  const p = source.price_per_1m
  const usd = (v: number) => `$${v < 0.1 ? v.toFixed(3) : v.toFixed(2)}`
  return (
    <div className="text-sm">
      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">Source</div>
      <div className="grid sm:grid-cols-2 gap-x-8 gap-y-1">
        <div>
          <span className="text-muted-foreground">Price list:</span>{' '}
          <a href={`https://openrouter.ai/${source.openrouter_id}`} className="font-mono text-[13px] text-[hsl(var(--primary))] hover:underline">
            openrouter.ai/{source.openrouter_id}
          </a>
        </div>
        <div>
          <span className="text-muted-foreground">{row.model} per 1M tokens:</span>{' '}
          <span className="font-mono text-[13px]">
            {usd(p.input)} in{p.cached_input !== p.input && ` (${usd(p.cached_input)} cached)`}, {usd(p.output)} out
          </span>
        </div>
        <div>
          <span className="text-muted-foreground">{row.model} size:</span>{' '}
          <span className="font-mono text-[13px]">{source.params ?? `not published by ${row.provider}`}</span>
        </div>
        {row.split && source.decision_params && (
          <div>
            <span className="text-muted-foreground">{row.split.decisionModel} size:</span>{' '}
            <span className="font-mono text-[13px]">{source.decision_params}</span>
          </div>
        )}
        {row.split && source.decision_price_per_1m_input != null && (
          <div>
            <span className="text-muted-foreground">{row.split.decisionModel} per 1M tokens:</span>{' '}
            <span className="font-mono text-[13px]">{usd(source.decision_price_per_1m_input)} in, output free</span>
          </div>
        )}
        <div>
          <span className="text-muted-foreground">Ran:</span>{' '}
          <span className="font-mono text-[13px]">
            {source.ran}, {source.repeats} passes, judged by {source.judge.split('/').pop()}
          </span>
        </div>
      </div>
    </div>
  )
}
