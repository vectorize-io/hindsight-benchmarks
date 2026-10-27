'use client'

import { Fragment, useState } from 'react'
import { ChevronDown, ArrowDown, Info } from 'lucide-react'
import type { Board, Row, Tab } from '@/lib/data'
import { PAIRS, SCORE } from '@/lib/data'
import { cn } from '@/lib/utils'
import { ModelCell, fmtCost, fmtTokens } from './ModelCell'

function QualityBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <span className="font-mono text-sm font-semibold tabular-nums w-11 text-right">{value.toFixed(1)}</span>
      <div className="h-1.5 flex-1 rounded-full bg-secondary overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${value}%`, background: 'linear-gradient(90deg, #0074d9, #009296)' }} />
      </div>
    </div>
  )
}

type SortKey = 'score' | 'quality' | 'cost' | 'speed'

const SCORE_HINT = `Combined score: ${SCORE.quality * 100}% quality, ${SCORE.cost * 100}% cost, ${SCORE.speed * 100}% speed. Cost and speed are scored against the reference row, which gets 50 on each.`

function InfoTip({ text }: { text: string }) {
  return (
    <span title={text} aria-label={text} className="inline-flex cursor-help text-muted-foreground hover:text-foreground">
      <Info className="w-3 h-3" />
    </span>
  )
}

export function BoardView({ board }: { board: Board }) {
  const [tabId, setTabId] = useState(board.tabs[0].id)
  const tab = board.tabs.find((t) => t.id === tabId)!

  return (
    <div>
      {board.tabs.length > 1 && (
        <div className="flex flex-wrap gap-1 p-1 rounded-xl bg-secondary w-fit mb-3" role="tablist">
          {board.tabs.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={t.id === tabId}
              onClick={() => setTabId(t.id)}
              className={cn(
                'px-3.5 h-8 rounded-lg text-sm font-heading font-bold transition-colors',
                t.id === tabId ? 'bg-background shadow-sm text-foreground' : 'text-muted-foreground hover:text-foreground',
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
      )}
      <p className="text-sm text-muted-foreground mb-6">
        {tab.description}
        {board.tabs.length > 1 && tab.id !== 'pairs' && ' Rows are ranked within this tab only.'}
      </p>

      {tab.id === 'pairs' ? <PairsGrid /> : <TabView board={board} tab={tab} key={tab.id} />}
    </div>
  )
}

function TabView({ board, tab }: { board: Board; tab: Tab }) {
  const [sort, setSort] = useState<SortKey>('quality')
  const [open, setOpen] = useState<string | null>(null)
  const isRetriever = board.id === 'recall'
  const hasDecision = tab.rows.some((r) => r.decisionCallsPerTask)
  const showTokens = !isRetriever && board.id !== 'decision'

  const rows = [...tab.rows].sort((a, b) =>
    sort === 'score'
      ? (b.score ?? 0) - (a.score ?? 0)
      : sort === 'quality'
      ? b.quality - a.quality || a.cost - b.cost
      : sort === 'cost'
        ? a.cost - b.cost
        : (isRetriever ? 1 : -1) * ((a.speed?.value ?? (isRetriever ? 1e9 : 0)) - (b.speed?.value ?? (isRetriever ? 1e9 : 0))),
  )
  const colCount = 6 + (isRetriever ? 0 : 1) + (showTokens ? 1 : 0) + (hasDecision ? 1 : 0) + (tab.extraColumns?.length ?? 0)

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
  const costTip = isRetriever
    ? 'Tokens per query times the list price on OpenRouter. Local models are free.'
    : 'Tokens each model really used in these evals, times its list price on OpenRouter. Includes decision model calls.'
  const speedTip = 'As stated by OpenRouter, not measured by us.'

  return (
    <div className="space-y-10">
      <div className="rounded-xl border border-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-xs text-muted-foreground text-left bg-[hsl(var(--band))]">
              <th className="px-3 py-2.5 font-medium w-10">#</th>
              <th className="px-3 py-2.5 font-medium">Model</th>
              <SortHead k="score" tip={SCORE_HINT}>Score</SortHead>
              <SortHead k="quality">Quality</SortHead>
              <th className="px-3 py-2.5 font-medium">Cases</th>
              {!isRetriever && <th className="px-3 py-2.5 font-medium text-right whitespace-nowrap">LLM calls</th>}
              {hasDecision && <th className="px-3 py-2.5 font-medium text-right whitespace-nowrap">Decision calls</th>}
              {showTokens && <th className="px-3 py-2.5 font-medium text-right whitespace-nowrap">Tokens in / out</th>}
              {tab.extraColumns?.map((c) => (
                <th key={c.key} className="px-3 py-2.5 font-medium text-right whitespace-nowrap" title={c.hint}>
                  {c.label}
                </th>
              ))}
              <SortHead k="cost" className="text-right" tip={costTip}>
                Cost <span className="font-normal">{board.costUnit}</span>
              </SortHead>
              <SortHead k="speed" className="text-right" tip={speedTip}>
                {isRetriever || board.id === 'decision' ? 'Latency' : 'Speed'}
              </SortHead>
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
                      'border-b border-border last:border-0 cursor-pointer hover:bg-secondary/60 transition-colors',
                      r.tag === 'reference' && 'bg-[hsl(var(--teal))]/[0.05]',
                    )}
                  >
                    <td className="px-3 py-3 font-mono text-xs text-muted-foreground">{sort === 'quality' || sort === 'score' ? i + 1 : ''}</td>
                    <td className="px-3 py-3 max-w-[260px]">
                      <div className="flex items-center gap-2">
                        <ChevronDown className={cn('w-3.5 h-3.5 shrink-0 text-muted-foreground transition-transform', expanded && 'rotate-180')} />
                        <ModelCell row={r} />
                      </div>
                    </td>
                    <td className="px-3 py-3 font-mono text-sm font-semibold tabular-nums">{r.score?.toFixed(1)}</td>
                    <td className="px-3 py-3">
                      <QualityBar value={r.quality} />
                    </td>
                    <td className="px-3 py-3 font-mono text-xs text-muted-foreground tabular-nums whitespace-nowrap">
                      {passed}/{total}
                    </td>
                    {!isRetriever && <td className="px-3 py-3 font-mono text-right tabular-nums">{r.callsPerTask.toFixed(1)}</td>}
                    {hasDecision && <td className="px-3 py-3 font-mono text-right tabular-nums text-muted-foreground">{r.decisionCallsPerTask ?? 0}</td>}
                    {showTokens && (
                      <td className="px-3 py-3 font-mono text-right tabular-nums text-muted-foreground whitespace-nowrap">
                        {fmtTokens(r.tokensPerTask.input)} / {fmtTokens(r.tokensPerTask.output)}
                      </td>
                    )}
                    {tab.extraColumns?.map((c) => (
                      <td key={c.key} className="px-3 py-3 font-mono text-right tabular-nums text-muted-foreground whitespace-nowrap">
                        {r.extra?.[c.key] ?? '—'}
                      </td>
                    ))}
                    <td className="px-3 py-3 font-mono text-right tabular-nums">{fmtCost(r.cost)}</td>
                    <td className="px-3 py-3 text-right whitespace-nowrap">
                      {r.speed ? (
                        <span title={`Stated by ${r.speed.source}, not measured by us`}>
                          <span className="font-mono tabular-nums">{r.speed.value}</span>{' '}
                          <span className="text-xs text-muted-foreground">{r.speed.unit}</span>
                        </span>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                  </tr>
                  {expanded && (
                    <tr className="border-b border-border bg-[hsl(var(--band))]">
                      <td colSpan={colCount} className="px-6 py-5">
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
      <p className="-mt-7 text-xs text-muted-foreground">
        Click a row for its categories and failed cases. Hover <Info className="w-3 h-3 inline" /> for how a column is computed.
      </p>

      <section className="rounded-xl border border-border p-5">
        <h3 className="font-heading font-bold text-base">Quality against cost</h3>
        <p className="text-xs text-muted-foreground mb-3">Top left is best value. The line joins models nothing beats on both.</p>
        <Scatter rows={tab.rows} />
      </section>

      <section>
        <h3 className="font-heading font-bold text-base mb-1">Cases passed by category</h3>
        <p className="text-xs text-muted-foreground mb-4">Where each model wins and loses. Darker is better.</p>
        <Heatmap board={board} rows={rows} />
      </section>

      <section className="rounded-xl border border-border p-5">
        <h3 className="font-heading font-bold text-base mb-1">Categories</h3>
        <p className="text-xs text-muted-foreground mb-4">From the evals at this commit. Click one for its cases.</p>
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
            Also at this commit, measured but not graded as cases:{' '}
            {board.benchmarks.map((b) => <span key={b} className="font-mono">{b} </span>)}
          </p>
        )}
      </section>
    </div>
  )
}

function cellColor(ratio: number) {
  if (ratio >= 0.999) return 'hsl(var(--good) / 0.32)'
  if (ratio >= 0.5) return `hsl(var(--good) / ${0.08 + ratio * 0.18})`
  return `hsl(var(--bad) / ${0.12 + (0.5 - ratio) * 0.3})`
}

function CategoryBreakdown({ board, row }: { board: Board; row: Row }) {
  const failed = board.categories.flatMap((c) => (row.categories[c.id].failed ?? []).map((id) => ({ cat: c.label, id })))
  return (
    <div className="space-y-5">
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
              <td className="px-3 py-1.5 font-mono whitespace-nowrap sticky left-0 bg-background">{r.model}</td>
              {board.categories.map((c) => {
                const v = r.categories[c.id]
                return (
                  <td key={c.id} className="px-1 py-1">
                    <div
                      className="rounded font-mono tabular-nums text-center py-1 min-w-[44px]"
                      style={{ background: cellColor(v.passed / v.total) }}
                      title={`${r.model} · ${c.label}: ${v.passed}/${v.total}`}
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

export function Scatter({ rows, height = 300 }: { rows: Row[]; height?: number }) {
  const W = 640
  const H = height
  const pad = { l: 44, r: 16, t: 12, b: 34 }
  const pts = rows.filter((r) => r.cost > 0)
  if (pts.length < 2) return <p className="text-sm text-muted-foreground">Not enough paid rows to plot.</p>
  const xs = pts.map((r) => Math.log10(r.cost))
  const x0 = Math.min(...xs) - 0.15
  const x1 = Math.max(...xs) + 0.15
  const qs = pts.map((r) => r.quality)
  const y0 = Math.max(0, Math.floor((Math.min(...qs) - 5) / 10) * 10)
  const y1 = 100
  // Rounded: server and browser differ in the last float digits, which breaks hydration.
  const r1 = (v: number) => Math.round(v * 10) / 10
  const sx = (c: number) => r1(pad.l + ((Math.log10(c) - x0) / (x1 - x0)) * (W - pad.l - pad.r))
  const sy = (q: number) => r1(pad.t + (1 - (q - y0) / (y1 - y0)) * (H - pad.t - pad.b))

  // Models no other model beats on both quality and cost.
  const frontier = [...pts].sort((a, b) => a.cost - b.cost || b.quality - a.quality)
  const front: Row[] = []
  for (const p of frontier) if (!front.length || p.quality > front[front.length - 1].quality) front.push(p)
  const labelled = new Set([...front.map((p) => p.id), ...pts.filter((p) => p.tag).map((p) => p.id)])

  const ticks: number[] = []
  for (let e = Math.floor(x0); e <= Math.ceil(x1); e++) for (const m of [1, 3]) {
    const v = m * 10 ** e
    if (Math.log10(v) >= x0 && Math.log10(v) <= x1) ticks.push(v)
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Quality against cost">
      {[...Array(5)].map((_, i) => {
        const q = y0 + ((y1 - y0) * i) / 4
        return (
          <g key={i}>
            <line x1={pad.l} x2={W - pad.r} y1={sy(q)} y2={sy(q)} stroke="hsl(var(--border))" />
            <text x={pad.l - 8} y={sy(q) + 4} textAnchor="end" fontSize="11" fill="hsl(var(--muted-foreground))">
              {Math.round(q)}
            </text>
          </g>
        )
      })}
      {ticks.map((t) => (
        <text key={t} x={sx(t)} y={H - 12} textAnchor="middle" fontSize="11" fill="hsl(var(--muted-foreground))">
          {fmtCost(t)}
        </text>
      ))}
      <polyline
        points={front.map((p) => `${sx(p.cost)},${sy(p.quality)}`).join(' ')}
        fill="none"
        stroke="hsl(var(--primary))"
        strokeWidth="1.5"
        strokeDasharray="4 3"
        opacity="0.7"
      />
      {pts.map((p) => {
        const ref = p.tag === 'reference'
        return (
          <g key={p.id}>
            <circle
              cx={sx(p.cost)}
              cy={sy(p.quality)}
              r={ref ? 6 : 4.5}
              fill={ref ? 'hsl(var(--teal))' : 'hsl(var(--primary))'}
              stroke="hsl(var(--background))"
              strokeWidth="1.5"
            >
              <title>{`${p.model}: ${p.quality.toFixed(1)} quality, ${fmtCost(p.cost)}`}</title>
            </circle>
            {labelled.has(p.id) && (
              <text x={sx(p.cost) + 8} y={sy(p.quality) + 4} fontSize="11" fill="hsl(var(--foreground))" className="font-mono">
                {p.model}
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}

function PairsGrid() {
  const all = PAIRS.quality.flat()
  const lo = Math.min(...all)
  const hi = Math.max(...all)
  return (
    <div className="rounded-xl border border-border overflow-x-auto">
      <table className="text-sm w-full">
        <thead>
          <tr className="border-b border-border bg-[hsl(var(--band))] text-xs text-muted-foreground">
            <th className="px-3 py-2.5 text-left font-medium">Embedder ↓ · Reranker →</th>
            {PAIRS.rerankers.map((r) => (
              <th key={r} className="px-3 py-2.5 font-mono font-medium">{r}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {PAIRS.embedders.map((e, i) => (
            <tr key={e} className="border-b border-border last:border-0">
              <td className="px-3 py-2 font-mono text-[13px]">{e}</td>
              {PAIRS.quality[i].map((q, j) => {
                const best = q === hi
                return (
                  <td key={j} className="px-2 py-2">
                    <div
                      className={cn('rounded-md text-center py-2 font-mono tabular-nums', best && 'ring-2 ring-[hsl(var(--teal))]')}
                      style={{ background: `hsl(var(--good) / ${0.06 + ((q - lo) / (hi - lo || 1)) * 0.3})` }}
                    >
                      {q.toFixed(1)}
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
