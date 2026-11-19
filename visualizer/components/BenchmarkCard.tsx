import type { Card, Panel, Row } from '@/lib/benchmarks'
import { gain, ordinal, rank } from '@/lib/benchmarks'
import { cn } from '@/lib/utils'
import { CostChart } from '@/components/CostChart'

// Rows a panel shows before "Show all": the best ones, plus Hindsight wherever it lands.
const SHOWN = 8
const pct = (v: number) => (v * 100).toFixed(1)
// Published numbers (not rerun on AMB) are striped, so they read apart from the ones AMB ran.
const STRIPES = 'repeating-linear-gradient(135deg, hsl(var(--muted-foreground) / 0.32) 0 6px, hsl(var(--muted-foreground) / 0.16) 6px 9px)'

function Bar({ r }: { r: Row }) {
  return (
    <a
      href={r.href}
      title={r.title ? `${r.label}: ${r.title}` : r.label}
      className="flex items-center gap-3 rounded-md -mx-1.5 px-1.5 py-[3px] hover:bg-secondary/60 transition-colors"
    >
      <span className={cn('w-32 sm:w-44 shrink-0 text-[13px] truncate', r.ours ? 'font-semibold' : 'text-muted-foreground')}>{r.label}</span>
      <div className="flex-1 h-5 rounded bg-secondary overflow-hidden">
        <div
          className="h-full rounded"
          style={{
            width: `${r.value * 100}%`,
            background: r.ours ? 'linear-gradient(90deg, #0074d9, #009296)' : r.published ? STRIPES : 'hsl(var(--muted-foreground) / 0.38)',
          }}
        />
      </div>
      <span className={cn('w-11 shrink-0 tabular-nums text-right text-[13px]', r.ours && 'font-heading font-bold')}>{pct(r.value)}</span>
    </a>
  )
}

function Bars({ p }: { p: Panel }) {
  const top = p.rows.slice(0, SHOWN)
  const extra = p.rows.slice(SHOWN)
  const oursBelow = extra.filter((r) => r.ours)
  return (
    <div className="space-y-0.5">
      {[...top, ...oursBelow].map((r) => (
        <Bar key={r.label} r={r} />
      ))}
      {extra.length > oursBelow.length && (
        <details className="group">
          <summary className="cursor-pointer list-none text-xs text-muted-foreground hover:text-foreground py-1 select-none">
            <span className="group-open:hidden">Show all {p.rows.length}</span>
            <span className="hidden group-open:inline">Show fewer</span>
          </summary>
          {extra.filter((r) => !r.ours).map((r) => (
            <Bar key={r.label} r={r} />
          ))}
        </details>
      )}
    </div>
  )
}

function Rank({ p }: { p: Panel }) {
  const k = rank(p)
  if (!k) return <span className="text-[11px] text-muted-foreground whitespace-nowrap">{gain(p) ?? 'Hindsight only'}</span>
  return (
    <span
      className={cn(
        'text-[11px] font-semibold px-2 py-0.5 rounded-full whitespace-nowrap',
        k.place === 1 ? 'text-white' : 'bg-secondary text-foreground',
      )}
      style={k.place === 1 ? { background: 'linear-gradient(90deg, #0074d9 0%, #009296 100%)' } : undefined}
    >
      {ordinal(k.place)} of {k.of}
    </span>
  )
}

export function BenchmarkCard({ c, n, children }: { c: Card; n: number; children?: React.ReactNode }) {
  const multi = c.panels.length > 1
  return (
    <article className={cn('rounded-xl border border-border p-5 flex flex-col', multi && 'md:col-span-2')}>
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground font-mono">
            {c.bench} · {c.metric}
          </div>
          <h3 className="font-heading font-bold text-lg leading-snug mt-0.5">
            {c.title}
            <sup className="ml-0.5 text-[10px] text-muted-foreground font-normal">
              <a href={`#note-${n}`}>{n}</a>
            </sup>
          </h3>
        </div>
        {!multi && <Rank p={c.panels[0]} />}
      </div>
      <p className="text-sm text-muted-foreground mt-1 mb-4">{c.about}</p>
      <div className={cn(multi && 'grid sm:grid-cols-2 gap-x-8 gap-y-5', c.panels.length === 3 && 'lg:grid-cols-3')}>
        {c.panels.map((p) => (
          <div key={p.label ?? 'all'}>
            {multi && (
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-medium">{p.label}</span>
                <Rank p={p} />
              </div>
            )}
            <Bars p={p} />
          </div>
        ))}
      </div>
      {c.cost && c.cost.dots.length > 1 && (
        <div className="mt-5 pt-4 border-t border-border">
          <CostChart cost={c.cost} />
        </div>
      )}
      {children && <div className="mt-5 pt-4 border-t border-border">{children}</div>}
    </article>
  )
}
