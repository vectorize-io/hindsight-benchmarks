import type { Card, Panel, Row } from '@/lib/benchmarks'
import { cn } from '@/lib/utils'

// Rows a panel shows before "Show all": the best ones, plus Hindsight wherever it lands.
const SHOWN = 8
const pct = (v: number) => (v * 100).toFixed(1)
// Published numbers (not rerun on AMB) are striped, so they read apart from the ones AMB ran.
const STRIPES = 'repeating-linear-gradient(135deg, hsl(var(--muted-foreground) / 0.32) 0 6px, hsl(var(--muted-foreground) / 0.16) 6px 9px)'

function Bar({ r, priced }: { r: Row; priced: boolean }) {
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
      {priced && <span className={cn('w-12 shrink-0 tabular-nums text-right text-[12px]', r.ours ? 'font-semibold text-[#009296]' : 'text-muted-foreground')}>{r.price ?? ''}</span>}
    </a>
  )
}

function Bars({ p }: { p: Panel }) {
  const top = p.rows.slice(0, SHOWN)
  const extra = p.rows.slice(SHOWN)
  const oursBelow = extra.filter((r) => r.ours)
  const priced = p.rows.some((r) => r.price)
  return (
    <div className="space-y-0.5">
      {[...top, ...oursBelow].map((r) => (
        <Bar key={r.label} r={r} priced={priced} />
      ))}
      {extra.length > oursBelow.length && (
        <details className="group">
          <summary className="cursor-pointer list-none text-xs text-muted-foreground hover:text-foreground py-1 select-none">
            <span className="group-open:hidden">Show all {p.rows.length}</span>
            <span className="hidden group-open:inline">Show fewer</span>
          </summary>
          {extra.filter((r) => !r.ours).map((r) => (
            <Bar key={r.label} r={r} priced={priced} />
          ))}
        </details>
      )}
    </div>
  )
}

export function BenchmarkCard({ c, n, children }: { c: Card; n: number; children?: React.ReactNode }) {
  const multi = c.panels.length > 1
  return (
    <article className={cn('rounded-xl border border-border p-5 flex flex-col', multi && 'md:col-span-2')}>
      <div>
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
      </div>
      <p className="text-sm text-muted-foreground mt-1 mb-4">{c.about}</p>
      <div className={cn(multi && 'grid sm:grid-cols-2 gap-x-8 gap-y-5', c.panels.length === 3 && 'lg:grid-cols-3')}>
        {c.panels.map((p) => (
          <div key={p.label ?? 'all'}>
            {multi && (
              <div className="text-xs font-medium mb-1.5">{p.label}</div>
            )}
            <Bars p={p} />
          </div>
        ))}
      </div>
      {children && <div className="mt-5 pt-4 border-t border-border">{children}</div>}
    </article>
  )
}
