import { ArrowUpRight } from 'lucide-react'
import { FAMILIES, type Family, type ModelRow } from '@/lib/models'
import { cn } from '@/lib/utils'
import { ModelScatter, QualityBar } from './BoardView'
import type { Row } from '@/lib/data'
import { PageHero, SubNav } from './Page'
import { ModelCell } from './ModelCell'

// The model leaderboards' parts (setup strip, table, categories list), for one model.
export function ModelsView({ family: f }: { family: Family }) {
  const rows = f.rows
  // Columns in the card's order, grouped by its sections.
  const benchmarks = f.benchmarks
  // Best per column, ties all win; lower wins where the card's metric says so (latency).
  const best = Object.fromEntries(
    benchmarks.map((b) => {
      const vs = rows.map((r) => r.scores[b.id]).filter((x): x is number => x != null)
      return [b.id, b.lower ? Math.min(...vs) : Math.max(...vs)]
    }),
  )
  // Score columns only: a speed or a size is no score.
  const scored = benchmarks.filter((b) => !b.plain)
  const sections = benchmarks.reduce<{ name: string; n: number }[]>((out, b) => {
    const last = out[out.length - 1]
    if (last?.name === b.section) last.n++
    else out.push({ name: b.section, n: 1 })
    return out
  }, [])
  // Overall: the mean over the score columns a model was measured on. Only models measured on every one compete for best.
  const overall = (r: ModelRow) => {
    const v = scored.map((b) => r.scores[b.id]).filter((x): x is number => x != null)
    return { mean: v.length ? v.reduce((a, b) => a + b, 0) / v.length : undefined, n: v.length }
  }
  const showOverall = scored.length > 1
  const bestOverall = Math.max(
    ...rows
      .map((r) => overall(r))
      .filter((o) => o.n === scored.length)
      .map((o) => o.mean!),
  )
  const winCell = (ours: boolean) =>
    ours
      ? 'bg-[hsl(var(--primary)/0.14)] shadow-[inset_0_0_0_1.5px_hsl(var(--primary))]'
      : 'bg-foreground/[0.07] shadow-[inset_0_0_0_1.5px_hsl(var(--foreground)/0.35)]'
  return (
    <>
      <div className="grid gap-4 md:grid-cols-[auto_1fr] md:items-center rounded-xl border border-border p-4 sm:p-5">
        <div className="md:pr-6 md:border-r border-border">
          <div className="text-xs text-muted-foreground mb-1">{f.title}</div>
          <div className="font-heading font-bold text-lg font-mono">{f.model}</div>
          <a href={f.card} className="text-xs font-medium text-[hsl(var(--primary))] inline-flex items-center gap-1 mt-1">
            Model card <ArrowUpRight className="w-3 h-3" />
          </a>
        </div>
        <div>
          <div className="text-xs text-muted-foreground mb-2">{f.short}</div>
          <ul className="grid sm:grid-cols-2 gap-x-8 gap-y-1 text-sm list-disc pl-5 marker:text-muted-foreground">
            {f.facts.map(([k, v]) => (
              <li key={k}>
                <span className="text-muted-foreground">{k}:</span> <span className="font-mono text-[13px]">{v}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {benchmarks.length > 0 && (
        <>
          <div className="rounded-xl border border-border overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                {sections.some((x) => x.name) && (
                  <tr className="text-[11px] uppercase tracking-wide text-muted-foreground text-left bg-[hsl(var(--band))]">
                    <th colSpan={showOverall ? 2 : 1} />
                    {sections.map((x, i) => (
                      <th key={i} colSpan={x.n} className="px-3 pt-2.5 font-semibold whitespace-nowrap border-l border-border">
                        {x.name}
                      </th>
                    ))}
                  </tr>
                )}
                <tr className="border-b border-border text-xs text-muted-foreground text-left bg-[hsl(var(--band))]">
                  <th className="px-3 py-2.5 font-medium">Model</th>
                  {showOverall && (
                    <th className="px-3 py-2.5 font-medium whitespace-nowrap" title="Mean over the columns a model was measured on">
                      Overall <span className="font-normal">mean</span>
                    </th>
                  )}
                  {benchmarks.map((b) => (
                    <th key={b.id} className="px-3 py-2.5 font-medium whitespace-nowrap" title={b.about}>
                      {b.href ? (
                        <a href={b.href ?? undefined} className="inline-flex items-center gap-1 hover:text-foreground">
                          {b.name} <ArrowUpRight className="w-3 h-3" />
                        </a>
                      ) : (
                        b.name
                      )}
                      <span className="font-normal"> {b.metric}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.model} className={cn('border-b border-border last:border-0', r.tag === 'hindsight' && 'bg-[hsl(var(--primary)/0.04)]')}>
                    <td className="px-3 py-3 max-w-[260px]">
                      <ModelCell row={r} />
                    </td>
                    {showOverall &&
                      (() => {
                        const o = overall(r)
                        const won = o.n === scored.length && o.mean === bestOverall
                        return (
                          <td className={cn('px-3 py-3 border-r border-border', won && winCell(r.tag === 'hindsight'))}>
                            {o.mean == null ? (
                              <span className="font-mono text-xs text-muted-foreground">—</span>
                            ) : (
                              <>
                                <QualityBar value={o.mean} />
                                {o.n < scored.length && (
                                  <div className="text-[11px] text-muted-foreground mt-0.5">
                                    {o.n} of {scored.length} measured
                                  </div>
                                )}
                              </>
                            )}
                          </td>
                        )
                      })()}
                    {benchmarks.map((b) => {
                      const v = r.scores[b.id]
                      const won = v != null && v === best[b.id]
                      return (
                        <td key={b.id} title={won ? 'Best in this column' : undefined} className={cn('px-3 py-3', won && winCell(r.tag === 'hindsight'))}>
                          {v == null ? (
                            <span className="font-mono text-xs text-muted-foreground">—</span>
                          ) : b.plain ? (
                            <span className="font-mono text-[13px] tabular-nums whitespace-nowrap">
                              {v.toLocaleString('en-US', { maximumFractionDigits: 1 })}
                            </span>
                          ) : (
                            <QualityBar value={v} />
                          )}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {(() => {
            // Quality against speed, the model boards' chart with latency for cost. Quality is the overall
            // mean, so only models measured on every score column are plotted.
            const lat = benchmarks.find((b) => b.id.includes('latency'))
            if (!lat || !showOverall) return null
            const points: Row[] = rows.flatMap((r) => {
              const o = overall(r)
              const ms = r.scores[lat.id]
              if (o.n < scored.length || o.mean == null || ms == null) return []
              return [{ id: r.model, model: r.model, provider: r.provider, quality: Math.round(o.mean * 10) / 10, cost: ms, callsPerTask: 0, tokensPerTask: { input: 0, output: 0 }, categories: {} }]
            })
            const left = rows.length - points.length
            return (
              <ModelScatter
                rows={points}
                task={f.title}
                text={{
                  title: `Quality vs. speed across ${points.length} models`,
                  noun: 'models',
                  x: `${lat.name} ${lat.metric}`,
                  unit: lat.name.toLowerCase(),
                  quality: 'Overall mean',
                  ms: true,
                  costNote: `Speed from the ${lat.name} column, on CPU.${left ? ` ${left} model${left > 1 ? 's' : ''} not measured on every column ${left > 1 ? 'are' : 'is'} left out.` : ''}`,
                }}
              />
            )
          })()}

          <section className="rounded-xl border border-border p-5">
            <h3 className="font-heading font-bold text-base mb-1">Benchmarks</h3>
            <p className="text-xs text-muted-foreground mb-4">
              Scores are shown 0 to 100, higher is better; speed and input size as measured, with lower latency better. A shaded, outlined cell is the best in
              its column: blue when it is ours. Overall is the mean over the score columns.
            </p>
            <ul className="grid md:grid-cols-2 gap-x-10 gap-y-1">
              {benchmarks.map((b) => (
                <li key={b.id} className="text-sm flex gap-3 py-1">
                  <a href={b.href ?? undefined} className={cn('font-medium w-32 shrink-0', b.href && 'hover:text-[hsl(var(--primary))]')}>
                    {b.name}
                  </a>
                  <span className="text-muted-foreground flex-1">{b.about}</span>
                </li>
              ))}
            </ul>
          </section>
        </>
      )}
    </>
  )
}

// /models shows the first model; /models/<id> each one, so a section can be linked directly.
export function ModelsPage({ family }: { family: Family }) {
  return (
    <>
      <PageHero
        title="Hindsight Models"
        subtitle="Our decision, embedding and reranking models on public industry benchmarks and on memories Hindsight wrote, next to the models people use today."
      />
      <SubNav items={FAMILIES.map((f) => ({ href: `/models/${f.id}`, title: f.title }))} current={`/models/${family.id}`} />
      <main className="max-w-6xl mx-auto px-4 pt-8 space-y-8">
        <ModelsView family={family} />
      </main>
    </>
  )
}
