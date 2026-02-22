import Link from 'next/link'
import { loadLongMemEval } from '@/lib/data'
import { notFound } from 'next/navigation'
import { Button } from '@/components/ui/button'

export function generateStaticParams() {
  const data = loadLongMemEval()
  if (!data) return []
  return data.item_results.map((_, idx) => ({ itemIdx: idx.toString() }))
}

export default async function LongMemEvalItemPage({ params }: { params: Promise<{ itemIdx: string }> }) {
  const data = loadLongMemEval()
  if (!data) notFound()

  const { itemIdx: itemIdxStr } = await params
  const itemIdx = parseInt(itemIdxStr)
  if (itemIdx >= data.item_results.length) notFound()

  const item = data.item_results[itemIdx]
  const metrics = item.metrics
  const accuracy = metrics.accuracy
  const detailedResults = metrics.detailed_results || []

  const statCells = [
    { label: 'Accuracy', value: `${accuracy.toFixed(2)}%`, highlight: true },
    { label: 'Correct', value: `${metrics.correct} / ${metrics.total}` },
    ...(metrics.invalid > 0 ? [{ label: 'Invalid', value: metrics.invalid }] : []),
    { label: 'Total', value: metrics.total },
  ]

  return (
    <main className="container mx-auto max-w-5xl px-4 py-8">
      <Button variant="ghost" size="sm" asChild className="mb-8 -ml-2 text-muted-foreground">
        <Link href="/longmemeval">← Back</Link>
      </Button>

      <div className="mb-10">
        <h1 className="text-2xl font-heading font-bold mb-1 text-foreground">{item.item_id}</h1>
        <p className="text-sm text-muted-foreground">LongMemEval — item detail</p>
      </div>

      {/* Stats */}
      <section className="mb-10">
        <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-3">Overview</p>
        <div className="grid gap-px bg-border rounded-lg overflow-hidden" style={{ gridTemplateColumns: `repeat(${statCells.length}, minmax(0, 1fr))` }}>
          {statCells.map((s) => (
            <div key={s.label} className="bg-card px-5 py-4">
              <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-1">{s.label}</p>
              <p className={`text-2xl font-bold tabular-nums ${s.highlight ? 'gradient-primary-text' : 'text-foreground'}`}>{s.value}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Questions */}
      <section>
        <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-3">
          Questions <span className="font-normal normal-case tracking-normal text-muted-foreground/50">({detailedResults.length})</span>
        </p>
        <div className="space-y-2">
          {detailedResults.map((result, qIdx) => {
            const isInvalid = result.is_invalid
            const isCorrect = result.is_correct
            const accentClass = isInvalid
              ? 'border-l-amber-500'
              : isCorrect
              ? 'border-l-emerald-500'
              : 'border-l-red-500'
            const statusLabel = isInvalid ? 'Invalid' : isCorrect ? 'Correct' : 'Incorrect'
            const statusColor = isInvalid
              ? 'text-amber-400'
              : isCorrect
              ? 'text-emerald-400'
              : 'text-red-400'

            return (
              <details key={qIdx} className={`group bg-card border border-border border-l-2 ${accentClass} rounded-lg`}>
                <summary className="flex items-center justify-between px-5 py-4 cursor-pointer list-none select-none">
                  <div className="flex items-center gap-4 min-w-0">
                    <span className="text-xs font-mono text-muted-foreground shrink-0">Q{qIdx + 1}</span>
                    <span className="text-sm text-foreground truncate">{result.question}</span>
                  </div>
                  <div className="flex items-center gap-3 ml-4 shrink-0">
                    <span className={`text-xs font-semibold ${statusColor}`}>{statusLabel}</span>
                    <span className="text-xs text-muted-foreground uppercase tracking-wide">{result.category}</span>
                    <svg className="w-4 h-4 text-muted-foreground transition-transform group-open:rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </summary>

                <div className="px-5 pb-5 border-t border-border pt-4 space-y-4">
                  {/* Question */}
                  <div>
                    <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-1">Question</p>
                    <p className="text-sm text-foreground">{result.question}</p>
                  </div>

                  {/* Answers */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-border rounded-md overflow-hidden">
                    <div className="bg-card px-4 py-3">
                      <p className="text-[11px] font-semibold tracking-widest uppercase text-emerald-500 mb-1">Expected</p>
                      <p className="text-sm text-foreground">{result.correct_answer}</p>
                    </div>
                    <div className="bg-card px-4 py-3">
                      <p className={`text-[11px] font-semibold tracking-widest uppercase mb-1 ${isCorrect ? 'text-emerald-500' : isInvalid ? 'text-amber-500' : 'text-red-500'}`}>
                        Predicted
                      </p>
                      <p className="text-sm text-foreground">{result.predicted_answer}</p>
                    </div>
                  </div>

                  {/* Reasoning */}
                  {result.reasoning && (
                    <div>
                      <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-1">Reasoning</p>
                      <pre className="text-xs text-muted-foreground bg-background border border-border rounded-md p-3 overflow-x-auto whitespace-pre-wrap font-mono">{result.reasoning}</pre>
                    </div>
                  )}

                  {result.correctness_reasoning && (
                    <div>
                      <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-1">Judge Reasoning</p>
                      <pre className="text-xs text-muted-foreground bg-background border border-border rounded-md p-3 overflow-x-auto whitespace-pre-wrap font-mono">{result.correctness_reasoning}</pre>
                    </div>
                  )}

                  {/* Memories */}
                  {result.retrieved_memories && result.retrieved_memories.length > 0 && (
                    <div>
                      <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-2">
                        Retrieved Memories ({result.retrieved_memories.length})
                      </p>
                      <div className="space-y-1">
                        {result.retrieved_memories.map((mem, i) => (
                          <div key={i} className="bg-secondary/40 border border-border rounded-md px-3 py-2">
                            <p className="text-[10px] font-semibold tracking-widest uppercase text-muted-foreground mb-0.5">
                              {mem.fact_type?.toUpperCase() || 'FACT'}
                              {mem.occurred_start && ` · ${mem.occurred_start.slice(0, 10)}`}
                            </p>
                            <p className="text-sm text-foreground">{mem.text}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </details>
            )
          })}
        </div>
      </section>
    </main>
  )
}
