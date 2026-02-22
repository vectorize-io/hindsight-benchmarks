import Link from 'next/link'
import { loadLoComo } from '@/lib/data'
import { notFound } from 'next/navigation'
import { QuestionList } from '@/components/question-list'
import { Button } from '@/components/ui/button'

export function generateStaticParams() {
  const data = loadLoComo()
  if (!data) return []
  return data.item_results.map((_, idx) => ({ itemIdx: idx.toString() }))
}

export default async function LoComoItemPage({ params }: { params: Promise<{ itemIdx: string }> }) {
  const data = loadLoComo()
  if (!data) notFound()

  const { itemIdx: itemIdxStr } = await params
  const itemIdx = parseInt(itemIdxStr)
  if (itemIdx >= data.item_results.length) notFound()

  const item = data.item_results[itemIdx]
  const metrics = item.metrics
  const accuracy = metrics.accuracy
  const detailedResults = metrics.detailed_results || []

  const categoryStats: Record<number, { name: string; correct: number; total: number; invalid: number }> = {
    1: { name: 'Multi-hop', correct: 0, total: 0, invalid: 0 },
    2: { name: 'Single-hop', correct: 0, total: 0, invalid: 0 },
    3: { name: 'Temporal', correct: 0, total: 0, invalid: 0 },
    4: { name: 'Open-domain', correct: 0, total: 0, invalid: 0 },
  }

  detailedResults.forEach((result) => {
    const category = result.category
    if (typeof category === 'number' && category in categoryStats) {
      categoryStats[category].total++
      if (result.is_invalid) categoryStats[category].invalid++
      else if (result.is_correct) categoryStats[category].correct++
    }
  })

  const activeCats = Object.values(categoryStats).filter((c) => c.total > 0)

  const statCells = [
    { label: 'Accuracy', value: `${accuracy.toFixed(2)}%`, highlight: true },
    { label: 'Correct', value: `${metrics.correct} / ${metrics.total}` },
    ...(metrics.invalid > 0 ? [{ label: 'Invalid', value: metrics.invalid }] : []),
    { label: 'Total', value: metrics.total },
  ]

  return (
    <main className="container mx-auto max-w-5xl px-4 py-8">
      <Button variant="ghost" size="sm" asChild className="mb-8 -ml-2 text-muted-foreground">
        <Link href="/locomo">← Back</Link>
      </Button>

      <div className="mb-10">
        <h1 className="text-2xl font-heading font-bold mb-1 text-foreground">{item.item_id}</h1>
        <p className="text-sm text-muted-foreground">LoComo — conversation detail</p>
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

      {/* By category */}
      {activeCats.length > 0 && (
        <section className="mb-10">
          <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-3">By Category</p>
          <div className="grid gap-px bg-border rounded-lg overflow-hidden" style={{ gridTemplateColumns: `repeat(${activeCats.length}, minmax(0, 1fr))` }}>
            {activeCats.map((cat) => {
              const validTotal = cat.total - cat.invalid
              const catAccuracy = validTotal > 0 ? (cat.correct / validTotal) * 100 : 0
              return (
                <div key={cat.name} className="bg-card px-5 py-4">
                  <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-1">{cat.name}</p>
                  <p className="text-2xl font-bold text-foreground tabular-nums">{catAccuracy.toFixed(1)}%</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{cat.correct} / {cat.total}</p>
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* Questions */}
      <section>
        <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-3">
          Questions <span className="font-normal normal-case tracking-normal text-muted-foreground/50">({detailedResults.length})</span>
        </p>
        <QuestionList questions={detailedResults} />
      </section>
    </main>
  )
}
