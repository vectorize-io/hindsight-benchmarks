import Link from 'next/link'
import { loadLongMemEval } from '@/lib/data'
import { notFound } from 'next/navigation'
import { ItemList } from '@/components/item-list'
import { Button } from '@/components/ui/button'

export default function LongMemEvalPage() {
  const data = loadLongMemEval()
  if (!data) notFound()

  const categoryStats: Record<string, { correct: number; total: number; invalid: number }> = {}

  data.item_results.forEach((item) => {
    item.metrics.detailed_results?.forEach((result) => {
      const category = typeof result.category === 'number' ? result.category.toString() : result.category
      if (!categoryStats[category]) {
        categoryStats[category] = { correct: 0, total: 0, invalid: 0 }
      }
      categoryStats[category].total++
      if (result.is_invalid) {
        categoryStats[category].invalid++
      } else if (result.is_correct) {
        categoryStats[category].correct++
      }
    })
  })

  const categoryEntries = Object.entries(categoryStats).sort(([a], [b]) => a.localeCompare(b))

  return (
    <main className="container mx-auto max-w-5xl px-4 py-8">
      <Button variant="ghost" size="sm" asChild className="mb-8 -ml-2 text-muted-foreground">
        <Link href="/">← Back</Link>
      </Button>

      <div className="mb-10">
        <h1 className="text-3xl font-heading font-bold mb-1">
          <span className="gradient-primary-text">LongMemEval</span>
          <span className="text-foreground"> Benchmark</span>
        </h1>
        <p className="text-sm text-muted-foreground">Long-term memory evaluation across extended conversations</p>
      </div>

      {/* Overall stats */}
      <section className="mb-10">
        <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-3">
          Overall
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-border rounded-lg overflow-hidden">
          {[
            { label: 'Accuracy', value: `${data.overall_accuracy.toFixed(2)}%`, highlight: true },
            { label: 'Correct', value: `${data.total_correct} / ${data.total_questions}` },
            ...(data.total_invalid > 0 ? [{ label: 'Invalid', value: data.total_invalid }] : []),
            { label: 'Items', value: data.num_items },
          ].map((s) => (
            <div key={s.label} className="bg-card px-5 py-4">
              <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-1">{s.label}</p>
              <p className={`text-2xl font-bold tabular-nums ${s.highlight ? 'gradient-primary-text' : 'text-foreground'}`}>
                {s.value}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* By category */}
      {categoryEntries.length > 0 && (
        <section className="mb-10">
          <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-3">
            By Category
          </p>
          <div
            className="grid gap-px bg-border rounded-lg overflow-hidden"
            style={{ gridTemplateColumns: `repeat(${Math.min(categoryEntries.length, 4)}, minmax(0, 1fr))` }}
          >
            {categoryEntries.map(([category, stats]) => {
              const validTotal = stats.total - stats.invalid
              const accuracy = validTotal > 0 ? (stats.correct / validTotal) * 100 : 0
              return (
                <div key={category} className="bg-card px-5 py-4">
                  <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-1 truncate">{category}</p>
                  <p className="text-2xl font-bold text-foreground tabular-nums">{accuracy.toFixed(1)}%</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{stats.correct} / {stats.total}</p>
                </div>
              )
            })}
          </div>
        </section>
      )}

      {/* Items */}
      <section>
        <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-3">
          Items <span className="text-muted-foreground/50 font-normal normal-case tracking-normal">({data.num_items})</span>
        </p>
        <ItemList items={data.item_results} basePath="/longmemeval" showCategories={true} />
      </section>
    </main>
  )
}
