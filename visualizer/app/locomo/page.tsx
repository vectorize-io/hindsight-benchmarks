import Link from 'next/link'
import { loadLoComo } from '@/lib/data'
import { notFound } from 'next/navigation'
import { Button } from '@/components/ui/button'

export default function LoComoPage() {
  const data = loadLoComo()
  if (!data) notFound()

  const categoryStats: Record<number, { name: string; correct: number; total: number; invalid: number }> = {
    1: { name: 'Multi-hop', correct: 0, total: 0, invalid: 0 },
    2: { name: 'Single-hop', correct: 0, total: 0, invalid: 0 },
    3: { name: 'Temporal', correct: 0, total: 0, invalid: 0 },
    4: { name: 'Open-domain', correct: 0, total: 0, invalid: 0 },
  }

  let totalInvalid = 0
  data.item_results.forEach((item) => {
    item.metrics.detailed_results?.forEach((result) => {
      const category = result.category
      if (typeof category === 'number' && category in categoryStats) {
        categoryStats[category].total++
        if (result.is_invalid) {
          categoryStats[category].invalid++
          totalInvalid++
        } else if (result.is_correct) {
          categoryStats[category].correct++
        }
      }
    })
  })

  return (
    <main className="container mx-auto max-w-5xl px-4 py-8">
      <Button variant="ghost" size="sm" asChild className="mb-8 -ml-2 text-muted-foreground">
        <Link href="/">← Back</Link>
      </Button>

      <div className="mb-10">
        <h1 className="text-3xl font-heading font-bold mb-1">
          <span className="gradient-primary-text">LoComo</span>
          <span className="text-foreground"> Benchmark</span>
        </h1>
        <p className="text-sm text-muted-foreground">Long conversation memory evaluation</p>
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
            ...(totalInvalid > 0 ? [{ label: 'Invalid', value: totalInvalid }] : []),
            { label: 'Conversations', value: data.num_items },
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
      <section className="mb-10">
        <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-3">
          By Category
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-border rounded-lg overflow-hidden">
          {Object.values(categoryStats).map((cat) => {
            const validTotal = cat.total - cat.invalid
            const accuracy = validTotal > 0 ? (cat.correct / validTotal) * 100 : 0
            return (
              <div key={cat.name} className="bg-card px-5 py-4">
                <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-1">{cat.name}</p>
                <p className="text-2xl font-bold text-foreground tabular-nums">{accuracy.toFixed(1)}%</p>
                <p className="text-xs text-muted-foreground mt-0.5">{cat.correct} / {cat.total}</p>
              </div>
            )
          })}
        </div>
      </section>

      {/* Conversations */}
      <section>
        <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-3">
          Conversations <span className="text-muted-foreground/50 font-normal normal-case tracking-normal">({data.num_items})</span>
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-border rounded-lg overflow-hidden">
          {data.item_results.map((item, idx) => {
            const accuracy = item.metrics.accuracy
            const accentClass =
              accuracy >= 70 ? 'border-l-emerald-500' : accuracy >= 50 ? 'border-l-amber-500' : 'border-l-red-500'

            return (
              <Link
                key={item.item_id}
                href={`/locomo/${idx}`}
                className={`group bg-card border-l-2 ${accentClass} px-5 py-4 hover:bg-secondary/30 transition-colors`}
              >
                <p className="text-xs font-medium text-muted-foreground mb-2 truncate">{item.item_id}</p>
                <div className="flex items-baseline gap-3">
                  <span className="text-xl font-bold text-foreground tabular-nums">{accuracy.toFixed(1)}%</span>
                  <span className="text-xs text-muted-foreground">{item.metrics.correct}/{item.metrics.total} correct</span>
                </div>
              </Link>
            )
          })}
        </div>
      </section>
    </main>
  )
}
