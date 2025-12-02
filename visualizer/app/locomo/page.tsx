import Link from 'next/link'
import { loadLoComo, getCategoryName } from '@/lib/data'
import { notFound } from 'next/navigation'

export default function LoComoPage() {
  const data = loadLoComo()

  if (!data) {
    notFound()
  }

  // Calculate category stats
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
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <Link
        href="/"
        className="inline-flex items-center px-4 py-2 bg-white border border-border rounded-md text-sm font-medium text-foreground hover:bg-accent mb-6 transition-colors"
      >
        ← Back to benchmarks
      </Link>

      <h1 className="text-2xl font-bold text-foreground mb-6">LoComo Benchmark - Overall Performance</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Overall Accuracy</p>
          <p className="text-3xl font-bold text-foreground">{data.overall_accuracy.toFixed(2)}%</p>
        </div>

        <div className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Correct Answers</p>
          <p className="text-3xl font-bold text-foreground">
            {data.total_correct} / {data.total_questions}
          </p>
        </div>

        {totalInvalid > 0 && (
          <div className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Invalid Questions</p>
            <p className="text-3xl font-bold text-foreground">{totalInvalid}</p>
          </div>
        )}

        <div className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Items</p>
          <p className="text-3xl font-bold text-foreground">{data.num_items}</p>
        </div>
      </div>

      <h2 className="text-xl font-semibold text-foreground mb-4">Accuracy by Category</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {Object.values(categoryStats).map((cat) => {
          const validTotal = cat.total - cat.invalid
          const accuracy = validTotal > 0 ? (cat.correct / validTotal) * 100 : 0
          return (
            <div key={cat.name} className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">{cat.name}</p>
              <p className="text-2xl font-bold text-foreground">{accuracy.toFixed(1)}%</p>
              <p className="text-sm text-muted-foreground mt-1">
                {cat.correct} / {cat.total}
              </p>
            </div>
          )
        })}
      </div>

      <p className="text-sm text-muted-foreground mb-6">Showing {data.num_items} conversations</p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {data.item_results.map((item, idx) => {
          const accuracy = item.metrics.accuracy
          const color = accuracy >= 70 ? '🟢' : accuracy >= 50 ? '🟡' : '🔴'
          const borderClass =
            accuracy >= 70
              ? 'border-l-4 border-green-600'
              : accuracy >= 50
              ? 'border-l-4 border-yellow-500'
              : 'border-l-4 border-red-600'
          const hoverClass =
            accuracy >= 70 ? 'hover:bg-green-50' : accuracy >= 50 ? 'hover:bg-yellow-50' : 'hover:bg-red-50'

          return (
            <Link
              key={item.item_id}
              href={`/locomo/${idx}`}
              className={`block bg-white border border-border rounded-lg shadow-sm transition-all ${borderClass} ${hoverClass}`}
            >
              <div className="p-6">
                <p className="text-lg font-semibold text-foreground mb-2">
                  {color} {item.item_id}
                </p>
                <div className="flex gap-6 items-center">
                  <div className="text-center">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Accuracy</p>
                    <p className="text-2xl font-bold text-foreground">{accuracy.toFixed(1)}%</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Correct</p>
                    <p className="text-xl font-semibold text-foreground">
                      {item.metrics.correct}/{item.metrics.total}
                    </p>
                  </div>
                </div>
              </div>
            </Link>
          )
        })}
      </div>
    </main>
  )
}
