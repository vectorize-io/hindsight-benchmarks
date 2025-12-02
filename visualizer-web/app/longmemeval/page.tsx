import Link from 'next/link'
import { loadLongMemEval } from '@/lib/data'
import { notFound } from 'next/navigation'
import { ItemList } from '@/components/item-list'

export default function LongMemEvalPage() {
  const data = loadLongMemEval()

  if (!data) {
    notFound()
  }

  // Calculate category stats
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

  return (
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <Link
        href="/"
        className="inline-flex items-center px-4 py-2 bg-white border border-border rounded-md text-sm font-medium text-foreground hover:bg-accent mb-6 transition-colors"
      >
        ← Back to benchmarks
      </Link>

      <h1 className="text-2xl font-bold text-foreground mb-6">LongMemEval Benchmark - Overall Performance</h1>

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

        {data.total_invalid > 0 && (
          <div className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Invalid Questions</p>
            <p className="text-3xl font-bold text-foreground">{data.total_invalid}</p>
          </div>
        )}

        <div className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Items</p>
          <p className="text-3xl font-bold text-foreground">{data.num_items}</p>
        </div>
      </div>

      {Object.keys(categoryStats).length > 0 && (
        <>
          <h2 className="text-xl font-semibold text-foreground mb-4">Accuracy by Category</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            {Object.entries(categoryStats)
              .sort(([a], [b]) => a.localeCompare(b))
              .map(([category, stats]) => {
                const validTotal = stats.total - stats.invalid
                const accuracy = validTotal > 0 ? (stats.correct / validTotal) * 100 : 0
                return (
                  <div key={category} className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                      {category}
                    </p>
                    <p className="text-2xl font-bold text-foreground">{accuracy.toFixed(1)}%</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      {stats.correct} / {stats.total}
                    </p>
                  </div>
                )
              })}
          </div>
        </>
      )}

      <ItemList items={data.item_results} basePath="/longmemeval" showCategories={true} />
    </main>
  )
}
