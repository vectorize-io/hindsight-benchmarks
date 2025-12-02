import Link from 'next/link'
import { loadLoComo, getCategoryName } from '@/lib/data'
import { notFound } from 'next/navigation'
import { QuestionList } from '@/components/question-list'

export function generateStaticParams() {
  const data = loadLoComo()
  if (!data) return []

  return data.item_results.map((_, idx) => ({
    itemIdx: idx.toString(),
  }))
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

  // Calculate category stats for this item
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
      if (result.is_invalid) {
        categoryStats[category].invalid++
      } else if (result.is_correct) {
        categoryStats[category].correct++
      }
    }
  })

  return (
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <Link
        href="/locomo"
        className="inline-flex items-center px-4 py-2 bg-white border border-border rounded-md text-sm font-medium text-foreground hover:bg-accent mb-6 transition-colors"
      >
        ← Back to LoComo
      </Link>

      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-6">
          {item.item_id} - Performance
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <div className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
              Overall Accuracy
            </p>
            <p className="text-3xl font-bold text-foreground">{accuracy.toFixed(2)}%</p>
          </div>

          <div className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
              Correct Answers
            </p>
            <p className="text-3xl font-bold text-foreground">
              {metrics.correct} / {metrics.total}
            </p>
          </div>

          {metrics.invalid > 0 && (
            <div className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
              <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                Invalid Questions
              </p>
              <p className="text-3xl font-bold text-foreground">{metrics.invalid}</p>
            </div>
          )}

          <div className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
              Total Questions
            </p>
            <p className="text-3xl font-bold text-foreground">{metrics.total}</p>
          </div>
        </div>

        {Object.values(categoryStats).some((cat) => cat.total > 0) && (
          <>
            <h2 className="text-xl font-semibold text-foreground mb-4">Accuracy by Category</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.values(categoryStats)
                .filter((cat) => cat.total > 0)
                .map((cat) => {
                  const validTotal = cat.total - cat.invalid
                  const catAccuracy = validTotal > 0 ? (cat.correct / validTotal) * 100 : 0
                  return (
                    <div key={cat.name} className="bg-white border border-border rounded-lg p-6 text-center shadow-sm">
                      <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
                        {cat.name}
                      </p>
                      <p className="text-2xl font-bold text-foreground">{catAccuracy.toFixed(1)}%</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        {cat.correct} / {cat.total}
                      </p>
                    </div>
                  )
                })}
            </div>
          </>
        )}
      </div>

      <hr className="my-6 border-border" />

      <QuestionList questions={detailedResults} />
    </main>
  )
}
