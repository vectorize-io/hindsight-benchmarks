import Link from 'next/link'
import { loadLongMemEval } from '@/lib/data'
import { notFound } from 'next/navigation'

export function generateStaticParams() {
  const data = loadLongMemEval()
  if (!data) return []

  return data.item_results.map((_, idx) => ({
    itemIdx: idx.toString(),
  }))
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

  return (
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <Link
        href="/longmemeval"
        className="inline-flex items-center px-4 py-2 bg-white border border-border rounded-md text-sm font-medium text-foreground hover:bg-accent mb-6 transition-colors"
      >
        ← Back to LongMemEval
      </Link>

      <div className="mb-8">
        <h1 className="text-3xl font-bold text-foreground mb-6">
          {item.item_id} - Performance
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
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
      </div>

      <hr className="my-6 border-border" />

      <p className="text-sm text-muted-foreground mb-6">Showing {detailedResults.length} questions</p>

      <div className="space-y-4">
        {detailedResults.map((result, qIdx) => {
          const isInvalid = result.is_invalid
          const isCorrect = result.is_correct
          const icon = isInvalid ? '⚠️' : isCorrect ? '✅' : '❌'
          const borderClass = isInvalid
            ? 'border-l-4 border-yellow-500'
            : isCorrect
            ? 'border-l-4 border-green-600'
            : 'border-l-4 border-red-600'

          return (
            <div key={qIdx} className={`bg-white border border-border rounded-lg p-6 shadow-sm ${borderClass}`}>
              <div className="mb-4">
                <p className="text-lg font-semibold text-foreground">
                  {icon} Question {qIdx + 1}
                </p>
                <p className="text-sm text-muted-foreground">Category: {result.category}</p>
              </div>

              <div className="mb-4">
                <p className="text-sm font-medium text-foreground mb-1">Question:</p>
                <p className="text-foreground">{result.question}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <p className="text-sm font-medium text-foreground mb-2">✓ Correct Answer</p>
                  <div className="bg-green-50 border border-green-200 rounded-md p-3 text-foreground text-sm">
                    {result.correct_answer}
                  </div>
                </div>

                <div>
                  <p className="text-sm font-medium text-foreground mb-2">
                    {isCorrect ? '✓' : '✗'} Predicted Answer
                  </p>
                  <div
                    className={`border rounded-md p-3 text-foreground text-sm ${
                      isCorrect ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
                    }`}
                  >
                    {result.predicted_answer}
                  </div>
                </div>
              </div>

              <details className="border border-border rounded-md p-4 bg-muted/30">
                <summary className="cursor-pointer font-medium text-foreground hover:text-primary py-2">
                  📝 Show Reasoning & Retrieved Memories
                </summary>
                <div className="mt-3 space-y-4">
                  <div>
                    <p className="text-sm font-medium text-foreground mb-2">System Reasoning:</p>
                    <pre className="bg-slate-900 text-slate-100 p-3 rounded-md overflow-x-auto text-sm whitespace-pre-wrap">
                      {result.reasoning || 'N/A'}
                    </pre>
                  </div>

                  {result.correctness_reasoning && (
                    <div>
                      <p className="text-sm font-medium text-foreground mb-2">Judge Reasoning:</p>
                      <pre className="bg-slate-900 text-slate-100 p-3 rounded-md overflow-x-auto text-sm whitespace-pre-wrap">
                        {result.correctness_reasoning}
                      </pre>
                    </div>
                  )}

                  <div>
                    <p className="text-sm font-medium text-foreground mb-2">
                      Retrieved Memories ({result.retrieved_memories?.length || 0}):
                    </p>
                    <p className="text-xs text-muted-foreground mb-3">
                      Hindsight also uses chunks and entities for retrieval.{' '}
                      <a href="http://hindsight.vectorize.io/" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                        Learn more
                      </a>
                    </p>
                    {result.retrieved_memories && result.retrieved_memories.length > 0 ? (
                      result.retrieved_memories.map((mem, i) => (
                        <div key={i} className="bg-muted/50 border border-border rounded-md p-3 mb-2">
                          <p className="text-xs text-muted-foreground mb-1">
                            #{i + 1} • Type: {mem.fact_type?.toUpperCase() || 'N/A'}
                            {mem.occurred_start && ` • Date: ${mem.occurred_start.slice(0, 10)}`}
                          </p>
                          <p className="text-sm text-foreground">{mem.text}</p>
                        </div>
                      ))
                    ) : (
                      <p className="text-sm text-muted-foreground">No memories retrieved</p>
                    )}
                  </div>
                </div>
              </details>
            </div>
          )
        })}
      </div>
    </main>
  )
}
