import Link from 'next/link'
import { loadLongMemEval, loadLoComo } from '@/lib/data'

export default function Home() {
  // Check which benchmarks are available
  const longmemeval = loadLongMemEval()
  const locomo = loadLoComo()

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-50">
      <div className="container mx-auto max-w-7xl px-4 py-16">
        <div className="text-center mb-16">
          <div className="inline-flex items-center justify-center w-20 h-20 mb-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl shadow-lg">
            <span className="text-4xl">📊</span>
          </div>
          <h1 className="text-5xl font-bold text-foreground mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
            Hindsight Benchmarks
          </h1>
          <p className="text-lg text-muted-foreground mb-4">
            Analyze and visualize LongMemEval and LoComo benchmark results for Hindsight
          </p>
          <a
            href="https://github.com/vectorize-io/hindsight"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded-lg hover:bg-slate-800 transition-colors text-sm font-medium"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
            </svg>
            View on GitHub
          </a>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-2 gap-6">
            {longmemeval && (
              <Link
                href="/longmemeval"
                className="group relative bg-white border-2 border-border rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1 overflow-hidden"
              >
                <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-green-100 to-emerald-100 rounded-bl-full opacity-50"></div>
                <div className="relative">
                  <div className="text-4xl mb-4">🎯</div>
                  <h2 className="text-2xl font-bold text-foreground mb-2 group-hover:text-blue-600 transition-colors">
                    LongMemEval
                  </h2>
                  <p className="text-muted-foreground text-sm mb-4">
                    Comprehensive long-term memory evaluation
                  </p>
                  <div className="flex items-center gap-4 text-sm">
                    <div>
                      <div className="text-2xl font-bold text-foreground">
                        {longmemeval.overall_accuracy.toFixed(1)}%
                      </div>
                      <div className="text-xs text-muted-foreground">Accuracy</div>
                    </div>
                    <div className="h-10 w-px bg-border"></div>
                    <div>
                      <div className="text-2xl font-bold text-foreground">{longmemeval.num_items}</div>
                      <div className="text-xs text-muted-foreground">Items</div>
                    </div>
                    <div className="h-10 w-px bg-border"></div>
                    <div>
                      <div className="text-2xl font-bold text-foreground">{longmemeval.total_questions}</div>
                      <div className="text-xs text-muted-foreground">Questions</div>
                    </div>
                  </div>
                  <div className="mt-6 flex items-center text-blue-600 font-medium">
                    View Results
                    <svg className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </div>
              </Link>
            )}

            {locomo && (
              <Link
                href="/locomo"
                className="group relative bg-white border-2 border-border rounded-2xl p-8 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1 overflow-hidden"
              >
                <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-purple-100 to-pink-100 rounded-bl-full opacity-50"></div>
                <div className="relative">
                  <div className="text-4xl mb-4">💬</div>
                  <h2 className="text-2xl font-bold text-foreground mb-2 group-hover:text-purple-600 transition-colors">
                    LoComo
                  </h2>
                  <p className="text-muted-foreground text-sm mb-4">
                    Long conversation memory benchmark
                  </p>
                  <div className="flex items-center gap-4 text-sm">
                    <div>
                      <div className="text-2xl font-bold text-foreground">
                        {locomo.overall_accuracy.toFixed(1)}%
                      </div>
                      <div className="text-xs text-muted-foreground">Accuracy</div>
                    </div>
                    <div className="h-10 w-px bg-border"></div>
                    <div>
                      <div className="text-2xl font-bold text-foreground">{locomo.num_items}</div>
                      <div className="text-xs text-muted-foreground">Convos</div>
                    </div>
                    <div className="h-10 w-px bg-border"></div>
                    <div>
                      <div className="text-2xl font-bold text-foreground">{locomo.total_questions}</div>
                      <div className="text-xs text-muted-foreground">Questions</div>
                    </div>
                  </div>
                  <div className="mt-6 flex items-center text-purple-600 font-medium">
                    View Results
                    <svg className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </div>
              </Link>
            )}

            {!longmemeval && !locomo && (
              <div className="col-span-2 text-center py-12 bg-white border-2 border-border rounded-2xl shadow-lg">
                <div className="text-6xl mb-4">📂</div>
                <h3 className="text-xl font-semibold text-foreground mb-2">No Results Found</h3>
                <p className="text-muted-foreground">
                  Add result files to the <code className="px-2 py-1 bg-slate-100 rounded">results/</code> directory to get
                  started.
                </p>
              </div>
            )}
          </div>
        </div>

      </div>
    </main>
  )
}
