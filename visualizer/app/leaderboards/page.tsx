import Link from 'next/link'
import { ArrowRight } from 'lucide-react'
import { BOARDS, RUN } from '@/lib/data'
import { ModelCell, fmtCost } from '@/components/ModelCell'
import { BoardNav, MockNotice, PageHero, SetupStrip } from '@/components/Page'

export default function LeaderboardsPage() {
  return (
    <>
      <PageHero
        title="Model leaderboards"
        subtitle="Which model to run behind each Hindsight operation. Every case is a real Hindsight server, graded by an independent judge."
      >
        <MockNotice />
      </PageHero>
      <BoardNav />
      <main className="max-w-6xl mx-auto px-4 pt-8 space-y-14">
        <section className="grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          {BOARDS.map((b) => {
            const tab = b.tabs[0]
            const top = [...tab.rows].sort((x, y) => y.quality - x.quality).slice(0, 3)
            const cases = b.categories.reduce((s, c) => s + c.cases, 0)
            return (
              <Link
                key={b.id}
                href={`/leaderboards/${b.id}`}
                className="group rounded-xl border border-border p-5 flex flex-col hover:border-[hsl(var(--primary))]/50 hover:shadow-sm transition"
              >
                <div className="flex items-baseline justify-between mb-1">
                  <h2 className="font-heading font-bold text-lg">{b.title}</h2>
                  <span className="text-xs text-muted-foreground font-mono">{cases} cases</span>
                </div>
                <p className="text-sm text-muted-foreground mb-4">{b.short}</p>
                <div className="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">
                  Top {tab.label.toLowerCase()}
                  {b.tabs.length > 1 && <span className="normal-case tracking-normal"> · +{b.tabs.length - 1} more tab{b.tabs.length > 2 ? 's' : ''}</span>}
                </div>
                <ol className="space-y-2.5 flex-1">
                  {top.map((r, i) => (
                    <li key={r.id} className="flex items-center gap-3">
                      <span className="font-mono text-xs text-muted-foreground w-3">{i + 1}</span>
                      <div className="flex-1 min-w-0">
                        <ModelCell row={r} />
                      </div>
                      <div className="text-right">
                        <div className="font-mono text-sm font-semibold tabular-nums">{r.quality.toFixed(1)}</div>
                        <div className="font-mono text-[11px] text-muted-foreground">{fmtCost(r.cost)}</div>
                      </div>
                    </li>
                  ))}
                </ol>
                <span className="mt-5 text-sm font-medium text-[hsl(var(--primary))] inline-flex items-center gap-1">
                  Open board <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
                </span>
              </Link>
            )
          })}
        </section>

        <section className="rounded-xl border border-border p-6 space-y-5">
          <h2 className="font-heading font-bold text-xl">How we test</h2>
          <ul className="space-y-2 text-sm text-muted-foreground list-disc pl-5">
            <li>
              Hindsight&apos;s own system evals, one suite per operation, at commit{' '}
              <span className="font-mono text-foreground">{RUN.hindsightCommit}</span>, against a fresh server through the public API.
            </li>
            <li>
              Each row swaps one part and keeps the reference setup below. Modes, like a decision model, are tabs with their own ranking.
            </li>
            <li>
              Graded by <span className="text-foreground">{RUN.judge}</span>, never the model under test. Each model runs once.
              Cost comes from the tokens really used; speed is what the provider states.
            </li>
          </ul>
          <SetupStrip bare />
        </section>
      </main>
    </>
  )
}
