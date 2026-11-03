import Link from 'next/link'
import { ArrowRight } from 'lucide-react'
import { BOARDS, RUN, USE_CASES, type Board } from '@/lib/data'
import { ModelCell, fmtCost } from '@/components/ModelCell'
import { BoardNav, PageHero, SetupStrip } from '@/components/Page'

export default function LeaderboardsPage() {
  return (
    <>
      <PageHero
        title="Model leaderboards"
        subtitle="Which model to run with Hindsight: one LLM for everything, the best per operation, and what each setup changes in a real task. Every case is a real Hindsight server, graded by an independent judge."
      ></PageHero>
      <BoardNav />
      <main className="max-w-6xl mx-auto px-4 pt-8 space-y-14">
        <Group title="Hindsight operations" about="The best model for each thing Hindsight does, and one LLM behind all of them.">
          {BOARDS.filter((b) => b.group === 'summary').map((b) => (
            <div key={b.id} className="md:col-span-2 grid">
              <BoardCard b={b} wide />
            </div>
          ))}
          {BOARDS.filter((b) => b.group === 'operation').map((b) => (
            <BoardCard key={b.id} b={b} />
          ))}
        </Group>
        <Group title="Use cases" about="Hindsight in a real task, end to end: what each setup changes in practice.">
          {USE_CASES.map((u) => (
            <Link
              key={u.id}
              href={`/leaderboards/${u.id}`}
              className="group rounded-xl border border-border p-5 flex flex-col hover:border-[hsl(var(--primary))]/50 transition"
            >
              <div className="flex items-baseline justify-between mb-1">
                <h3 className="font-heading font-bold text-lg">{u.title}</h3>
              </div>
              <p className="text-sm text-muted-foreground mb-3">{u.short}</p>
              <p className="text-xs text-muted-foreground flex-1">{u.dataset}</p>
              <span className="mt-5 text-sm font-medium text-[hsl(var(--primary))] inline-flex items-center gap-1">
                Open board <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
              </span>
            </Link>
          ))}
        </Group>

        <section className="rounded-xl border border-border p-6 space-y-5">
          <h2 className="font-heading font-bold text-xl">How we test</h2>
          <ul className="space-y-2 text-sm text-muted-foreground list-disc pl-5">
            <li>
              Hindsight&apos;s own system evals, one suite per operation, on Hindsight <span className="font-mono text-foreground">{RUN.hindsightVersion}</span>,
              against a fresh server through the public API.
            </li>
            <li>Each row swaps one part and keeps the reference setup below. A row marked “with” also runs a decision model.</li>
            <li>
              Graded by <span className="text-foreground">{RUN.judge}</span>, never the model under test. Each model runs once. Cost comes from the tokens
              really used, at OpenRouter list prices.
            </li>
          </ul>
          <SetupStrip bare />
        </section>
      </main>
    </>
  )
}

function Group({ title, about, children }: { title: string; about: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="font-heading font-bold text-xl">{title}</h2>
      <p className="text-sm text-muted-foreground mb-4">{about}</p>
      <div className="grid md:grid-cols-2 gap-5">{children}</div>
    </section>
  )
}

// wide: the all-operations board, a full-width card with its top six in two columns.
function BoardCard({ b, wide }: { b: Board; wide?: boolean }) {
  const tab = b.tabs[0]
  const top = [...tab.rows].sort((x, y) => (y.score ?? 0) - (x.score ?? 0)).slice(0, wide ? 6 : 3)
  const cases = b.categories.reduce((s, c) => s + c.cases, 0)
  return (
    <Link
      key={b.id}
      href={`/leaderboards/${b.id}`}
      className="group rounded-xl border border-border p-5 flex flex-col hover:border-[hsl(var(--primary))]/50 hover:shadow-sm transition"
    >
      <div className="flex items-baseline justify-between mb-1">
        <h2 className="font-heading font-bold text-lg">{b.title}</h2>
        {cases > 0 && <span className="text-xs text-muted-foreground font-mono">{cases} cases</span>}
      </div>
      <p className="text-sm text-muted-foreground mb-4">{b.short}</p>
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">
        Top {tab.label.toLowerCase()} by score
        {b.tabs.length > 1 && (
          <span className="normal-case tracking-normal">
            {' '}
            · +{b.tabs.length - 1} more tab{b.tabs.length > 2 ? 's' : ''}
          </span>
        )}
      </div>
      <ol className={wide ? 'grid md:grid-cols-2 md:grid-flow-col md:grid-rows-3 gap-x-10 gap-y-2.5 flex-1' : 'space-y-2.5 flex-1'}>
        {top.map((r, i) => (
          <li key={r.id} className="flex items-center gap-3">
            <span className="font-mono text-xs text-muted-foreground w-3">{i + 1}</span>
            <div className="flex-1 min-w-0">
              <ModelCell row={r} />
            </div>
            <div className="text-right">
              <div className="font-mono text-sm font-semibold tabular-nums" title="Score">
                {(r.score ?? r.quality).toFixed(1)}
              </div>
              {/* The score's two parts under it. */}
              <div className="font-mono text-[11px] text-muted-foreground whitespace-nowrap">
                {r.quality.toFixed(0)}% · {fmtCost(r.cost)}
              </div>
            </div>
          </li>
        ))}
      </ol>
      <span className="mt-5 text-sm font-medium text-[hsl(var(--primary))] inline-flex items-center gap-1">
        Open board <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
      </span>
    </Link>
  )
}
