import type { Metadata } from 'next'
import Link from 'next/link'
import { ArrowRight, ArrowUpRight } from 'lucide-react'
import { AMB_RESULTS } from '@/lib/amb'
import { BOARDS } from '@/lib/v2/data'
import { MockNotice, PageHero } from '@/components/v2/Page'

export const metadata: Metadata = {
  title: { absolute: 'Hindsight Benchmarks' },
  description: 'How Hindsight scores as a whole on the Agent Memory Benchmark, and which models to run inside it.',
}

export default function Home() {
  return (
    <>
      <PageHero
        title="Hindsight Benchmarks"
        subtitle="How Hindsight scores as a whole on public memory benchmarks, and which models to run inside it."
      />
      <main className="max-w-6xl mx-auto px-4 pt-12 space-y-16">
        {/* Benchmarks first: the whole system. */}
        <section className="grid lg:grid-cols-[1fr_1.4fr] gap-8 lg:gap-12 items-start">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-3">Benchmarks</div>
            <h2 className="font-heading font-extrabold text-3xl tracking-[-0.02em] leading-tight mb-3">
              #1 on every <span className="gradient-primary-text">Agent Memory Benchmark</span> dataset
            </h2>
            <p className="text-muted-foreground leading-relaxed mb-6">
              Hindsight end to end, with its recommended setup, on LongMemEval, LoComo, PersonaMem, BEAM and LifeBench. The same datasets
              other memory systems run, with every run public.
            </p>
            <Link
              href="/v2/benchmarks"
              className="inline-flex items-center gap-1.5 h-9 px-4 rounded-lg font-heading font-bold text-sm text-white hover:brightness-110 transition"
              style={{ background: 'linear-gradient(90deg, #0074d9 0%, #009296 100%)' }}
            >
              See all results <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {AMB_RESULTS.map((r) => (
              <a
                key={`${r.benchmark}-${r.split}`}
                href={r.href}
                title={`${r.benchmark} ${r.split}: open the run on agentmemorybenchmark.ai`}
                className="group rounded-xl border border-border p-3.5 hover:border-[hsl(var(--primary))]/50 transition"
              >
                <div className="flex items-baseline justify-between text-xs mb-2">
                  <span className="font-medium truncate">{r.benchmark}</span>
                  <span className="font-mono text-muted-foreground">{r.split}</span>
                </div>
                <div className="font-heading font-extrabold text-2xl tabular-nums tracking-tight">
                  {r.score}
                  <span className="text-sm text-muted-foreground font-bold">%</span>
                </div>
                <div className="mt-2 h-1 rounded-full bg-secondary overflow-hidden">
                  <div className="h-full rounded-full" style={{ width: `${r.score}%`, background: 'linear-gradient(90deg, #0074d9, #009296)' }} />
                </div>
                <ArrowUpRight className="w-3 h-3 mt-2 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
              </a>
            ))}
          </div>
        </section>

        <div className="border-t border-border" />

        {/* Then the parts: which model behind each operation. */}
        <section>
          <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
            <div className="max-w-2xl">
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-3">Model leaderboards</div>
              <h2 className="font-heading font-extrabold text-3xl tracking-[-0.02em] leading-tight mb-3">Which model to run, operation by operation</h2>
              <p className="text-muted-foreground leading-relaxed">
                One leaderboard per Hindsight operation. Each swaps one model and keeps the rest fixed, with quality, cost and calls side by
                side.
              </p>
              <MockNotice />
            </div>
            <Link href="/v2/leaderboards" className="text-sm font-medium text-[hsl(var(--primary))] inline-flex items-center gap-1">
              All leaderboards <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {BOARDS.map((b) => {
              // A baseline row ("no decision model") is a yardstick, not a candidate.
              const leader = b.tabs[0].rows.filter((r) => r.tag !== 'baseline').sort((x, y) => y.quality - x.quality)[0]
              return (
                <Link
                  key={b.id}
                  href={`/v2/leaderboards/${b.id}`}
                  className="group rounded-xl border border-border p-5 hover:border-[hsl(var(--primary))]/50 transition"
                >
                  <div className="flex items-baseline justify-between mb-1">
                    <h3 className="font-heading font-bold">{b.title}</h3>
                    <ArrowRight className="w-3.5 h-3.5 text-muted-foreground group-hover:translate-x-0.5 transition-transform" />
                  </div>
                  <p className="text-sm text-muted-foreground mb-4">{b.short}</p>
                  {leader && (
                    <div className="flex items-center justify-between text-sm rounded-lg bg-secondary px-3 py-2">
                      <span className="text-muted-foreground">Leader</span>
                      <span className="font-mono text-[13px] truncate ml-3">{leader.model}</span>
                      <span className="font-mono font-semibold tabular-nums ml-3">{leader.quality.toFixed(1)}</span>
                    </div>
                  )}
                </Link>
              )
            })}
          </div>
        </section>
      </main>
    </>
  )
}
