import type { Metadata } from 'next'
import { ArrowUpRight } from 'lucide-react'
import { AMB_RESULTS, AMB_SITE } from '@/lib/amb'
import { PageHero } from '@/components/Page'

export const metadata: Metadata = {
  title: 'Benchmarks',
  description: 'Hindsight as a whole system on the Agent Memory Benchmark: LongMemEval, LoComo, PersonaMem, BEAM and LifeBench.',
}

const BEAM_ORDER = ['100K', '500K', '1M', '10M']

export default function BenchmarksPage() {
  const groups = [
    {
      label: 'Datasets',
      rows: AMB_RESULTS.filter((r) => r.benchmark !== 'BEAM').sort((a, b) => b.score - a.score),
    },
    {
      label: 'BEAM, as memory grows',
      rows: AMB_RESULTS.filter((r) => r.benchmark === 'BEAM').sort((a, b) => BEAM_ORDER.indexOf(a.split) - BEAM_ORDER.indexOf(b.split)),
    },
  ]

  return (
    <>
      <PageHero
        title="Benchmarks"
        subtitle="Hindsight as a whole system, end to end, on the public datasets of the Agent Memory Benchmark."
      >
        <span
          className="inline-flex items-center mt-6 px-3 py-1 rounded-full text-xs font-semibold text-white"
          style={{ background: 'linear-gradient(90deg, #0074d9 0%, #009296 100%)' }}
        >
          #1 on every dataset
        </span>
      </PageHero>

      <main className="max-w-6xl mx-auto px-4 pt-10 space-y-8">
        <section className="rounded-xl border border-border p-6">
          <div className="flex items-baseline justify-between flex-wrap gap-2 mb-6">
            <h2 className="font-heading font-bold text-xl">Accuracy by dataset</h2>
            <span className="text-xs text-muted-foreground">Click a bar for the full run</span>
          </div>
          {/* Shared 0-100 scale, so bars compare across groups. */}
          <div className="relative">
            <div className="absolute inset-y-0 left-[9.5rem] right-[4.5rem] pointer-events-none hidden sm:block" aria-hidden>
              {[25, 50, 75].map((t) => (
                <div key={t} className="absolute inset-y-0 border-l border-dashed border-border" style={{ left: `${t}%` }} />
              ))}
            </div>
            <div className="space-y-6">
              {groups.map((g) => (
                <div key={g.label}>
                  <div className="text-[11px] uppercase tracking-wide text-muted-foreground mb-2 sm:pl-[9.5rem]">{g.label}</div>
                  <div className="space-y-2">
                    {g.rows.map((r) => (
                      <a
                        key={`${r.benchmark}-${r.split}`}
                        href={r.href}
                        title={`${r.benchmark} ${r.split}: open the run on agentmemorybenchmark.ai`}
                        className="group relative flex items-center gap-3 rounded-md -mx-2 px-2 py-0.5 hover:bg-secondary/60 transition-colors"
                      >
                        <span className="w-36 shrink-0 text-sm sm:text-right truncate">
                          {g.label === 'Datasets' ? r.benchmark : 'BEAM'}{' '}
                          <span className="font-mono text-xs text-muted-foreground">{r.split}</span>
                        </span>
                        <div className="flex-1 h-7 rounded-md bg-secondary overflow-hidden">
                          <div className="h-full rounded-md" style={{ width: `${r.score}%`, background: 'linear-gradient(90deg, #0074d9, #009296)' }} />
                        </div>
                        <span className="w-12 shrink-0 font-heading font-bold tabular-nums text-right">{r.score}%</span>
                        <ArrowUpRight className="w-3.5 h-3.5 shrink-0 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                      </a>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <div className="hidden sm:flex relative mt-3 ml-[9.5rem] mr-[4.5rem] text-[11px] text-muted-foreground font-mono h-4" aria-hidden>
              {[0, 25, 50, 75, 100].map((t) => (
                <span key={t} className="absolute -translate-x-1/2" style={{ left: `${t}%` }}>
                  {t}
                </span>
              ))}
            </div>
          </div>
        </section>

        <a
          href={AMB_SITE}
          className="flex items-center justify-between gap-4 rounded-xl border border-border p-6 hover:border-[hsl(var(--primary))]/50 transition"
        >
          <div>
            <div className="font-heading font-bold text-lg">Compare with other memory systems</div>
            <div className="text-sm text-muted-foreground">Every system, every dataset, with the raw runs, on agentmemorybenchmark.ai.</div>
          </div>
          <ArrowUpRight className="w-5 h-5 text-muted-foreground shrink-0" />
        </a>
      </main>
    </>
  )
}
