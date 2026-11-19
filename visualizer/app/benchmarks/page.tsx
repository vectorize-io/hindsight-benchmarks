import type { Metadata } from 'next'
import { ArrowUpRight } from 'lucide-react'
import { AMB_COMMIT, AMB_REPO, AMB_SITE, SECTIONS, rank } from '@/lib/benchmarks'
import { PageHero } from '@/components/Page'
import { BenchmarkCard } from '@/components/BenchmarkCard'
import { ModelScatter } from '@/components/BoardView'
import { CODING_SCATTER_TEXT, codingPoints } from '@/lib/coding'
import type { CodingRow } from '@/lib/data'
import CODING from '@/lib/coding-agents.json'

export const metadata: Metadata = {
  title: 'Benchmarks',
  description:
    'Hindsight next to other systems on the Agent Memory Benchmark: memory (LongMemEval, LoCoMo, PersonaMem, LifeBench, BEAM, PrecisionMemBench), knowledge-bank search (ViDoRe V3, ATM-Bench) and both together (SupportBench).',
}

export default function BenchmarksPage() {
  // Footnotes are numbered down the page, one per card.
  let n = 0
  const sections = SECTIONS.map((s) => ({ ...s, cards: s.cards.map((c) => ({ c, n: ++n })) }))
  const summary = sections.map((s) => {
    const ranked = s.cards.flatMap(({ c }) => c.panels.map(rank)).filter((r) => r !== null)
    return { key: s.key, title: s.title, first: ranked.filter((r) => r.place === 1).length, ranked: ranked.length, cards: s.cards.length }
  })

  return (
    <>
      <PageHero
        title="Benchmarks"
        subtitle="Hindsight next to other systems on the public datasets of the Agent Memory Benchmark: memory, knowledge-bank search, and the two together."
      />

      <main className="max-w-6xl mx-auto px-4 pt-8 space-y-14">
        <nav className="grid sm:grid-cols-3 gap-3" aria-label="Sections">
          {summary.map((s) => (
            <a key={s.key} href={`#${s.key}`} className="rounded-xl border border-border p-4 hover:border-[hsl(var(--primary))]/50 transition">
              <div className="font-heading font-bold">{s.title}</div>
              <div className="text-sm text-muted-foreground">
                {s.ranked ? `1st on ${s.first} of ${s.ranked} ${s.ranked === 1 ? 'test' : 'tests'}` : `${s.cards} ${s.cards === 1 ? 'benchmark' : 'benchmarks'}, no other system yet`}
              </div>
            </a>
          ))}
        </nav>

        <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-muted-foreground -mt-8">
          <span className="inline-flex items-center gap-1.5">
            <i className="inline-block w-4 h-2.5 rounded-sm" style={{ background: 'linear-gradient(90deg, #0074d9, #009296)' }} /> Hindsight
          </span>
          <span className="inline-flex items-center gap-1.5">
            <i className="inline-block w-4 h-2.5 rounded-sm" style={{ background: 'hsl(var(--muted-foreground) / 0.38)' }} /> Run on AMB, same setup
          </span>
          <span className="inline-flex items-center gap-1.5">
            <i
              className="inline-block w-4 h-2.5 rounded-sm"
              style={{ background: 'repeating-linear-gradient(135deg, hsl(var(--muted-foreground) / 0.32) 0 4px, hsl(var(--muted-foreground) / 0.16) 4px 6px)' }}
            />{' '}
            Published by the system itself
          </span>
          <span>Click a bar for the run or the source.</span>
        </div>

        {sections.map((s) => (
          <section key={s.key} id={s.key} className="scroll-mt-24">
            <h2 className="font-heading font-extrabold text-2xl">{s.title}</h2>
            <p className="text-muted-foreground mt-1 mb-5 max-w-3xl">{s.about}</p>
            <div className="grid md:grid-cols-2 gap-4">
              {s.cards.map(({ c, n }) => (
                <BenchmarkCard key={c.key} c={c} n={n}>
                  {c.key === 'sdebench' && (
                    <ModelScatter rows={codingPoints(CODING.rows as CodingRow[])} task="SDE-Bench" text={CODING_SCATTER_TEXT} maxH={260} />
                  )}
                </BenchmarkCard>
              ))}
            </div>
          </section>
        ))}

        <section className="border-t border-border pt-6">
          <h2 className="font-heading font-bold text-sm mb-2">Notes</h2>
          <ol className="space-y-1.5 text-xs text-muted-foreground list-decimal pl-5 max-w-4xl">
            {sections.flatMap((s) =>
              s.cards.map(({ c, n }) => (
                <li key={c.key} id={`note-${n}`} className="scroll-mt-24">
                  <span className="text-foreground">{c.bench}.</span> {c.note}{' '}
                  <a href={`${AMB_SITE}/dataset/${c.key}`} className="text-[hsl(var(--primary))] hover:underline">
                    Dataset and runs
                  </a>
                </li>
              )),
            )}
          </ol>
          <p className="text-xs text-muted-foreground mt-4">
            Data: <a href={`${AMB_REPO}/tree/${AMB_COMMIT}`} className="font-mono hover:underline">agent-memory-benchmark@{AMB_COMMIT.slice(0, 7)}</a>.
          </p>
        </section>

        <a
          href={AMB_SITE}
          className="flex items-center justify-between gap-4 rounded-xl border border-border p-6 hover:border-[hsl(var(--primary))]/50 transition"
        >
          <div>
            <div className="font-heading font-bold text-lg">Every system, every dataset</div>
            <div className="text-sm text-muted-foreground">With the raw runs and every answer, on agentmemorybenchmark.ai.</div>
          </div>
          <ArrowUpRight className="w-5 h-5 text-muted-foreground shrink-0" />
        </a>
      </main>
    </>
  )
}

