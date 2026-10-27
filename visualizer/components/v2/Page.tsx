import Link from 'next/link'
import { FlaskConical } from 'lucide-react'
import { BOARDS, MOCK, PART_LABEL, REFERENCE, RUN, type Board, type Part } from '@/lib/v2/data'
import { cn } from '@/lib/utils'

// The docs' standalone-page band (PageHero + hs-hero-band): flat surface, a
// rule under it, a centred 2.75rem/800 title and a soft subtitle.
export function PageHero({ title, subtitle, children }: { title: string; subtitle: string; children?: React.ReactNode }) {
  return (
    <div className="bg-[hsl(var(--band))] border-b border-border px-6 pt-[3.25rem] pb-12 text-center">
      <h1 className="font-heading text-[2rem] sm:text-[2.75rem] font-extrabold tracking-[-0.03em] leading-[1.15] mb-3">{title}</h1>
      <p className="text-base text-muted-foreground max-w-[620px] mx-auto leading-relaxed">{subtitle}</p>
      {children}
    </div>
  )
}

export function MockNotice() {
  if (!MOCK) return null
  return (
    <div className="inline-flex items-center gap-2 mt-6 px-3 py-1.5 rounded-full border border-amber-500/40 bg-amber-500/10 text-amber-600 text-xs font-medium">
      <FlaskConical className="w-3.5 h-3.5" />
      Preview with mock numbers. Nothing on this page is measured yet.
    </div>
  )
}

export function BoardNav({ current }: { current?: string }) {
  const items = [{ id: '', title: 'Overview' }, ...BOARDS.map((b) => ({ id: b.id, title: b.title }))]
  return (
    <div className="sticky top-16 z-40 bg-background/95 backdrop-blur border-b border-border">
      <div className="max-w-6xl mx-auto px-4 flex gap-1 overflow-x-auto py-2">
        {items.map((i) => {
          const active = (current ?? '') === i.id
          return (
            <Link
              key={i.id || 'overview'}
              href={i.id ? `/v2/leaderboards/${i.id}` : '/v2/leaderboards'}
              className={cn(
                'shrink-0 inline-flex items-center h-8 px-3 rounded-[0.625rem] font-heading font-bold text-sm transition-colors',
                active ? 'nav-active' : 'text-muted-foreground hover:text-foreground hover:bg-secondary',
              )}
            >
              {i.title}
            </Link>
          )
        })}
      </div>
    </div>
  )
}

export function SetupStrip({ board, bare }: { board?: Board; bare?: boolean }) {
  const parts: Part[] = board ? board.fixed : (['llm', 'embeddings', 'reranker', 'decision'] as Part[])
  return (
    <div className={cn('grid gap-4 md:grid-cols-[auto_1fr] md:items-center', !bare && 'rounded-xl border border-border p-4 sm:p-5')}>
      {board && (
        <div className="md:pr-6 md:border-r border-border">
          <div className="text-xs text-muted-foreground mb-1">Swapped on this board</div>
          <div className="font-heading font-bold text-lg">{board.swaps.map((s) => PART_LABEL[s]).join(' / ')}</div>
        </div>
      )}
      <div>
        <div className="text-xs text-muted-foreground mb-2">{board ? 'Held fixed (reference setup)' : 'Reference setup'}</div>
        <ul className="grid sm:grid-cols-2 gap-x-8 gap-y-1 text-sm list-disc pl-5 marker:text-muted-foreground">
          {[
            ...parts.map((p) => [PART_LABEL[p], REFERENCE[p].model]),
            ['Hindsight', `${RUN.hindsightVersion} @ ${RUN.hindsightCommit}`],
            ['Judge', RUN.judge],
          ].map(([k, v]) => (
            <li key={k}>
              <span className="text-muted-foreground">{k}:</span> <span className="font-mono text-[13px]">{v}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
