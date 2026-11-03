import { Fragment } from 'react'
import Link from 'next/link'
import { BOARDS, PART_LABEL, REFERENCE, RUN, USE_CASES, type Board, type Part } from '@/lib/data'
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

export function BoardNav({ current }: { current?: string }) {
  const items = [
    { href: '/leaderboards', title: 'Overview' },
    ...BOARDS.map((b, i) => ({
      href: `/leaderboards/${b.id}`,
      title: b.title,
      // One rule before the operation boards: All operations sums them up, so it sits with them.
      sep: i === 0,
    })),
    ...USE_CASES.map((u, i) => ({
      href: `/leaderboards/${u.id}`,
      title: u.title,
      sep: i === 0,
    })),
  ]
  return <SubNav items={items} current={current ? `/leaderboards/${current}` : '/leaderboards'} />
}

// The sticky row of section links under a page hero; each link is its own URL, so it can be shared.
export function SubNav({ items, current }: { items: { href: string; title: string; sep?: boolean }[]; current: string }) {
  return (
    <div className="sticky top-16 z-40 bg-background/95 backdrop-blur border-b border-border">
      <div className="max-w-6xl mx-auto px-4 flex gap-1 overflow-x-auto py-2">
        {items.map((i) => (
          <Fragment key={i.href}>
            {i.sep && <span className="shrink-0 w-px h-4 self-center bg-border mx-1.5" aria-hidden />}
            <Link
              href={i.href}
              className={cn(
                'shrink-0 inline-flex items-center h-8 px-3 rounded-[0.625rem] font-heading font-bold text-sm transition-colors',
                i.href === current ? 'nav-active' : 'text-muted-foreground hover:text-foreground hover:bg-secondary',
              )}
            >
              {i.title}
            </Link>
          </Fragment>
        ))}
      </div>
    </div>
  )
}

export function SetupStrip({ board, bare }: { board?: Board; bare?: boolean }) {
  const parts: Part[] = board ? board.fixed : (['llm', 'embeddings', 'reranker'] as Part[])
  const fixed = board?.setup ?? [
    ...parts.map((p) => [PART_LABEL[p], REFERENCE[p].model]),
    ['Hindsight', RUN.hindsightVersion],
    ['Judge', RUN.judge],
  ]
  return (
    <div className={cn(!bare && 'rounded-xl border border-border p-4 sm:p-5')}>
      {board && <p className="text-sm text-muted-foreground mb-4 pb-4 border-b border-border">{board.about}</p>}
      <div className="grid gap-4 md:grid-cols-[auto_1fr] md:items-center">
        {board && (
          <div className="md:pr-6 md:border-r border-border">
            <div className="text-xs text-muted-foreground mb-1">Swapped on this board</div>
            <div className="font-heading font-bold text-lg">{board.swaps.map((s) => PART_LABEL[s]).join(' / ')}</div>
          </div>
        )}
        <div>
          <div className="text-xs text-muted-foreground mb-2">{board ? 'Held fixed' : 'Reference setup'}</div>
          <ul className="grid sm:grid-cols-2 gap-x-8 gap-y-1 text-sm list-disc pl-5 marker:text-muted-foreground">
            {fixed.map(([k, v]) => (
              <li key={k}>
                <span className="text-muted-foreground">{k}:</span> <span className="font-mono text-[13px]">{v}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}
