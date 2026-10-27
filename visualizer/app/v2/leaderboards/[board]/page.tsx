import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import { BOARDS, boardById } from '@/lib/v2/data'
import { BoardView } from '@/components/v2/BoardView'
import { BoardNav, MockNotice, PageHero, SetupStrip } from '@/components/v2/Page'

export function generateStaticParams() {
  return BOARDS.map((b) => ({ board: b.id }))
}

export async function generateMetadata({ params }: { params: Promise<{ board: string }> }): Promise<Metadata> {
  const board = boardById((await params).board)
  return board ? { title: `${board.title} leaderboard`, description: board.description } : {}
}

export default async function BoardPage({ params }: { params: Promise<{ board: string }> }) {
  const board = boardById((await params).board)
  if (!board) notFound()
  const cases = board.categories.reduce((s, c) => s + c.cases, 0)

  return (
    <>
      <PageHero title={`${board.title} leaderboard`} subtitle={board.description}>
        <MockNotice />
      </PageHero>
      <BoardNav current={board.id} />
      <main className="max-w-6xl mx-auto px-4 pt-8 space-y-8">
        <SetupStrip board={board} />
        <div className="flex items-baseline justify-between gap-4 flex-wrap">
          <h2 className="font-heading font-bold text-2xl tracking-tight">Results</h2>
          <span className="text-sm text-muted-foreground">
            {cases} cases in {board.categories.length} categories
          </span>
        </div>
        <BoardView board={board} />
      </main>
    </>
  )
}
