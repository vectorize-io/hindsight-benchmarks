import type { Row } from '@/lib/v2/data'
import { cn } from '@/lib/utils'

// Shared by server pages and the client board, so no 'use client' here.
export const fmtCost = (v: number) => (v === 0 ? 'free' : v < 0.01 ? '<$0.01' : `$${v < 10 ? v.toFixed(2) : v.toFixed(0)}`)
export const fmtTokens = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k` : String(v))

export function ModelCell({ row }: { row: Row }) {
  return (
    <div className="flex items-center gap-2.5 min-w-0">
      {row.icon ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={row.icon} alt="" className="w-5 h-5 object-contain shrink-0 rounded-sm" />
      ) : (
        <span className="w-5 h-5 shrink-0 rounded-sm bg-secondary text-[10px] font-bold flex items-center justify-center text-muted-foreground">
          {row.provider[0]}
        </span>
      )}
      <div className="min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-[13px] font-medium truncate">{row.model}</span>
          {row.tag === 'reference' && <Tag className="text-[hsl(var(--teal))] border-[hsl(var(--teal))]/40">reference</Tag>}
          {row.tag === 'baseline' && <Tag className="text-muted-foreground border-border">baseline</Tag>}
        </div>
        <div className="text-xs text-muted-foreground truncate">
          {row.provider}
          {row.openWeights && ' · open weights'}
        </div>
      </div>
    </div>
  )
}

function Tag({ children, className }: { children: React.ReactNode; className?: string }) {
  return <span className={cn('text-[10px] font-semibold uppercase tracking-wide border rounded px-1 py-px', className)}>{children}</span>
}

