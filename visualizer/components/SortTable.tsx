'use client'

import { useState } from 'react'
import { ArrowDown } from 'lucide-react'
import { cn } from '@/lib/utils'

// The leaderboard table's sorting for tables a server page renders: cells come in ready-made,
// with one value per column to sort on. `low` columns sort smallest first (cost, time).
export type SortCol = { label: string; tip?: string; right?: boolean; low?: boolean; sortable?: boolean }
export type SortRow = { key: string; cells: React.ReactNode[]; values: (number | string | null)[]; className?: string }

export function SortTable({ cols, rows, initial }: { cols: SortCol[]; rows: SortRow[]; initial: number }) {
  const [sort, setSort] = useState(initial)
  const low = cols[sort].low
  const sorted = [...rows].sort((a, b) => {
    const [x, y] = [a.values[sort], b.values[sort]]
    if (x == null || y == null) return x == null ? 1 : -1
    const d = typeof x === 'string' ? x.localeCompare(String(y)) : x - (y as number)
    return typeof x === 'string' || low ? d : -d
  })
  return (
    <div className="rounded-xl border border-border overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-xs text-muted-foreground text-left bg-[hsl(var(--band))]">
            {cols.map((c, i) => (
              <th key={c.label} title={c.tip} className={cn('px-3 py-2.5 font-medium whitespace-nowrap', c.right && 'text-right')}>
                {c.sortable === false ? (
                  c.label
                ) : (
                  <button onClick={() => setSort(i)} className={cn('inline-flex items-center gap-1 hover:text-foreground', sort === i && 'text-foreground')}>
                    {c.label}
                    {sort === i && <ArrowDown className="w-3 h-3" />}
                  </button>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r) => (
            <tr key={r.key} className={cn('border-b border-border last:border-0', r.className)}>
              {r.cells.map((c, i) => (
                <td key={i} className={cn('px-3 py-3', cols[i].right && 'text-right')}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
