'use client'

import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  SortingState,
  flexRender,
  ColumnDef,
} from '@tanstack/react-table'
import { useState, useMemo } from 'react'
import { EmbeddingRow } from '@/lib/embeddings'

function scoreToBackground(score: number): string {
  const hue = Math.round(score * 1.2)
  return `hsl(${hue}, 45%, 12%)`
}

function scoreToBar(score: number): string {
  const hue = Math.round(score * 1.2)
  return `hsl(${hue}, 55%, 22%)`
}

function scoreToText(score: number): string {
  const hue = Math.round(score * 1.2)
  return `hsl(${hue}, 70%, 65%)`
}

function metricCell(value: number, label: string) {
  const score = value * 100
  return (
    <div className="relative px-6 py-4" style={{ backgroundColor: scoreToBackground(score) }}>
      <div className="absolute inset-0" style={{ width: `${score}%`, backgroundColor: scoreToBar(score) }} />
      <div className="relative z-10">
        <div className="text-sm font-semibold" style={{ color: scoreToText(score) }}>{label}</div>
      </div>
    </div>
  )
}

interface EmbeddingsTableProps {
  embeddings: EmbeddingRow[]
}

interface TableRow extends EmbeddingRow {
  fixedRank: number
}

export default function EmbeddingsTable({ embeddings }: EmbeddingsTableProps) {
  const tableData: TableRow[] = useMemo(() => {
    const sorted = [...embeddings].sort((a, b) => b.score.totalScore - a.score.totalScore)
    const rankMap = new Map<string, number>()
    sorted.forEach((r, i) => rankMap.set(r.embedding_id, i + 1))
    return embeddings.map(r => ({ ...r, fixedRank: rankMap.get(r.embedding_id) ?? 999 }))
  }, [embeddings])

  const [sorting, setSorting] = useState<SortingState>([{ id: 'totalScore', desc: true }])

  const columns: ColumnDef<TableRow>[] = [
    {
      id: 'fixedRank',
      accessorFn: row => row.fixedRank,
      header: 'Rank',
      cell: ({ row }) => {
        const rank = row.original.fixedRank
        return (
          <div className="flex items-center gap-2 px-6 py-4">
            <span className="text-base font-semibold text-foreground">{rank}</span>
            {rank === 1 && <span className="text-xl">🏆</span>}
            {rank === 2 && <span className="text-xl">🥈</span>}
            {rank === 3 && <span className="text-xl">🥉</span>}
          </div>
        )
      },
      size: 80,
    },
    {
      id: 'name',
      accessorFn: row => row.embedding_id,
      header: 'Embedding Model',
      cell: ({ row }) => (
        <div className="px-6 py-4">
          <div className="text-sm font-medium text-foreground">{row.original.config.display_name}</div>
          {row.original.model && (
            <div className="text-xs text-muted-foreground font-mono mt-0.5">{row.original.model}</div>
          )}
          {row.original.config.dimensions && (
            <div className="text-xs text-muted-foreground mt-0.5">{row.original.config.dimensions}-dim</div>
          )}
        </div>
      ),
      size: 280,
    },
    {
      id: 'provider',
      accessorFn: row => row.config.provider_name,
      header: 'Provider',
      cell: ({ row }) => {
        const { provider_name, provider_icon } = row.original.config
        if (!provider_name) return <div className="px-6 py-4" />
        return (
          <div className="flex items-center gap-2 px-6 py-4">
            {provider_icon && (
              <img src={provider_icon} alt={provider_name} className="w-5 h-5 object-contain" />
            )}
            <span className="text-sm text-muted-foreground">{provider_name}</span>
          </div>
        )
      },
      size: 140,
    },
    {
      id: 'totalScore',
      accessorFn: row => row.score.totalScore,
      header: () => (
        <div>
          <div>Total Score</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">MRR + Speed + Cost</div>
        </div>
      ),
      cell: ({ getValue }) => {
        const value = getValue() as number
        return (
          <div className="flex items-center gap-2 px-6 py-4">
            <div className="w-20 bg-secondary rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all"
                style={{ width: `${value}%`, background: 'linear-gradient(to right, #0074d9, #009296)' }}
              />
            </div>
            <span className="text-sm font-bold text-foreground min-w-[3rem]">{value.toFixed(1)}</span>
          </div>
        )
      },
      size: 200,
    },
    {
      id: 'mrr',
      accessorFn: row => row.mrr,
      header: () => (
        <div>
          <div>MRR</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">Mean Reciprocal Rank</div>
        </div>
      ),
      cell: ({ getValue }) => {
        const v = getValue() as number
        return metricCell(v, v.toFixed(3))
      },
      size: 130,
    },
    {
      id: 'recall_at_1',
      accessorFn: row => row.recall_at_1,
      header: () => (
        <div>
          <div>R@1</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">Recall at 1</div>
        </div>
      ),
      cell: ({ getValue }) => {
        const v = getValue() as number
        return metricCell(v, `${(v * 100).toFixed(1)}%`)
      },
      size: 110,
    },
    {
      id: 'recall_at_3',
      accessorFn: row => row.recall_at_3,
      header: () => (
        <div>
          <div>R@3</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">Recall at 3</div>
        </div>
      ),
      cell: ({ getValue }) => {
        const v = getValue() as number
        return metricCell(v, `${(v * 100).toFixed(1)}%`)
      },
      size: 110,
    },
    {
      id: 'recall_at_5',
      accessorFn: row => row.recall_at_5,
      header: () => (
        <div>
          <div>R@5</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">Recall at 5</div>
        </div>
      ),
      cell: ({ getValue }) => {
        const v = getValue() as number
        return metricCell(v, `${(v * 100).toFixed(1)}%`)
      },
      size: 110,
    },
    {
      id: 'speedScore',
      accessorFn: row => row.score.speedScore,
      header: () => (
        <div>
          <div>Speed</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">avg s/recall</div>
        </div>
      ),
      cell: ({ getValue, row }) => {
        const score = getValue() as number
        const latency = row.original.avg_latency_s
        return (
          <div className="relative px-6 py-4" style={{ backgroundColor: scoreToBackground(score) }}>
            <div className="absolute inset-0" style={{ width: `${score}%`, backgroundColor: scoreToBar(score) }} />
            <div className="relative z-10">
              <div className="text-sm font-semibold" style={{ color: scoreToText(score) }}>{score.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground">{latency.toFixed(2)}s/call</div>
            </div>
          </div>
        )
      },
      size: 130,
    },
    {
      id: 'costScore',
      accessorFn: row => row.score.costScore,
      header: () => (
        <div>
          <div>Cost</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">$ per recall()</div>
        </div>
      ),
      cell: ({ getValue, row }) => {
        const score = getValue() as number
        const { pricePerQuery, pricingType } = row.original.score
        const priceLabel = pricingType === 'free' || pricePerQuery === 0
          ? 'Free'
          : pricePerQuery < 0.0001
            ? `$${(pricePerQuery * 1_000_000).toFixed(2)}/1M`
            : `$${pricePerQuery.toFixed(5)}`
        return (
          <div className="relative px-6 py-4" style={{ backgroundColor: scoreToBackground(score) }}>
            <div className="absolute inset-0" style={{ width: `${score}%`, backgroundColor: scoreToBar(score) }} />
            <div className="relative z-10">
              <div className="text-sm font-semibold" style={{ color: scoreToText(score) }}>{score.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground">{priceLabel}</div>
            </div>
          </div>
        )
      },
      size: 130,
    },
  ]

  const table = useReactTable({
    data: tableData,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  return (
    <div className="bg-card rounded-lg border border-border overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-border">
          <thead className="bg-secondary/40">
            {table.getHeaderGroups().map(headerGroup => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map(header => (
                  <th
                    key={header.id}
                    scope="col"
                    className="px-6 py-3 text-left text-xs font-medium text-muted-foreground uppercase tracking-wider cursor-pointer hover:bg-secondary/60 select-none transition-colors"
                    onClick={header.column.getToggleSortingHandler()}
                    style={{ width: header.getSize() }}
                  >
                    <div className="flex items-center gap-2">
                      {flexRender(header.column.columnDef.header, header.getContext())}
                      {header.column.getIsSorted() && (
                        <span className="text-primary text-base">
                          {header.column.getIsSorted() === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="divide-y divide-border">
            {table.getRowModel().rows.map(row => (
              <tr key={row.id} className="hover:bg-secondary/20 transition-colors">
                {row.getVisibleCells().map(cell => (
                  <td key={cell.id} className="whitespace-nowrap text-left">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
