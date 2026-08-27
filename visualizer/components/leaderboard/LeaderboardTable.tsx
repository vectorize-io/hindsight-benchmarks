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
import { ModelWithResult } from '@/lib/types'
import { calculateModelScore, calculateReflectScore, calculateLegacyModelScore, ModelScore } from '@/lib/scoring'

// Dark-mode score cell: score 0–100 → dark HSL background (red→yellow→green)
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

interface LeaderboardTableProps {
  models: ModelWithResult[]
  type: 'pay-per-use' | 'subscription' | 'local' | 'mixed'
  variant?: 'retain' | 'reflect' | 'retain-legacy'
}

interface TableRow extends ModelWithResult {
  score: ModelScore
  fixedRank: number
}

export default function LeaderboardTable({ models, type, variant = 'retain' }: LeaderboardTableProps) {
  const scoreCalculator = variant === 'reflect' ? calculateReflectScore
    : variant === 'retain-legacy' ? calculateLegacyModelScore
    : calculateModelScore
  const tableData: TableRow[] = useMemo(() => {
    const withScores = models.map(model => ({
      ...model,
      score: scoreCalculator(model),
    }))

    const sorted = [...withScores].sort((a, b) => b.score.totalScore - a.score.totalScore)
    const rankMap = new Map<string, number>()
    sorted.forEach((model, index) => {
      rankMap.set(model.config.model_id, index + 1)
    })

    return withScores.map(model => ({
      ...model,
      fixedRank: rankMap.get(model.config.model_id) || 999,
    }))
  }, [models])

  const [sorting, setSorting] = useState<SortingState>([
    { id: 'totalScore', desc: true },
  ])

  const columns: ColumnDef<TableRow>[] = [
    {
      id: 'fixedRank',
      accessorFn: (row) => row.fixedRank,
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
      id: 'modelId',
      accessorFn: (row) => row.config.model_name,
      header: 'Model',
      cell: ({ row }) => (
        <div className="px-6 py-4">
          <span className="text-sm font-medium text-foreground">{row.original.config.model_name}</span>
        </div>
      ),
      size: 240,
    },
    ...(type === 'pay-per-use' || type === 'subscription' || type === 'mixed' ? [{
      id: 'provider',
      accessorFn: (row: TableRow) => row.score.provider.name,
      header: 'Provider',
      cell: ({ row }: any) => {
        const provider = row.original.score.provider
        return (
          <div className="flex items-center gap-2 px-6 py-4">
            {provider.iconUrl && (
              <img
                src={provider.iconUrl}
                alt={provider.name}
                className="w-5 h-5 object-contain"
                title={provider.name}
              />
            )}
            <span className="text-sm text-muted-foreground">{provider.name}</span>
          </div>
        )
      },
      size: 140,
    }] : []),
    {
      id: 'totalScore',
      accessorFn: (row) => row.score.totalScore,
      header: () => (
        <div>
          <div>Total Score</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">
            {variant === 'reflect' ? 'Quality + Speed + Cost'
              : variant === 'retain-legacy' ? 'Quality + Speed + Cost + Reliability'
              : 'Quality + Efficiency + Conformance + Speed + Cost'}
          </div>
        </div>
      ),
      cell: ({ getValue }) => {
        const value = getValue() as number
        return (
          <div className="flex items-center gap-2 px-6 py-4">
            <div className="w-20 bg-secondary rounded-full h-2">
              <div
                className="h-2 rounded-full transition-all"
                style={{
                  width: `${value}%`,
                  background: 'linear-gradient(to right, #0074d9, #009296)'
                }}
              />
            </div>
            <span className="text-sm font-bold text-foreground min-w-[3rem]">
              {value.toFixed(1)}
            </span>
          </div>
        )
      },
      size: 200,
    },
    {
      id: 'qualityScore',
      accessorFn: (row) => row.score.qualityScore,
      header: () => (
        <div>
          <div>Quality</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">
            {variant === 'retain' ? 'BEAM extraction accuracy' : 'LoComo accuracy'}
          </div>
        </div>
      ),
      cell: ({ getValue, row }) => {
        const value = getValue() as number
        const accuracy = variant === 'reflect'
          ? (row.original.reflectResult?.accuracy ?? null)
          : variant === 'retain-legacy'
            ? (row.original.legacyQualityResult?.accuracy ?? null)
            : (row.original.qualityResult?.accuracy ?? null)
        const perAbility = variant === 'retain' ? row.original.qualityResult?.per_ability : undefined
        const abilityTooltip = perAbility
          ? Object.entries(perAbility)
              .map(([ability, s]) => `${ability.replace(/_/g, ' ')}: ${s.correct}/${s.total}`)
              .join('\n')
          : undefined
        return (
          <div className="relative px-6 py-4" style={{ backgroundColor: scoreToBackground(value) }} title={abilityTooltip}>
            <div className="absolute inset-0" style={{ width: `${value}%`, backgroundColor: scoreToBar(value) }} />
            <div className="relative z-10">
              <div className="text-sm font-semibold" style={{ color: scoreToText(value) }}>{value.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground">
                {accuracy !== null ? `${accuracy.toFixed(0)}% accuracy` : 'No data'}
              </div>
            </div>
          </div>
        )
      },
      size: 130,
    },
    ...(variant === 'retain' ? [{
      id: 'reasoningEffort',
      accessorFn: (row: TableRow) => {
        const dep = row.deployment
        const kwargs = dep?.llm_extra_body?.chat_template_kwargs
        if (dep?.reasoning_effort === 'none' || kwargs?.enable_thinking === false || row.config.reasoning_effort === 'none') return 'off'
        if (kwargs?.reasoning_effort) return kwargs.reasoning_effort
        if (kwargs?.enable_thinking === true) return 'on'
        return dep?.reasoning_effort || row.config.reasoning_effort || 'default'
      },
      header: () => (
        <div>
          <div>Reasoning</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">Effort tested</div>
        </div>
      ),
      cell: ({ getValue }: any) => {
        const value = getValue() as string
        return (
          <div className="px-6 py-4">
            <span className={`text-sm font-medium ${value === 'off' ? 'text-foreground' : 'text-amber-400'}`}>
              {value}
            </span>
          </div>
        )
      },
      size: 100,
    }, {
      id: 'testConfig',
      accessorFn: (row: TableRow) => row.config.model_id,
      enableSorting: false,
      header: () => (
        <div>
          <div>Serving</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">Context · hardware</div>
        </div>
      ),
      cell: ({ row }: any) => {
        const dep = row.original.deployment
        const ctx = dep?.max_model_len ? `${Math.round(dep.max_model_len / 1024)}K ctx` : null
        const gpu = dep?.accelerator_type
          ? `${dep.accelerator_count || 1}× ${dep.accelerator_type.replace('NVIDIA_', '').replace(/_/g, ' ')}`
          : null
        const ctxFlagged = dep?.container?.args?.some((a: string) => a.startsWith('--max-model-len'))
        const ctxLine = dep?.max_model_len
          ? `context length: ${dep.max_model_len.toLocaleString()} tokens ${ctxFlagged ? '(--max-model-len)' : '(model native; no flag set)'}`
          : null
        const servingTooltip = dep?.container?.args?.length
          ? [dep.container.image, ctxLine, '', ...dep.container.args].filter(l => l !== null).join('\n')
          : ctxLine || undefined
        const hsVersion = row.original.qualityResult?.hindsight_version
        return (
          <div className="px-6 py-4 text-xs text-muted-foreground leading-relaxed" title={servingTooltip}>
            {ctx && <div>{ctx}</div>}
            {gpu && <div>{gpu}</div>}
            {!dep && <div>API provider</div>}
            {hsVersion && <div>Hindsight v{hsVersion}</div>}
          </div>
        )
      },
      size: 130,
    }, {
      id: 'efficiencyScore',
      accessorFn: (row: TableRow) => row.score.efficiencyScore,
      header: () => (
        <div>
          <div>Efficiency</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">Accuracy per stored token</div>
        </div>
      ),
      cell: ({ getValue, row }: any) => {
        const value = getValue() as number
        const storedTokens = row.original.score.storedFactTokens
        if (!storedTokens) {
          return <div className="px-6 py-4 text-sm text-muted-foreground">—</div>
        }
        return (
          <div className="relative px-6 py-4" style={{ backgroundColor: scoreToBackground(value) }}>
            <div className="absolute inset-0" style={{ width: `${value}%`, backgroundColor: scoreToBar(value) }} />
            <div className="relative z-10">
              <div className="text-sm font-semibold" style={{ color: scoreToText(value) }}>{value.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground">{(storedTokens / 1000).toFixed(1)}k tokens stored</div>
            </div>
          </div>
        )
      },
      size: 140,
    }] : []),
    {
      id: 'speedScore',
      accessorFn: (row) => row.score.speedScore,
      header: () => (
        <div>
          <div>Speed</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">
            {variant === 'reflect' ? 'End-to-end per agentic call' : 'Latency + Throughput'}
          </div>
        </div>
      ),
      cell: ({ getValue, row }) => {
        const value = getValue() as number
        if (variant === 'reflect') {
          const latency = row.original.reflectResult?.avg_latency_s || 0
          if (!row.original.reflectResult) {
            return <div className="px-6 py-4 text-sm text-muted-foreground">—</div>
          }
          return (
            <div className="relative px-6 py-4" style={{ backgroundColor: scoreToBackground(value) }}>
              <div className="absolute inset-0" style={{ width: `${value}%`, backgroundColor: scoreToBar(value) }} />
              <div className="relative z-10">
                <div className="text-sm font-semibold" style={{ color: scoreToText(value) }}>{value.toFixed(1)}</div>
                <div className="text-xs text-muted-foreground">
                  {latency > 0 ? `${latency.toFixed(1)}s/call` : '-'}
                </div>
              </div>
            </div>
          )
        }
        if (!row.original.result) {
          return <div className="px-6 py-4 text-sm text-muted-foreground">—</div>
        }
        const latency = row.original.result?.summary?.avg_latency_s || 0
        const speed = row.original.result?.summary?.completion_toks_s || 0
        return (
          <div className="relative px-6 py-4" style={{ backgroundColor: scoreToBackground(value) }}>
            <div className="absolute inset-0" style={{ width: `${value}%`, backgroundColor: scoreToBar(value) }} />
            <div className="relative z-10">
              <div className="text-sm font-semibold" style={{ color: scoreToText(value) }}>{value.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground">
                {latency > 0 ? `${latency.toFixed(1)}s` : '-'}
                {speed > 0 ? ` · ${speed.toFixed(0)} tok/s` : ''}
              </div>
            </div>
          </div>
        )
      },
      size: 150,
    },
    ...(type === 'pay-per-use' || type === 'mixed' ? [{
      id: 'costScore',
      accessorFn: (row: TableRow) => row.score.costScore,
      header: () => (
        <div>
          <div>Cost</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">$ per 1M tokens</div>
        </div>
      ),
      cell: ({ getValue, row }: any) => {
        const value = getValue() as number
        const inputPrice = row.original.score.inputPricePerM
        const outputPrice = row.original.score.outputPricePerM
        const priceNote = row.original.config.price_note

        return (
          <div className="relative px-6 py-4" style={{ backgroundColor: scoreToBackground(value) }}>
            <div className="absolute inset-0" style={{ width: `${value}%`, backgroundColor: scoreToBar(value) }} />
            <div className="relative z-10">
              <div className="text-sm font-semibold" style={{ color: scoreToText(value) }}>{value.toFixed(1)}</div>
              <div
                className={`text-xs text-muted-foreground${priceNote ? ' cursor-help underline decoration-dotted underline-offset-2' : ''}`}
                title={priceNote}
                aria-label={priceNote ? `Price: ${priceNote}` : undefined}
              >
                {inputPrice === 0 && outputPrice === 0 ? (
                  'Free'
                ) : (
                  <div>${inputPrice.toFixed(2)}/${outputPrice.toFixed(2)}</div>
                )}
              </div>
            </div>
          </div>
        )
      },
      size: 140,
    }] : type === 'subscription' ? [{
      id: 'subscription',
      accessorFn: (row: TableRow) => row.score.costScore,
      header: () => (
        <div>
          <div>Subscription</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">Monthly cost</div>
        </div>
      ),
      cell: ({ getValue }: any) => {
        const value = getValue() as number

        return (
          <div className="relative px-6 py-4" style={{ backgroundColor: scoreToBackground(value) }}>
            <div className="absolute inset-0" style={{ width: `${value}%`, backgroundColor: scoreToBar(value) }} />
            <div className="relative z-10">
              <div className="text-sm font-semibold" style={{ color: scoreToText(value) }}>{value.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground">Free tier available</div>
            </div>
          </div>
        )
      },
      size: 140,
    }] : [{
      id: 'modelSize',
      accessorFn: (row: TableRow) => row.config.size_gb,
      header: () => (
        <div>
          <div>Size</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">Model size (GB)</div>
        </div>
      ),
      cell: ({ getValue }: any) => {
        const sizeGb = getValue() as number
        return (
          <div className="px-6 py-4">
            <div className="text-sm font-semibold text-foreground">
              {sizeGb ? `${sizeGb.toFixed(1)} GB` : '-'}
            </div>
          </div>
        )
      },
      size: 120,
    }]),
    ...(variant === 'retain' || variant === 'retain-legacy' ? [{
      id: 'conformanceScore',
      accessorFn: (row: TableRow) => row.score.conformanceScore,
      header: () => (
        <div>
          <div>{variant === 'retain-legacy' ? 'Reliability' : 'JSON Conformance'}</div>
          <div className="text-xs font-normal text-muted-foreground normal-case">
            {variant === 'retain-legacy' ? 'Schema conformance' : 'Valid JSON / total tests'}
          </div>
        </div>
      ),
      cell: ({ getValue, row }: any) => {
        const value = getValue() as number
        if (!row.original.result) {
          return <div className="px-6 py-4 text-sm text-muted-foreground">—</div>
        }
        const summary = row.original.result?.summary
        return (
          <div className="relative px-6 py-4" style={{ backgroundColor: scoreToBackground(value) }}>
            <div className="absolute inset-0" style={{ width: `${value}%`, backgroundColor: scoreToBar(value) }} />
            <div className="relative z-10">
              <div className="text-sm font-semibold" style={{ color: scoreToText(value) }}>{value.toFixed(1)}</div>
              <div className="text-xs text-muted-foreground">
                {summary?.success || 0}/{summary?.total || 0} tests
              </div>
            </div>
          </div>
        )
      },
      size: 130,
    }] : []),
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
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
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
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="hover:bg-secondary/20 transition-colors">
                {row.getVisibleCells().map((cell) => (
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
