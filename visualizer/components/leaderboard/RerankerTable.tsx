'use client'

import { useState } from 'react'
import type { RerankerRow } from '@/lib/reranker'

type SortKey = 'ndcg_at_10' | 'mrr' | 'recall_at_5' | 'p50_latency_s'

function quality(row: RerankerRow, key: string, value: number) {
  const interval = row.confidence_intervals[key]
  return <td className="px-4 py-3 tabular-nums">
    <div>{value.toFixed(3)}</div>
    {interval && <div className="text-xs text-muted-foreground">
      [{interval[0].toFixed(3)}, {interval[1].toFixed(3)}]
    </div>}
  </td>
}

export default function RerankerTable({ rerankers }: { rerankers: RerankerRow[] }) {
  const [sort, setSort] = useState<SortKey>('ndcg_at_10')
  const rows = [...rerankers].sort((a, b) => sort === 'p50_latency_s'
    ? a[sort] - b[sort] : b[sort] - a[sort])
  return <div>
    <label className="flex gap-3 items-center mb-4 text-sm">
      Sort by
      <select value={sort} onChange={e => setSort(e.target.value as SortKey)} className="bg-secondary border border-border rounded px-3 py-2">
        <option value="ndcg_at_10">nDCG@10</option>
        <option value="mrr">MRR</option>
        <option value="recall_at_5">Recall@5</option>
        <option value="p50_latency_s">Median latency</option>
      </select>
    </label>
    <div className="overflow-x-auto border border-border rounded-lg">
      <table className="w-full text-sm text-left">
        <thead className="bg-secondary text-muted-foreground">
          <tr>{['Model', 'nDCG@10', 'MRR', 'Recall@5', 'Recall@10', 'Hit@1', 'p50 / p95 (s)', 'API $ / 1K queries', 'Retries'].map(name =>
            <th key={name} className="px-4 py-3 whitespace-nowrap">{name}</th>)}</tr>
        </thead>
        <tbody>{rows.map(row => <tr key={row.reranker_id} className="border-t border-border">
          <td className="px-4 py-3 min-w-48">
            <div className="font-medium">{row.config.display_name}</div>
            <div className="text-xs text-muted-foreground">{row.config.provider_name}</div>
            <a className="text-xs underline text-muted-foreground" href={`https://github.com/vectorize-io/hindsight-benchmarks/blob/main/results/leaderboard/reranker/${row.reranker_id}--${row.dataset}.json`}>Run details</a>
          </td>
          {quality(row, 'ndcg_at_10', row.ndcg_at_10)}
          {quality(row, 'mrr', row.mrr)}
          {quality(row, 'recall_at_5', row.recall_at_5)}
          {quality(row, 'recall_at_10', row.recall_at_10)}
          {quality(row, 'hit_at_1', row.hit_at_1)}
          <td className="px-4 py-3 tabular-nums whitespace-nowrap">{row.p50_latency_s.toFixed(3)} / {row.p95_latency_s.toFixed(3)}</td>
          <td className="px-4 py-3" title={row.cost_basis}>{row.estimated_api_cost_usd === null
            ? (row.provider === 'local' ? 'Local compute' : 'Rate unverified')
            : `$${(row.estimated_api_cost_usd / row.total_questions * 1000).toFixed(3)}`}</td>
          <td className="px-4 py-3 tabular-nums">{row.retries}</td>
        </tr>)}</tbody>
      </table>
    </div>
    <p className="text-xs text-muted-foreground mt-3">Brackets show 95% bootstrap intervals. Nearby scores do not establish a significant difference. All displayed runs completed every eligible query.</p>
  </div>
}
