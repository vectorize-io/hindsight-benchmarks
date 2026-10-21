import fs from 'fs'
import path from 'path'
import { RerankerResult } from './types'

export interface RerankerConfig {
  reranker_id: string
  display_name: string
  provider: string
  provider_name: string
  provider_icon: string
  pricing_type: 'free' | 'local' | 'pay-per-use'
  model?: string
  notes?: string
}

export interface RerankerRow extends RerankerResult {
  config: RerankerConfig
}

export function loadRerankerData(): RerankerRow[] {
  const dir = path.join(process.cwd(), '..', 'results', 'leaderboard', 'reranker')
  if (!fs.existsSync(dir)) return []
  const configPath = path.join(process.cwd(), '..', 'benchmark-runner', 'reranker_models.json')
  const configs = new Map<string, RerankerConfig>(
    (JSON.parse(fs.readFileSync(configPath, 'utf-8')).rerankers as RerankerConfig[])
      .map(r => [r.reranker_id, r]),
  )
  const results: RerankerRow[] = []
  for (const file of fs.readdirSync(dir).filter(f => f.endsWith('.json'))) {
    const result: RerankerResult = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf-8'))
    if (result.schema_version !== 2 || result.failed_queries !== 0) {
      throw new Error(`Unsupported or incomplete reranker result: ${file}`)
    }
    const config = configs.get(result.reranker_id)
    if (!config) throw new Error(`Missing reranker configuration: ${result.reranker_id}`)
    results.push({ ...result, config })
  }
  // Quality scores are comparable only on exactly the same frozen input fixture.
  for (const dataset of new Set(results.map(r => r.dataset))) {
    if (new Set(results.filter(r => r.dataset === dataset).map(r => `${r.fixture_sha256}:${r.evaluation_label_sha256 ?? ''}`)).size !== 1) {
      throw new Error(`Mixed reranker fixtures or labels for ${dataset}`)
    }
  }
  return results
}

export function getRerankerStats(rows: RerankerRow[]) {
  return { count: new Set(rows.map(r => r.reranker_id)).size }
}
