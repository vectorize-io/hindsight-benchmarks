import fs from 'fs'
import path from 'path'
import { RerankerResult } from './types'

// Speed reference: a reranker at this latency scores 50.
// Calibrated for recall() reranking latencies (0.4–4s range).
const LATENCY_REFERENCE = 1.0

// Cost reference: a reranker at this price per search scores 50.
// Cohere costs $0.002/search — calibrated so free = 100, Cohere ≈ 33.
const COST_REFERENCE = 0.001

export interface RerankerConfig {
  reranker_id: string
  display_name: string
  provider: string
  provider_name: string
  provider_icon: string
  pricing_type: 'free' | 'pay-per-use'
  price_per_search: number
  model?: string
  notes?: string
}

export interface RerankerScore {
  totalScore: number    // 0–100 composite
  rankingScore: number  // 0–100 from MRR
  speedScore: number    // 0–100 from latency
  costScore: number     // 0–100 from price per search
  pricePerSearch: number
  pricingType: 'free' | 'pay-per-use'
}

export function calculateRerankerScore(
  mrr: number,
  avg_latency_s: number,
  price_per_search: number,
  pricing_type: 'free' | 'pay-per-use',
): RerankerScore {
  const rankingScore = mrr * 100
  const speedScore = avg_latency_s === 0
    ? 0
    : (LATENCY_REFERENCE / (LATENCY_REFERENCE + avg_latency_s)) * 100
  const costScore = price_per_search === 0
    ? 100
    : (COST_REFERENCE / (COST_REFERENCE + price_per_search)) * 100

  // Weights: MRR 70%, Speed 15%, Cost 15%
  const totalScore = rankingScore * 0.70 + speedScore * 0.15 + costScore * 0.15

  return {
    totalScore: Math.round(totalScore * 10) / 10,
    rankingScore: Math.round(rankingScore * 10) / 10,
    speedScore: Math.round(speedScore * 10) / 10,
    costScore: Math.round(costScore * 10) / 10,
    pricePerSearch: price_per_search,
    pricingType: pricing_type,
  }
}

export interface RerankerRow extends RerankerResult {
  config: RerankerConfig
  score: RerankerScore
}

function loadRerankerConfigs(): Map<string, RerankerConfig> {
  const configPath = path.join(process.cwd(), '..', 'benchmark-runner', 'reranker_models.json')
  if (!fs.existsSync(configPath)) return new Map()
  const data = JSON.parse(fs.readFileSync(configPath, 'utf-8'))
  return new Map((data.rerankers as RerankerConfig[]).map(r => [r.reranker_id, r]))
}

export function loadRerankerData(): RerankerRow[] {
  const dir = path.join(process.cwd(), '..', 'results', 'leaderboard', 'reranker')
  if (!fs.existsSync(dir)) return []

  const configs = loadRerankerConfigs()
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.json'))

  return files.map(file => {
    const result: RerankerResult = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf-8'))
    const config: RerankerConfig = configs.get(result.reranker_id) ?? {
      reranker_id: result.reranker_id,
      display_name: result.reranker_id,
      provider: result.provider,
      provider_name: result.provider,
      provider_icon: '',
      pricing_type: 'free',
      price_per_search: 0,
    }
    return {
      ...result,
      config,
      score: calculateRerankerScore(
        result.mrr,
        result.avg_latency_s,
        config.price_per_search,
        config.pricing_type,
      ),
    }
  })
}

export function getRerankerStats(rows: RerankerRow[]) {
  if (rows.length === 0) return { count: 0, topReranker: null }
  const top = [...rows].sort((a, b) => b.score.totalScore - a.score.totalScore)[0]
  return {
    count: rows.length,
    topReranker: top ? { id: top.reranker_id, score: top.score.totalScore } : null,
  }
}
