import fs from 'fs'
import path from 'path'
import { EmbeddingsResult } from './types'

// Speed reference: a model at this latency scores 50.
const LATENCY_REFERENCE = 1.0

// Cost reference: a model at this price per 1M tokens scores 50.
// $0.10/1M tokens (roughly Cohere pricing) → score ~50.
const COST_REFERENCE_PER_1M = 0.10

export interface EmbeddingConfig {
  embedding_id: string
  display_name: string
  provider: string
  provider_name: string
  provider_icon: string
  pricing_type: 'free' | 'pay-per-use'
  price_per_1m_tokens: number  // $ per 1M tokens (applies to both ingestion and queries)
  model?: string
  dimensions?: number
  notes?: string
}

export interface EmbeddingScore {
  totalScore: number       // 0–100 composite
  rankingScore: number     // 0–100 from MRR
  speedScore: number       // 0–100 from latency
  costScore: number        // 0–100 from price per 1M tokens
  pricePer1mTokens: number
  pricingType: 'free' | 'pay-per-use'
}

export function calculateEmbeddingScore(
  mrr: number,
  avg_latency_s: number,
  price_per_1m_tokens: number,
  pricing_type: 'free' | 'pay-per-use',
): EmbeddingScore {
  const rankingScore = mrr * 100
  const speedScore = avg_latency_s === 0
    ? 0
    : (LATENCY_REFERENCE / (LATENCY_REFERENCE + avg_latency_s)) * 100
  const costScore = price_per_1m_tokens === 0
    ? 100
    : (COST_REFERENCE_PER_1M / (COST_REFERENCE_PER_1M + price_per_1m_tokens)) * 100

  // Weights: MRR 70%, Speed 15%, Cost 15%
  const totalScore = rankingScore * 0.70 + speedScore * 0.15 + costScore * 0.15

  return {
    totalScore: Math.round(totalScore * 10) / 10,
    rankingScore: Math.round(rankingScore * 10) / 10,
    speedScore: Math.round(speedScore * 10) / 10,
    costScore: Math.round(costScore * 10) / 10,
    pricePer1mTokens: price_per_1m_tokens,
    pricingType: pricing_type,
  }
}

export interface EmbeddingRow extends EmbeddingsResult {
  config: EmbeddingConfig
  score: EmbeddingScore
}

function loadEmbeddingConfigs(): Map<string, EmbeddingConfig> {
  const configPath = path.join(process.cwd(), '..', 'benchmark-runner', 'embedding_models.json')
  if (!fs.existsSync(configPath)) return new Map()
  const data = JSON.parse(fs.readFileSync(configPath, 'utf-8'))
  return new Map((data.embeddings as EmbeddingConfig[]).map(e => [e.embedding_id, e]))
}

export function loadEmbeddingsData(): EmbeddingRow[] {
  const dir = path.join(process.cwd(), '..', 'results', 'leaderboard', 'embeddings')
  if (!fs.existsSync(dir)) return []

  const configs = loadEmbeddingConfigs()
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.json'))

  return files.map(file => {
    const result: EmbeddingsResult = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf-8'))
    const config: EmbeddingConfig = configs.get(result.embedding_id) ?? {
      embedding_id: result.embedding_id,
      display_name: result.embedding_id,
      provider: result.provider,
      provider_name: result.provider,
      provider_icon: '',
      pricing_type: 'free',
      price_per_1m_tokens: 0,
    }
    return {
      ...result,
      config,
      score: calculateEmbeddingScore(
        result.mrr,
        result.avg_latency_s,
        config.price_per_1m_tokens,
        config.pricing_type,
      ),
    }
  })
}

export function getEmbeddingsStats(rows: EmbeddingRow[]) {
  if (rows.length === 0) return { count: 0, topEmbedding: null }
  const top = [...rows].sort((a, b) => b.score.totalScore - a.score.totalScore)[0]
  return {
    count: rows.length,
    topEmbedding: top ? {
      id: top.embedding_id,
      name: top.config.display_name,
      providerIcon: top.config.provider_icon,
      providerName: top.config.provider_name,
      score: top.score.totalScore,
    } : null,
  }
}
