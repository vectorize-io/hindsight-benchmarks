import { BenchmarkRun, ModelConfig, QualityResult, ReflectResult } from './types'

export interface Provider {
  name: string
  iconUrl: string
}

export interface ModelScore {
  totalScore: number        // 0-100 composite score
  qualityScore: number      // 0-100 BEAM facts-only accuracy
  efficiencyScore: number   // 0-100 accuracy per stored fact token
  speedScore: number        // 0-100 based on latency
  costScore: number         // 0-100 based on cost per request
  conformanceScore: number  // 0-100 valid-JSON rate
  storedFactTokens: number | null // extraction footprint (tokens stored in memory)
  avgCostPerRequest: number // $ per request (total)
  avgInputCost: number      // $ per request (input)
  avgOutputCost: number     // $ per request (output)
  inputPricePerM: number    // $ per 1M input tokens
  outputPricePerM: number   // $ per 1M output tokens
  provider: Provider        // Provider info with icon
}

// Quality baseline - accuracy score (0-100)
// We use accuracy from LoComo conversation testing via Hindsight
const BASELINE_QUALITY = 100 // Perfect score

// Reference point for speed (retain/extraction): a model at this latency (seconds) scores 50.
// Score = 100 * REF / (REF + latency) → approaches 100 as latency → 0, never reaches it.
// Calibrated for fact-extraction latencies (10–30s range).
const LATENCY_REFERENCE = 10.0

// Reference point for speed (reflect): reflect() calls are single API calls (2–10s typical).
// A model at 5s scores 50; faster models score higher.
const REFLECT_LATENCY_REFERENCE = 5.0

// Fixed token counts used to normalize pricing into a comparable per-request cost.
// Same pricing → same cost score, regardless of actual output verbosity.
const COST_REF_INPUT_TOKENS = 1000
const COST_REF_OUTPUT_TOKENS = 500

// Reference point for cost: a model at this normalized cost scores 50.
// Score = 100 * REF / (REF + normalizedCost)
const COST_REFERENCE = 0.001

// Reference point for efficiency: accuracy points per 1k stored fact tokens at
// which a model scores 50. Frozen at the median of the first full BEAM fleet
// run (15 models, raw values 0.163-0.548).
const EFFICIENCY_REFERENCE = 0.306

// Accuracy anchors for the quality subscore: this benchmark's accuracies cluster
// in a narrow band, so the raw number is rescaled to 0-100 before weighting.
// Fixed rather than field-relative, so adding a model leaves other scores alone.
const QUALITY_FLOOR = 25
const QUALITY_CEILING = 65

function scaleQuality(accuracy: number): number {
  const scaled = ((accuracy - QUALITY_FLOOR) / (QUALITY_CEILING - QUALITY_FLOOR)) * 100
  return Math.max(0, Math.min(100, scaled))
}

interface ModelWithResult {
  config: ModelConfig
  result: BenchmarkRun | null
  qualityResult?: QualityResult | null
  legacyQualityResult?: QualityResult | null
  reflectResult?: ReflectResult | null
}

export function calculateReflectScore(model: ModelWithResult): ModelScore {
  const inputPricePerM = model.config.input_price_per_1m || 0
  const outputPricePerM = model.config.output_price_per_1m || 0
  const provider = { name: model.config.provider_name, iconUrl: model.config.provider_icon }

  const reflectResult = model.reflectResult
  if (!reflectResult || reflectResult.total === 0) {
    return {
      totalScore: 0,
      qualityScore: 0,
      efficiencyScore: 0,
      speedScore: 0,
      costScore: 0,
      conformanceScore: 0,
      storedFactTokens: null,
      avgCostPerRequest: 0,
      avgInputCost: 0,
      avgOutputCost: 0,
      inputPricePerM,
      outputPricePerM,
      provider,
    }
  }

  // Quality Score (60% weight) - reflect accuracy on LoComo
  const qualityScore = reflectResult.accuracy

  // Speed Score (25% weight) - avg latency per reflect() call
  const latency = reflectResult.avg_latency_s || 0
  const speedScore = latency === 0
    ? 0
    : (REFLECT_LATENCY_REFERENCE / (REFLECT_LATENCY_REFERENCE + latency)) * 100

  // Cost Score (15% weight) - pricing only
  const normalizedCost =
    (COST_REF_INPUT_TOKENS * inputPricePerM + COST_REF_OUTPUT_TOKENS * outputPricePerM) / 1_000_000
  const costScore = normalizedCost === 0
    ? 100
    : (COST_REFERENCE / (COST_REFERENCE + normalizedCost)) * 100

  const totalScore =
    (qualityScore * 0.60) +
    (speedScore * 0.25) +
    (costScore * 0.15)

  return {
    totalScore: Math.round(totalScore * 10) / 10,
    qualityScore: Math.round(qualityScore * 10) / 10,
    efficiencyScore: 0,
    speedScore: Math.round(speedScore * 10) / 10,
    costScore: Math.round(costScore * 10) / 10,
    conformanceScore: 0,
    storedFactTokens: null,
    avgCostPerRequest: 0,
    avgInputCost: 0,
    avgOutputCost: 0,
    inputPricePerM,
    outputPricePerM,
    provider,
  }
}

// Legacy retain scoring: the LoComo-era formula (Quality 40 / Speed 25 /
// Cost 20 / Reliability 15), fed by the legacy quality numbers. Serves only
// /leaderboard/retain-legacy, kept until the old board is retired.
export function calculateLegacyModelScore(model: ModelWithResult): ModelScore {
  const inputPricePerM = model.config.input_price_per_1m || 0
  const outputPricePerM = model.config.output_price_per_1m || 0
  const provider = { name: model.config.provider_name, iconUrl: model.config.provider_icon }

  const qualityScore = model.legacyQualityResult?.accuracy || 0

  const normalizedCost =
    (COST_REF_INPUT_TOKENS * inputPricePerM + COST_REF_OUTPUT_TOKENS * outputPricePerM) / 1_000_000
  const hasCostData = model.result ? (model.result.summary?.success ?? 0) > 0 : !!model.legacyQualityResult?.total
  const costScore = !hasCostData
    ? 0
    : normalizedCost === 0
      ? 100
      : (COST_REFERENCE / (COST_REFERENCE + normalizedCost)) * 100

  const summary = model.result?.summary
  const latency = summary?.avg_latency_s || 0
  const speedScore = latency === 0
    ? 0
    : (LATENCY_REFERENCE / (LATENCY_REFERENCE + latency)) * 100
  const reliabilityScore = summary?.total
    ? (summary.success / summary.total) * 100
    : 0

  const totalScore = model.result
    ? (qualityScore * 0.40) + (speedScore * 0.25) + (costScore * 0.20) + (reliabilityScore * 0.15)
    : (qualityScore * 0.40) + (costScore * 0.20)

  return {
    totalScore: Math.round(totalScore * 10) / 10,
    qualityScore: Math.round(qualityScore * 10) / 10,
    efficiencyScore: 0,
    speedScore: Math.round(speedScore * 10) / 10,
    costScore: Math.round(costScore * 10) / 10,
    conformanceScore: Math.round(reliabilityScore * 10) / 10,
    storedFactTokens: null,
    avgCostPerRequest: 0,
    avgInputCost: 0,
    avgOutputCost: 0,
    inputPricePerM,
    outputPricePerM,
    provider,
  }
}

export function calculateModelScore(model: ModelWithResult): ModelScore {
  const inputPricePerM = model.config.input_price_per_1m || 0
  const outputPricePerM = model.config.output_price_per_1m || 0
  const provider = { name: model.config.provider_name, iconUrl: model.config.provider_icon }

  // 1. Quality Score (60% weight) - BEAM facts-only accuracy via Hindsight
  const accuracy = model.qualityResult?.accuracy || 0
  const qualityScore = scaleQuality(accuracy)

  // 2. Efficiency Score (20% weight) - accuracy per 1k stored fact tokens.
  // Punishes extractors that dump near-verbatim conversation into memory.
  // A missing token count means the quality run was incomplete; the row scores
  // 0 on this dimension (shown as a dash) until the model is re-run.
  const storedFactTokens = model.qualityResult?.stored_fact_tokens ?? null
  const efficiencyRaw = storedFactTokens && storedFactTokens > 0
    ? accuracy / (storedFactTokens / 1000)
    : null
  const efficiencyScore = efficiencyRaw === null
    ? 0
    : (efficiencyRaw / (efficiencyRaw + EFFICIENCY_REFERENCE)) * 100

  // 3. Cost Score (5% weight) - based purely on pricing, not actual token usage.
  const normalizedCost =
    (COST_REF_INPUT_TOKENS * inputPricePerM + COST_REF_OUTPUT_TOKENS * outputPricePerM) / 1_000_000
  const hasCostData = model.result ? (model.result.summary?.success ?? 0) > 0 : !!model.qualityResult?.total
  const costScore = !hasCostData
    ? 0
    : normalizedCost === 0
      ? 100
      : (COST_REFERENCE / (COST_REFERENCE + normalizedCost)) * 100

  if (!model.result) {
    // Only quality data available — omit speed and conformance from total score weighting
    const totalScore = qualityScore * 0.60 + efficiencyScore * 0.20 + costScore * 0.05
    return {
      totalScore: Math.round(totalScore * 10) / 10,
      qualityScore: Math.round(qualityScore * 10) / 10,
      efficiencyScore: Math.round(efficiencyScore * 10) / 10,
      speedScore: 0,
      costScore: Math.round(costScore * 10) / 10,
      conformanceScore: 0,
      storedFactTokens,
      avgCostPerRequest: 0,
      avgInputCost: 0,
      avgOutputCost: 0,
      inputPricePerM,
      outputPricePerM,
      provider,
    }
  }

  const summary = model.result.summary

  // 4. Speed Score (10% weight) - hyperbolic decay on latency
  const latency = summary?.avg_latency_s || 0
  const speedScore = latency === 0
    ? 0
    : (LATENCY_REFERENCE / (LATENCY_REFERENCE + latency)) * 100

  // Actual per-request cost (for display only, not used in scoring)
  const totalPromptTokens = model.result.tests.reduce((sum, t) => sum + t.prompt_tokens, 0)
  const totalCompletionTokens = model.result.tests.reduce((sum, t) => sum + t.completion_tokens, 0)
  const inputCost = (totalPromptTokens / 1_000_000) * inputPricePerM
  const outputCost = (totalCompletionTokens / 1_000_000) * outputPricePerM
  const totalCost = inputCost + outputCost
  const avgCostPerRequest = summary?.success ? totalCost / summary.success : 0
  const avgPromptTokens = summary?.success ? totalPromptTokens / summary.success : 0
  const avgCompletionTokens = summary?.success ? totalCompletionTokens / summary.success : 0
  const avgInputCost = (avgPromptTokens / 1_000_000) * inputPricePerM
  const avgOutputCost = (avgCompletionTokens / 1_000_000) * outputPricePerM

  // 5. JSON Conformance Score (5% weight) - valid-JSON rate on the extraction tests
  const conformanceScore = summary?.total
    ? (summary.success / summary.total) * 100
    : 0

  const totalScore =
    (qualityScore * 0.60) +
    (efficiencyScore * 0.20) +
    (conformanceScore * 0.05) +
    (speedScore * 0.10) +
    (costScore * 0.05)

  return {
    totalScore: Math.round(totalScore * 10) / 10,
    qualityScore: Math.round(qualityScore * 10) / 10,
    efficiencyScore: Math.round(efficiencyScore * 10) / 10,
    speedScore: Math.round(speedScore * 10) / 10,
    costScore: Math.round(costScore * 10) / 10,
    conformanceScore: Math.round(conformanceScore * 10) / 10,
    storedFactTokens,
    avgCostPerRequest: Math.round(avgCostPerRequest * 1000) / 1000,
    avgInputCost: Math.round(avgInputCost * 1000) / 1000,
    avgOutputCost: Math.round(avgOutputCost * 1000) / 1000,
    inputPricePerM,
    outputPricePerM,
    provider,
  }
}
