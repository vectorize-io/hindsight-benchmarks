import fs from 'fs'
import path from 'path'
import { ModelConfig, ModelWithResult, BenchmarkRun, QualityResult, ReflectResult, DeploymentInfo } from './types'
import { calculateModelScore, calculateReflectScore } from './scoring'

export function loadLeaderboardData(): ModelWithResult[] {
  // Load model configurations
  const configPath = path.join(process.cwd(), '..', 'benchmark-runner', 'benchmark_models.json')
  const leaderboardDir = path.join(process.cwd(), '..', 'results', 'leaderboard', 'llm')

  // Read benchmark models config
  const configData = JSON.parse(fs.readFileSync(configPath, 'utf-8'))
  const models: ModelConfig[] = configData.models

  // Read all unified result files
  const unifiedFiles = fs.existsSync(leaderboardDir)
    ? fs.readdirSync(leaderboardDir).filter(f => f.endsWith('.json'))
    : []

  // Build maps keyed by "provider_id-model_id"
  const retainMap = new Map<string, BenchmarkRun>()
  const qualityMap = new Map<string, QualityResult>()
  const legacyQualityMap = new Map<string, QualityResult>()
  const reflectMap = new Map<string, ReflectResult>()
  const deploymentMap = new Map<string, DeploymentInfo>()

  for (const file of unifiedFiles) {
    const filePath = path.join(leaderboardDir, file)
    const unified = JSON.parse(fs.readFileSync(filePath, 'utf-8'))
    const key = `${unified.provider_id}-${unified.model_id}`

    if (unified.retain)   retainMap.set(key, unified.retain)
    // Quality blocks without a dataset field are legacy LoComo runs, not
    // comparable to the BEAM facts-only benchmark; they feed only the legacy
    // leaderboard. Partial blocks come from truncated smoke runs; their
    // accuracy and efficiency are not comparable to anything. Skip those.
    if (unified.quality?.partial) {
      // dropped
    } else if (unified.quality?.dataset) {
      qualityMap.set(key, unified.quality)
    } else if (unified.quality) {
      legacyQualityMap.set(key, unified.quality)
    }
    if (unified.reflect)  reflectMap.set(key, unified.reflect)
    if (unified.deployment) deploymentMap.set(key, unified.deployment)
  }

  // Match configs with results and quality scores
  const modelsWithResults: ModelWithResult[] = models.map(config => {
    const key = `${config.provider_id}-${config.model_id}`
    return {
      config,
      result:              retainMap.get(key)        || null,
      qualityResult:       qualityMap.get(key)       || null,
      legacyQualityResult: legacyQualityMap.get(key) || null,
      reflectResult:       reflectMap.get(key)       || null,
      deployment:          deploymentMap.get(key)    || null,
    }
  })

  return modelsWithResults
}

export function getLeaderboardStats(models: ModelWithResult[]) {
  const retainModels = models.filter(m => !m.config.benchmarks || m.config.benchmarks.includes('retain'))
  const reflectModels = models.filter(m => m.config.benchmarks?.includes('reflect'))

  const viableRetain = retainModels.filter(m =>
    (m.result && m.result.summary.success > 0) || (m.qualityResult && m.qualityResult.total > 0)
  )
  const viableReflect = reflectModels.filter(m => m.reflectResult && m.reflectResult.total > 0)

  const topRetain = viableRetain
    .map(m => ({ model: m, score: calculateModelScore(m).totalScore }))
    .sort((a, b) => b.score - a.score)[0]?.model ?? null

  const topReflect = viableReflect
    .map(m => ({ model: m, score: calculateReflectScore(m).totalScore }))
    .sort((a, b) => b.score - a.score)[0]?.model ?? null

  return {
    viableModels: viableRetain.length,
    topRetainModel: topRetain ? { name: topRetain.config.model_name, providerIcon: topRetain.config.provider_icon, providerName: topRetain.config.provider_name } : null,
    viableReflectModels: viableReflect.length,
    topReflectModel: topReflect ? { name: topReflect.config.model_name, providerIcon: topReflect.config.provider_icon, providerName: topReflect.config.provider_name } : null,
  }
}
