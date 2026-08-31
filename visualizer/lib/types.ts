export interface BenchmarkResult {
  overall_accuracy: number
  total_correct: number
  total_questions: number
  total_invalid: number
  total_valid: number
  num_items: number
  item_results: ItemResult[]
}

export interface ItemResult {
  item_id: string
  metrics: {
    accuracy: number
    correct: number
    total: number
    invalid: number
    valid_total: number
    category_stats?: Record<string, CategoryStats>
    detailed_results: DetailedResult[]
  }
}

export interface CategoryStats {
  correct: number
  total: number
  invalid: number
}

export interface DetailedResult {
  question: string
  correct_answer: string
  predicted_answer: string
  reasoning: string
  correctness_reasoning?: string
  category: string | number
  is_correct?: boolean
  is_invalid?: boolean
  retrieved_memories?: Memory[]
}

export interface Memory {
  id: string
  text: string
  fact_type: string
  context?: string
  occurred_start?: string
  occurred_end?: string
  mentioned_at?: string
  event_date?: string
  activation?: number
}

// Leaderboard types
export interface BenchmarkTest {
  test_index: number
  latency_s: number
  num_facts: number
  valid_json: boolean
  success: boolean
  retries: number
  prompt_tokens: number
  completion_tokens: number
  error: string
}

export interface BenchmarkSummary {
  success: number
  total: number
  wall_s: number
  avg_latency_s: number | null
  throughput_rps: number | null
  completion_toks_s: number | null
  total_toks_s: number | null
  out_in_ratio: number | null
  tokens_per_fact: number | null
}

export interface BenchmarkRun {
  timestamp: string
  model_id: string
  model_name: string
  provider_id: string
  size_gb: number
  dataset: string
  concurrency: number
  wall_s: number
  summary: BenchmarkSummary
  tests: BenchmarkTest[]
}

export interface ModelConfig {
  provider_id: string
  provider_name: string
  provider_icon: string
  pricing_type: 'pay-per-use' | 'subscription' | 'local'
  model_id: string
  model_name: string
  api_model: string
  method: string
  url?: string
  api_key_env?: string
  input_price_per_1m?: number
  output_price_per_1m?: number
  size_gb?: number
  benchmarks?: string[]  // e.g. ["retain", "quality", "reflect"]
  notes?: string  // provenance for the repo (serving config, pricing source); not rendered
  price_note?: string  // pricing caveat, shown as a tooltip on the Cost cell
  reasoning_effort?: string  // "none" = thinking disabled for this model's runs
}

export interface QualityResult {
  accuracy: number
  correct: number
  total: number
  model_id: string
  provider_id: string
  // Legacy LoComo runs carried a single sample_id and none of the fields below.
  // The loader drops quality blocks without `dataset`, so scored results always
  // come from the BEAM facts-only benchmark.
  sample_id?: string
  dataset?: string
  judge_model?: string
  context_mode?: string
  hindsight_version?: string | null
  stored_fact_tokens?: number | null
  per_ability?: Record<string, { correct: number; total: number }>
  sample_ids?: string[]
}

export interface ReflectResult {
  accuracy: number
  correct: number
  total: number
  avg_latency_s: number
  model_id: string
  provider_id: string
  sample_id: string
}

export interface RerankerResult {
  reranker_id: string
  provider: string
  model: string | null
  recall_at_1: number
  recall_at_3: number
  recall_at_5: number
  mrr: number
  avg_latency_s: number
  total_questions: number
  sample_id: string
}

export interface EmbeddingsResult {
  embedding_id: string
  provider: string
  model: string
  reranker_id: string
  recall_at_1: number
  recall_at_3: number
  recall_at_5: number
  mrr: number
  avg_latency_s: number
  total_questions: number
  sample_id: string
}

// Hardware and serving config for self-hosted (GCP) rows, written by
// run_gcp_model.py so speed numbers carry their conditions.
export interface DeploymentInfo {
  machine_type?: string
  accelerator_type?: string
  accelerator_count?: number
  region?: string
  serving?: string
  container?: { image?: string; args?: string[] }
  max_model_len?: number
  max_model_len_source?: string
  reasoning_effort?: string
  llm_extra_body?: { chat_template_kwargs?: { enable_thinking?: boolean; reasoning_effort?: string } } & Record<string, unknown>
}

export interface ModelWithResult {
  config: ModelConfig
  result: BenchmarkRun | null
  qualityResult?: QualityResult | null
  // LoComo-era quality numbers (no `dataset` field), shown only on the legacy
  // retain leaderboard.
  legacyQualityResult?: QualityResult | null
  reflectResult?: ReflectResult | null
  deployment?: DeploymentInfo | null
}
