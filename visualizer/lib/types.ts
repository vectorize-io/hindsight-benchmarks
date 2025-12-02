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
