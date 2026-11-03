import RAW from './models.json'

// Our models and their scores, from each model's Hugging Face card: `python benchmark-runner/pull_hf_cards.py`.
// Typed here, since a card with no results yet leaves the JSON's lists empty.
export type ModelRow = { model: string; provider: string; openWeights?: boolean; tag?: 'hindsight'; size?: string; scores: Record<string, number | undefined> }
export type Family = {
  id: string
  title: string
  model: string
  short: string
  card: string
  facts: [string, string][]
  benchmarks: { id: string; name: string; metric: string; href?: string | null; about: string; section: string; plain?: boolean; lower?: boolean }[]
  rows: ModelRow[]
}
export const FAMILIES = RAW.families as unknown as Family[]
