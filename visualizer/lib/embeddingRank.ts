import type { EmbeddingRow } from './embeddings'

// Rank by the displayed total; rows that tie at one decimal rank by retrieval quality (MRR).
export function compareEmbeddingRows(a: EmbeddingRow, b: EmbeddingRow): number {
  return b.score.totalScore - a.score.totalScore || b.mrr - a.mrr
}
