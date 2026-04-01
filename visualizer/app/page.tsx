import React from 'react'
import { loadLeaderboardData, getLeaderboardStats } from '@/lib/leaderboard'
import { loadRerankerData, getRerankerStats } from '@/lib/reranker'
import { loadEmbeddingsData, getEmbeddingsStats } from '@/lib/embeddings'
import { AMBResults } from '@/components/amb-results'
import { LeaderboardBlock } from '@/components/leaderboard-card'

export default function Home() {
  const leaderboardModels = loadLeaderboardData()
  const leaderboardStats = getLeaderboardStats(leaderboardModels)
  const rerankers = loadRerankerData()
  const rerankerStats = getRerankerStats(rerankers)
  const embeddingModels = loadEmbeddingsData()
  const embeddingsStats = getEmbeddingsStats(embeddingModels)

  const leaderboardEntries = [
    {
      href: '/leaderboard/retain',
      title: 'Retain',
      description: <>Ranked LLMs for <span className="gradient-primary-text font-semibold">retain()</span> — fact extraction quality, speed, cost, and reliability.</>,
      modelCount: leaderboardStats.viableModels,
      winner: leaderboardStats.topRetainModel,
    },
    {
      href: '/leaderboard/reflect',
      title: 'Reflect',
      description: <>Ranked LLMs for the <span className="gradient-primary-text font-semibold">reflect()</span> operation.</>,
      modelCount: leaderboardStats.viableReflectModels,
      winner: leaderboardStats.topReflectModel,
    },
    {
      href: '/leaderboard/reranker',
      title: 'Reranker',
      description: <>Ranked rerankers for <span className="gradient-primary-text font-semibold">recall()</span> — which reranker surfaces the most relevant facts first.</>,
      modelCount: rerankerStats.count,
      winner: rerankerStats.topReranker ? {
        name: rerankerStats.topReranker.name,
        providerIcon: rerankerStats.topReranker.providerIcon,
        providerName: rerankerStats.topReranker.providerName,
      } : null,
    },
    {
      href: '/leaderboard/embeddings',
      title: 'Embeddings',
      description: <>Ranked embedding models — affects both <span className="gradient-primary-text font-semibold">retain()</span> storage and <span className="gradient-primary-text font-semibold">recall()</span> retrieval quality.</>,
      modelCount: embeddingsStats.count,
      winner: embeddingsStats.topEmbedding ? {
        name: embeddingsStats.topEmbedding.name,
        providerIcon: embeddingsStats.topEmbedding.providerIcon,
        providerName: embeddingsStats.topEmbedding.providerName,
      } : null,
    },
  ]

  const ambResults = [
    { benchmark: 'LongMemEval', split: 'S', score: 94.6 },
    { benchmark: 'LoComo', split: '10', score: 92.0 },
    { benchmark: 'PersonaMem', split: '32K', score: 86.6 },
    { benchmark: 'BEAM', split: '100K', score: 75.0 },
    { benchmark: 'BEAM', split: '1M', score: 73.9 },
    { benchmark: 'LifeBench', split: 'EN', score: 71.5 },
    { benchmark: 'BEAM', split: '500K', score: 71.1 },
    { benchmark: 'BEAM', split: '10M', score: 64.1 },
  ]

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'WebPage',
    name: 'Hindsight Benchmarks — #1 on Agent Memory Benchmark',
    description:
      'Hindsight leads every dataset on the Agent Memory Benchmark. Explore results across LongMemEval, LoComo, BEAM, LifeBench, and PersonaMem.',
    url: 'https://benchmarks.hindsight.vectorize.io',
    mainEntity: {
      '@type': 'Dataset',
      name: 'Hindsight Agent Memory Benchmark Results',
      description:
        'Benchmark scores for Hindsight across all Agent Memory Benchmark datasets including LongMemEval, LoComo, BEAM, LifeBench, and PersonaMem.',
      creator: {
        '@type': 'Organization',
        name: 'Vectorize',
        url: 'https://vectorize.io',
      },
      distribution: ambResults.map((r) => ({
        '@type': 'DataDownload',
        name: `${r.benchmark} ${r.split}`,
        description: `Hindsight scored ${r.score}% on ${r.benchmark} (${r.split} split)`,
      })),
    },
  }

  return (
    <main className="min-h-screen">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />
      {/* Hero + AMB section */}
      <div className="container mx-auto max-w-5xl px-4 pt-20 pb-24">
        <div className="mb-10">
          <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-4">
            Agent Memory Benchmark
          </p>
          <h1 className="text-4xl font-heading font-bold mb-3">
            <span className="gradient-primary-text">Hindsight</span>
            <span className="text-foreground"> is #1</span>
          </h1>
          <p className="text-muted-foreground max-w-xl">
            Leading every dataset on the Agent Memory Benchmark — the industry standard for evaluating memory and retrieval systems.
          </p>
        </div>
        <AMBResults />
      </div>

      {/* Divider */}
      <div className="border-t border-border" />

      {/* Leaderboard section — distinct background */}
      <div className="bg-[hsl(240_5%_5.5%)]">
        <div className="container mx-auto max-w-5xl px-4 py-24">
          <div className="mb-10">
            <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-4">
              Choose your models
            </p>
            <h2 className="text-3xl font-heading font-bold mb-3 text-foreground">
              Model Leaderboard
            </h2>
            <p className="text-muted-foreground max-w-xl">
              Find the best LLM, reranker, and embedding model for your Hindsight setup — ranked by quality, speed, and cost.
            </p>
          </div>
          <LeaderboardBlock entries={leaderboardEntries} />
        </div>
      </div>
    </main>
  )
}
