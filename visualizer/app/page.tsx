import Link from 'next/link'
import Image from 'next/image'
import React from 'react'
import { loadLongMemEval, loadLoComo } from '@/lib/data'
import { loadLeaderboardData, getLeaderboardStats } from '@/lib/leaderboard'
import { loadRerankerData, getRerankerStats } from '@/lib/reranker'
import { loadEmbeddingsData, getEmbeddingsStats } from '@/lib/embeddings'

function StatBlock({ value, label, highlight }: { value: string | number; label: string; highlight?: boolean }) {
  return (
    <div>
      <div className={`text-xl font-bold tabular-nums ${highlight ? 'gradient-primary-text' : 'text-foreground'}`}>
        {value}
      </div>
      <div className="text-xs text-muted-foreground uppercase tracking-wide mt-0.5">{label}</div>
    </div>
  )
}

function WinnerBlock({ name, providerIcon, providerName }: { name: string; providerIcon: string; providerName: string }) {
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1">
        {providerIcon && (
          <Image src={providerIcon} alt={providerName} width={14} height={14} className="rounded-sm opacity-80" />
        )}
        <span className="text-xs font-medium text-foreground bg-secondary px-2 py-0.5 rounded-full border border-border">
          {name}
        </span>
      </div>
      <div className="text-xs text-muted-foreground uppercase tracking-wide mt-0.5">Top model</div>
    </div>
  )
}

function BenchmarkCard({
  href,
  tag,
  title,
  description,
  stats,
  winner,
  cta,
}: {
  href: string
  tag: string
  title: string
  description: React.ReactNode
  stats: { value: string | number; label: string; highlight?: boolean }[]
  winner?: { name: string; providerIcon: string; providerName: string } | null
  cta: string
}) {
  return (
    <Link href={href} className="group block">
      <div className="h-full bg-card border border-border rounded-lg p-6 transition-colors duration-200 hover:bg-secondary/40 cursor-pointer">
        <div className="mb-4">
          <span className="text-[10px] font-semibold tracking-widest uppercase text-muted-foreground">
            {tag}
          </span>
        </div>

        <h3 className="text-sm font-semibold tracking-widest uppercase text-foreground mb-1">
          {title}
        </h3>
        <p className="text-sm text-muted-foreground mb-6">{description}</p>

        <div className="flex items-center gap-8 mb-6">
          {stats.map((s, i) => (
            <StatBlock key={i} value={s.value} label={s.label} highlight={s.highlight} />
          ))}
          {winner && <WinnerBlock name={winner.name} providerIcon={winner.providerIcon} providerName={winner.providerName} />}
        </div>

        <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground group-hover:text-foreground transition-colors">
          {cta}
          <svg
            className="w-3.5 h-3.5 transition-transform group-hover:translate-x-0.5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </div>
      </div>
    </Link>
  )
}

export default function Home() {
  const longmemeval = loadLongMemEval()
  const locomo = loadLoComo()
  const leaderboardModels = loadLeaderboardData()
  const leaderboardStats = getLeaderboardStats(leaderboardModels)
  const rerankers = loadRerankerData()
  const rerankerStats = getRerankerStats(rerankers)
  const embeddingModels = loadEmbeddingsData()
  const embeddingsStats = getEmbeddingsStats(embeddingModels)

  return (
    <main className="min-h-screen">
      <div className="container mx-auto max-w-5xl px-4 py-20">
        <div className="mb-16">
          <h1 className="text-4xl font-heading font-bold mb-3">
            <span className="gradient-primary-text">Hindsight</span>
            <span className="text-foreground"> Benchmarks</span>
          </h1>
          <p className="text-muted-foreground max-w-xl">
            Industry benchmarks and model leaderboard for Hindsight — the memory layer for AI agents.
          </p>
        </div>

        <div className="space-y-10">
          {/* Industry Benchmarks */}
          <section>
            <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-4">
              Industry Benchmarks
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              {longmemeval ? (
                <BenchmarkCard
                  href="/longmemeval"
                  tag="Benchmark"
                  title="LongMemEval"
                  description="Comprehensive long-term memory evaluation across 500 items."
                  stats={[
                    { value: `${longmemeval.overall_accuracy.toFixed(1)}%`, label: 'Accuracy', highlight: true },
                    { value: longmemeval.num_items, label: 'Items' },
                    { value: longmemeval.total_questions, label: 'Questions' },
                  ]}
                  cta="View results"
                />
              ) : (
                <div className="bg-card p-6 text-sm text-muted-foreground">No LongMemEval data</div>
              )}

              {locomo ? (
                <BenchmarkCard
                  href="/locomo"
                  tag="Benchmark"
                  title="LoComo"
                  description="Long conversation memory benchmark across real-world dialogues."
                  stats={[
                    { value: `${locomo.overall_accuracy.toFixed(1)}%`, label: 'Accuracy', highlight: true },
                    { value: locomo.num_items, label: 'Convos' },
                    { value: locomo.total_questions, label: 'Questions' },
                  ]}
                  cta="View results"
                />
              ) : (
                <div className="bg-card p-6 text-sm text-muted-foreground">No LoComo data</div>
              )}
            </div>
          </section>

          {/* Model Leaderboards */}
          <section>
            <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-4">
              Model Leaderboards
            </p>
            <div className="grid md:grid-cols-2 gap-4">
              <BenchmarkCard
                href="/leaderboard/retain"
                tag="Leaderboard"
                title="Retain Leaderboard"
                description={<>Ranked LLMs for <span className="gradient-primary-text font-semibold">retain()</span> and <span className="gradient-primary-text font-semibold">observation consolidation</span> — fact extraction quality, speed, cost, and reliability.</>}
                stats={[
                  { value: leaderboardStats.viableModels, label: 'Models', highlight: true },
                ]}
                winner={leaderboardStats.topRetainModel}
                cta="View leaderboard"
              />
              <BenchmarkCard
                href="/leaderboard/reflect"
                tag="Leaderboard"
                title="Reflect Leaderboard"
                description={<>Ranked LLMs for the <span className="gradient-primary-text font-semibold">reflect</span> operation in Hindsight.</>}
                stats={[
                  { value: leaderboardStats.viableReflectModels, label: 'Models', highlight: true },
                ]}
                winner={leaderboardStats.topReflectModel}
                cta="View leaderboard"
              />
              <BenchmarkCard
                href="/leaderboard/reranker"
                tag="Leaderboard"
                title="Reranker Leaderboard"
                description={<>Ranked rerankers for <span className="gradient-primary-text font-semibold">recall()</span> — which reranker surfaces the most relevant facts first.</>}
                stats={[
                  { value: rerankerStats.count, label: 'Rerankers', highlight: true },
                ]}
                winner={rerankerStats.topReranker ? {
                  name: rerankerStats.topReranker.name,
                  providerIcon: rerankerStats.topReranker.providerIcon,
                  providerName: rerankerStats.topReranker.providerName,
                } : null}
                cta="View leaderboard"
              />
              <BenchmarkCard
                href="/leaderboard/embeddings"
                tag="Leaderboard"
                title="Embeddings Leaderboard"
                description={<>Ranked embedding models for <span className="gradient-primary-text font-semibold">recall()</span> — which embedding retrieves the most relevant facts.</>}
                stats={[
                  { value: embeddingsStats.count, label: 'Models', highlight: true },
                ]}
                winner={embeddingsStats.topEmbedding ? {
                  name: embeddingsStats.topEmbedding.name,
                  providerIcon: embeddingsStats.topEmbedding.providerIcon,
                  providerName: embeddingsStats.topEmbedding.providerName,
                } : null}
                cta="View leaderboard"
              />
            </div>
          </section>
        </div>
      </div>
    </main>
  )
}
