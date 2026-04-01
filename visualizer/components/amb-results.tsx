'use client'

import { useEffect, useState } from 'react'

const AMB_RESULTS = [
  { benchmark: 'LongMemEval', split: 'S', score: 94.6 },
  { benchmark: 'LoComo', split: '10', score: 92.0 },
  { benchmark: 'PersonaMem', split: '32K', score: 86.6 },
  { benchmark: 'BEAM', split: '100K', score: 75.0 },
  { benchmark: 'BEAM', split: '1M', score: 73.9 },
  { benchmark: 'LifeBench', split: 'EN', score: 71.5 },
  { benchmark: 'BEAM', split: '500K', score: 71.1 },
  { benchmark: 'BEAM', split: '10M', score: 64.1 },
]

function ScoreBar({ benchmark, split, score, index, animate }: {
  benchmark: string
  split: string
  score: number
  index: number
  animate: boolean
}) {
  const width = animate ? score : 0
  const delay = index * 80

  return (
    <div
      className="group relative"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex items-center gap-4">
        {/* Label */}
        <div className="w-[140px] shrink-0 text-right">
          <span className="text-sm font-medium text-foreground">{benchmark}</span>
          <span className="text-xs text-muted-foreground ml-1.5 font-mono">{split}</span>
        </div>

        {/* Bar track */}
        <div className="flex-1 h-8 bg-secondary/50 rounded-sm overflow-hidden relative">
          {/* Filled bar */}
          <div
            className="h-full rounded-sm relative overflow-hidden"
            style={{
              width: `${width}%`,
              transition: `width 1.2s cubic-bezier(0.16, 1, 0.3, 1) ${delay}ms`,
            }}
          >
            {/* Gradient fill */}
            <div
              className="absolute inset-0"
              style={{
                background: score >= 90
                  ? 'linear-gradient(90deg, #0074d9 0%, #00d4aa 100%)'
                  : score >= 80
                    ? 'linear-gradient(90deg, #0074d9 0%, #009296 100%)'
                    : 'linear-gradient(90deg, #0074d9 0%, #3396e8 100%)',
              }}
            />
            {/* Shimmer overlay */}
            <div
              className="absolute inset-0 opacity-20"
              style={{
                background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.3) 50%, transparent 100%)',
                animation: animate ? `shimmer 2s ease-in-out ${delay + 1200}ms` : 'none',
              }}
            />
          </div>
        </div>

        {/* Score */}
        <div className="w-[60px] shrink-0">
          <span
            className="text-lg font-heading font-bold tabular-nums"
            style={{
              color: score >= 90 ? '#00d4aa' : score >= 80 ? '#009296' : '#3396e8',
            }}
          >
            {score}%
          </span>
        </div>
      </div>
    </div>
  )
}

export function AMBResults() {
  const [animate, setAnimate] = useState(false)

  useEffect(() => {
    const timer = requestAnimationFrame(() => setAnimate(true))
    return () => cancelAnimationFrame(timer)
  }, [])

  return (
    <div className="relative">
      {/* Glow effect behind the card */}
      <div className="absolute -inset-px rounded-xl opacity-30 blur-xl" style={{
        background: 'linear-gradient(135deg, #0074d9 0%, #009296 50%, #00d4aa 100%)',
      }} />

      <div className="relative bg-card border border-border rounded-xl p-8 overflow-hidden">
        {/* Subtle corner glow */}
        <div className="absolute top-0 right-0 w-64 h-64 opacity-[0.03]" style={{
          background: 'radial-gradient(circle at top right, #00d4aa, transparent 70%)',
        }} />

        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-lg font-heading font-bold text-foreground">
                Agent Memory Benchmark
              </h2>
              <span className="text-[10px] font-semibold tracking-widest uppercase px-2 py-0.5 rounded-full border gradient-primary text-white">
                #1
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              Hindsight scores across all AMB datasets — leading every benchmark with verified results.
            </p>
          </div>
        </div>

        {/* Bars */}
        <div className="space-y-3">
          {AMB_RESULTS.map((r, i) => (
            <ScoreBar
              key={`${r.benchmark}-${r.split}`}
              benchmark={r.benchmark}
              split={r.split}
              score={r.score}
              index={i}
              animate={animate}
            />
          ))}
        </div>

        {/* Footer */}
        <div className="mt-8 pt-5 border-t border-border flex items-center justify-end">
          <a
            href="https://agentmemorybenchmark.ai"
            target="_blank"
            rel="noopener noreferrer"
            className="group/link flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
          >
            View full comparison on agentmemorybenchmark.ai
            <svg
              className="w-3.5 h-3.5 transition-transform group-hover/link:translate-x-0.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
        </div>
      </div>
    </div>
  )
}
