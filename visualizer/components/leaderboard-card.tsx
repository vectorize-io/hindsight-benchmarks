'use client'

import { useRef, useState, useEffect } from 'react'
import Link from 'next/link'
import Image from 'next/image'
import React from 'react'

export type LeaderboardEntry = {
  href: string
  title: string
  description: React.ReactNode
  modelCount: number
  winner?: { name: string; providerIcon: string; providerName: string } | null
}

function LeaderboardRow({ entry, animate, delay }: { entry: LeaderboardEntry; animate: boolean; delay: number }) {
  return (
    <Link href={entry.href} className="group/row block">
      <div
        className="flex items-center gap-6 px-5 py-4 rounded-lg transition-all duration-300 hover:bg-secondary/40 cursor-pointer"
        style={{
          opacity: animate ? 1 : 0,
          transform: animate ? 'translateX(0)' : 'translateX(-12px)',
          transition: `opacity 0.5s ease-out ${delay}ms, transform 0.5s ease-out ${delay}ms, background-color 0.2s`,
        }}
      >
        {/* Title + description */}
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-foreground mb-0.5">{entry.title}</h3>
          <p className="text-xs text-muted-foreground leading-relaxed">{entry.description}</p>
        </div>

        {/* Model count */}
        <div className="shrink-0 text-center">
          <div className="text-lg font-bold tabular-nums gradient-primary-text">{entry.modelCount}</div>
          <div className="text-[10px] text-muted-foreground uppercase tracking-wide">Models</div>
        </div>

        {/* Winner */}
        {entry.winner && (
          <div className="shrink-0 flex items-center gap-1.5">
            {entry.winner.providerIcon && (
              <Image src={entry.winner.providerIcon} alt={entry.winner.providerName} width={14} height={14} className="rounded-sm opacity-80" />
            )}
            <span className="text-xs font-medium text-foreground bg-secondary px-2 py-0.5 rounded-full border border-border whitespace-nowrap">
              {entry.winner.name}
            </span>
          </div>
        )}

        {/* Arrow */}
        <svg
          className="w-4 h-4 shrink-0 text-muted-foreground transition-transform group-hover/row:translate-x-0.5 group-hover/row:text-foreground"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </div>
    </Link>
  )
}

export function LeaderboardBlock({ entries }: { entries: LeaderboardEntry[] }) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true)
          observer.disconnect()
        }
      },
      { threshold: 0.15 }
    )
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return (
    <div
      ref={ref}
      className="transition-all duration-700 ease-out"
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(24px)',
      }}
    >
      <div className="relative">
        {/* Glow */}
        <div className="absolute -inset-px rounded-xl opacity-30 blur-xl" style={{
          background: 'linear-gradient(135deg, #0074d9 0%, #009296 50%, #00d4aa 100%)',
        }} />

        <div className="relative bg-card border border-border rounded-xl p-8 overflow-hidden">
          {/* Subtle corner glow */}
          <div className="absolute top-0 right-0 w-64 h-64 opacity-[0.03]" style={{
            background: 'radial-gradient(circle at top right, #00d4aa, transparent 70%)',
          }} />

          {/* Header */}
          <div className="mb-6">
            <h2 className="text-lg font-heading font-bold text-foreground mb-2">
              Model Leaderboard
            </h2>
            <p className="text-sm text-muted-foreground">
              Ranked models across all Hindsight operations — find the best LLM, reranker, and embedding model for your setup.
            </p>
          </div>

          {/* Rows */}
          <div className="divide-y divide-border/50">
            {entries.map((entry, i) => (
              <LeaderboardRow
                key={entry.href}
                entry={entry}
                animate={visible}
                delay={i * 100}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
