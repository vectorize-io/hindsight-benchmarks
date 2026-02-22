'use client'

import { useState, useMemo } from 'react'
import { getCategoryName } from '@/lib/utils'
import type { DetailedResult } from '@/lib/types'

interface QuestionListProps {
  questions: DetailedResult[]
}

export function QuestionList({ questions }: QuestionListProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [correctnessFilter, setCorrectnessFilter] = useState<'all' | 'correct' | 'incorrect' | 'invalid'>('all')

  const categories = useMemo(() => {
    const categorySet = new Set<string>()
    questions.forEach((q) => {
      const cat = typeof q.category === 'number' ? getCategoryName(q.category) : q.category
      categorySet.add(cat)
    })
    return Array.from(categorySet).sort()
  }, [questions])

  const filteredQuestions = useMemo(() => {
    return questions.filter((question) => {
      if (correctnessFilter !== 'all') {
        if (correctnessFilter === 'invalid' && !question.is_invalid) return false
        if (correctnessFilter === 'correct' && (!question.is_correct || question.is_invalid)) return false
        if (correctnessFilter === 'incorrect' && (question.is_correct || question.is_invalid)) return false
      }
      if (selectedCategory !== 'all') {
        const cat = typeof question.category === 'number' ? getCategoryName(question.category) : question.category
        if (cat !== selectedCategory) return false
      }
      return true
    })
  }, [questions, correctnessFilter, selectedCategory])

  return (
    <div>
      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <select
          value={correctnessFilter}
          onChange={(e) => setCorrectnessFilter(e.target.value as any)}
          className="px-3 py-1.5 border border-border rounded-md text-xs bg-secondary text-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
        >
          <option value="all">All results</option>
          <option value="correct">Correct only</option>
          <option value="incorrect">Incorrect only</option>
          <option value="invalid">Invalid only</option>
        </select>

        {categories.length > 0 && (
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-1.5 border border-border rounded-md text-xs bg-secondary text-foreground focus:outline-none focus:ring-1 focus:ring-primary/50"
          >
            <option value="all">All categories</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
        )}

        <span className="text-xs text-muted-foreground ml-auto">
          {filteredQuestions.length} / {questions.length}
        </span>
      </div>

      {filteredQuestions.length === 0 ? (
        <div className="text-center py-12 bg-card border border-border rounded-lg">
          <p className="text-muted-foreground text-sm">No questions match your filters</p>
        </div>
      ) : (
        <div className="space-y-2">
          {filteredQuestions.map((result) => {
            const isInvalid = result.is_invalid
            const isCorrect = result.is_correct
            const category = typeof result.category === 'number' ? getCategoryName(result.category) : result.category
            const originalIdx = questions.findIndex((q) => q.question === result.question)

            const accentClass = isInvalid
              ? 'border-l-amber-500'
              : isCorrect
              ? 'border-l-emerald-500'
              : 'border-l-red-500'
            const statusLabel = isInvalid ? 'Invalid' : isCorrect ? 'Correct' : 'Incorrect'
            const statusColor = isInvalid ? 'text-amber-400' : isCorrect ? 'text-emerald-400' : 'text-red-400'

            return (
              <details key={originalIdx} className={`group bg-card border border-border border-l-2 ${accentClass} rounded-lg`}>
                <summary className="flex items-center justify-between px-5 py-4 cursor-pointer list-none select-none">
                  <div className="flex items-center gap-4 min-w-0">
                    <span className="text-xs font-mono text-muted-foreground shrink-0">Q{originalIdx + 1}</span>
                    <span className="text-sm text-foreground truncate">{result.question}</span>
                  </div>
                  <div className="flex items-center gap-3 ml-4 shrink-0">
                    <span className={`text-xs font-semibold ${statusColor}`}>{statusLabel}</span>
                    <span className="text-xs text-muted-foreground uppercase tracking-wide hidden sm:inline">{category}</span>
                    <svg className="w-4 h-4 text-muted-foreground transition-transform group-open:rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </summary>

                <div className="px-5 pb-5 border-t border-border pt-4 space-y-4">
                  <div>
                    <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-1">Question</p>
                    <p className="text-sm text-foreground">{result.question}</p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-px bg-border rounded-md overflow-hidden">
                    <div className="bg-card px-4 py-3">
                      <p className="text-[11px] font-semibold tracking-widest uppercase text-emerald-500 mb-1">Expected</p>
                      <p className="text-sm text-foreground">{result.correct_answer}</p>
                    </div>
                    <div className="bg-card px-4 py-3">
                      <p className={`text-[11px] font-semibold tracking-widest uppercase mb-1 ${isCorrect ? 'text-emerald-500' : isInvalid ? 'text-amber-500' : 'text-red-500'}`}>
                        Predicted
                      </p>
                      <p className="text-sm text-foreground">{result.predicted_answer}</p>
                    </div>
                  </div>

                  {result.reasoning && (
                    <div>
                      <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-1">Reasoning</p>
                      <pre className="text-xs text-muted-foreground bg-background border border-border rounded-md p-3 overflow-x-auto whitespace-pre-wrap font-mono">{result.reasoning}</pre>
                    </div>
                  )}

                  {result.correctness_reasoning && (
                    <div>
                      <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-1">Judge Reasoning</p>
                      <pre className="text-xs text-muted-foreground bg-background border border-border rounded-md p-3 overflow-x-auto whitespace-pre-wrap font-mono">{result.correctness_reasoning}</pre>
                    </div>
                  )}

                  {result.retrieved_memories && result.retrieved_memories.length > 0 && (
                    <div>
                      <p className="text-[11px] font-semibold tracking-widest uppercase text-muted-foreground mb-2">
                        Retrieved Memories ({result.retrieved_memories.length})
                      </p>
                      <div className="space-y-1">
                        {result.retrieved_memories.map((mem, i) => (
                          <div key={i} className="bg-secondary/40 border border-border rounded-md px-3 py-2">
                            <p className="text-[10px] font-semibold tracking-widest uppercase text-muted-foreground mb-0.5">
                              {mem.fact_type?.toUpperCase() || 'FACT'}
                              {mem.occurred_start && ` · ${mem.occurred_start.slice(0, 10)}`}
                            </p>
                            <p className="text-sm text-foreground">{mem.text}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </details>
            )
          })}
        </div>
      )}
    </div>
  )
}
