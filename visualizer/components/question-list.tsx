'use client'

import { useState, useMemo } from 'react'
import { getCategoryName } from '@/lib/utils'
import type { DetailedResult } from '@/lib/types'

interface QuestionListProps {
  questions: DetailedResult[]
}

export function QuestionList({ questions }: QuestionListProps) {
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [correctnessFilter, setCorrectnessFilter] = useState<'all' | 'correct' | 'incorrect' | 'invalid'>(
    'all'
  )

  // Get all unique categories from the questions
  const categories = useMemo(() => {
    const categorySet = new Set<string>()
    questions.forEach((q) => {
      const cat = typeof q.category === 'number' ? getCategoryName(q.category) : q.category
      categorySet.add(cat)
    })
    return Array.from(categorySet).sort()
  }, [questions])

  // Filter questions
  const filteredQuestions = useMemo(() => {
    return questions.filter((question, idx) => {
      // Correctness filter
      if (correctnessFilter !== 'all') {
        if (correctnessFilter === 'invalid' && !question.is_invalid) return false
        if (correctnessFilter === 'correct' && (!question.is_correct || question.is_invalid)) return false
        if (correctnessFilter === 'incorrect' && (question.is_correct || question.is_invalid)) return false
      }

      // Category filter
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
      <div className="bg-white border border-border rounded-lg p-6 shadow-sm mb-6">
        <h3 className="text-sm font-semibold text-foreground mb-4">Filter Questions</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Correctness */}
          <div>
            <label htmlFor="correctness" className="block text-xs font-medium text-muted-foreground mb-2">
              Correctness
            </label>
            <select
              id="correctness"
              value={correctnessFilter}
              onChange={(e) => setCorrectnessFilter(e.target.value as any)}
              className="w-full px-3 py-2 border border-border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All</option>
              <option value="correct">Correct</option>
              <option value="incorrect">Incorrect</option>
              <option value="invalid">Invalid</option>
            </select>
          </div>

          {/* Category */}
          {categories.length > 0 && (
            <div>
              <label htmlFor="category" className="block text-xs font-medium text-muted-foreground mb-2">
                Category
              </label>
              <select
                id="category"
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="w-full px-3 py-2 border border-border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Categories</option>
                {categories.map((cat) => (
                  <option key={cat} value={cat}>
                    {cat}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
        <p className="text-xs text-muted-foreground mt-3">
          Showing {filteredQuestions.length} of {questions.length} questions
        </p>
      </div>

      {/* Question List */}
      <div className="space-y-4">
        {filteredQuestions.map((result, qIdx) => {
          const isInvalid = result.is_invalid
          const isCorrect = result.is_correct
          const category = typeof result.category === 'number' ? getCategoryName(result.category) : result.category
          const icon = isInvalid ? '⚠️' : isCorrect ? '✅' : '❌'
          const borderClass = isInvalid
            ? 'border-l-4 border-yellow-500'
            : isCorrect
            ? 'border-l-4 border-green-600'
            : 'border-l-4 border-red-600'

          // Find original index
          const originalIdx = questions.findIndex((q) => q.question === result.question)

          return (
            <div key={originalIdx} className={`bg-white border border-border rounded-lg p-6 shadow-sm ${borderClass}`}>
              <div className="mb-4">
                <p className="text-lg font-semibold text-foreground">
                  {icon} Question {originalIdx + 1}
                </p>
                <p className="text-sm text-muted-foreground">Category: {category}</p>
              </div>

              <div className="mb-4">
                <p className="text-sm font-medium text-foreground mb-1">Question:</p>
                <p className="text-foreground">{result.question}</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <p className="text-sm font-medium text-foreground mb-2">✓ Correct Answer</p>
                  <div className="bg-green-50 border border-green-200 rounded-md p-3 text-foreground text-sm">
                    {result.correct_answer}
                  </div>
                </div>

                <div>
                  <p className="text-sm font-medium text-foreground mb-2">
                    {isCorrect ? '✓' : '✗'} Predicted Answer
                  </p>
                  <div
                    className={`border rounded-md p-3 text-foreground text-sm ${
                      isCorrect ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
                    }`}
                  >
                    {result.predicted_answer}
                  </div>
                </div>
              </div>

              <details className="border border-border rounded-md p-4 bg-muted/30">
                <summary className="cursor-pointer font-medium text-foreground hover:text-primary py-2">
                  📝 Show Reasoning & Retrieved Memories
                </summary>
                <div className="mt-3 space-y-4">
                  <div>
                    <p className="text-sm font-medium text-foreground mb-2">System Reasoning:</p>
                    <pre className="bg-slate-900 text-slate-100 p-3 rounded-md overflow-x-auto text-sm whitespace-pre-wrap">
                      {result.reasoning || 'N/A'}
                    </pre>
                  </div>

                  {result.correctness_reasoning && (
                    <div>
                      <p className="text-sm font-medium text-foreground mb-2">Judge Reasoning:</p>
                      <pre className="bg-slate-900 text-slate-100 p-3 rounded-md overflow-x-auto text-sm whitespace-pre-wrap">
                        {result.correctness_reasoning}
                      </pre>
                    </div>
                  )}

                  <div>
                    <p className="text-sm font-medium text-foreground mb-2">
                      Retrieved Memories ({result.retrieved_memories?.length || 0}):
                    </p>
                    {result.retrieved_memories && result.retrieved_memories.length > 0 ? (
                      result.retrieved_memories.map((mem, i) => (
                        <div key={i} className="bg-muted/50 border border-border rounded-md p-3 mb-2">
                          <p className="text-xs text-muted-foreground mb-1">
                            #{i + 1} • Type: {mem.fact_type?.toUpperCase() || 'N/A'}
                            {mem.occurred_start && ` • Date: ${mem.occurred_start.slice(0, 10)}`}
                          </p>
                          <p className="text-sm text-foreground">{mem.text}</p>
                        </div>
                      ))
                    ) : (
                      <p className="text-sm text-muted-foreground">No memories retrieved</p>
                    )}
                  </div>
                </div>
              </details>
            </div>
          )
        })}
      </div>

      {filteredQuestions.length === 0 && (
        <div className="text-center py-12 bg-white border border-border rounded-lg">
          <p className="text-muted-foreground">No questions match your filters</p>
        </div>
      )}
    </div>
  )
}
