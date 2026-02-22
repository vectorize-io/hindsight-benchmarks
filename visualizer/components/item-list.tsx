'use client'

import { useState, useMemo } from 'react'
import Link from 'next/link'
import type { ItemResult } from '@/lib/types'
import { Card, CardContent } from '@/components/ui/card'

interface ItemListProps {
  items: ItemResult[]
  basePath: string
  showCategories?: boolean
}

export function ItemList({ items, basePath, showCategories = false }: ItemListProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [correctnessFilter, setCorrectnessFilter] = useState<'all' | 'correct' | 'incorrect' | 'invalid'>(
    'all'
  )

  const categories = useMemo(() => {
    if (!showCategories) return []
    const categorySet = new Set<string>()
    items.forEach((item) => {
      item.metrics.detailed_results?.forEach((result) => {
        const cat = typeof result.category === 'number' ? result.category.toString() : result.category
        categorySet.add(cat)
      })
    })
    return Array.from(categorySet).sort()
  }, [items, showCategories])

  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      if (searchTerm && !item.item_id.toLowerCase().includes(searchTerm.toLowerCase())) {
        return false
      }

      if (correctnessFilter !== 'all') {
        const accuracy = item.metrics.accuracy
        const hasInvalid = item.metrics.invalid > 0
        if (correctnessFilter === 'invalid' && !hasInvalid) return false
        if (correctnessFilter === 'correct' && accuracy < 100) return false
        if (correctnessFilter === 'incorrect' && (accuracy >= 100 || hasInvalid)) return false
      }

      if (selectedCategory !== 'all') {
        const hasCategory = item.metrics.detailed_results?.some((result) => {
          const cat = typeof result.category === 'number' ? result.category.toString() : result.category
          return cat === selectedCategory
        })
        if (!hasCategory) return false
      }

      return true
    })
  }, [items, searchTerm, correctnessFilter, selectedCategory])

  return (
    <div>
      {/* Filters */}
      <div className="bg-card border border-border rounded-lg p-5 mb-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div>
            <label htmlFor="search" className="block text-xs font-medium text-muted-foreground mb-2">
              Search by ID
            </label>
            <input
              id="search"
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search..."
              className="w-full px-3 py-2 border border-border rounded-md text-sm bg-secondary text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50"
            />
          </div>

          <div>
            <label htmlFor="correctness" className="block text-xs font-medium text-muted-foreground mb-2">
              Correctness
            </label>
            <select
              id="correctness"
              value={correctnessFilter}
              onChange={(e) => setCorrectnessFilter(e.target.value as any)}
              className="w-full px-3 py-2 border border-border rounded-md text-sm bg-secondary text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50"
            >
              <option value="all">All</option>
              <option value="correct">Correct (100%)</option>
              <option value="incorrect">Incorrect</option>
              <option value="invalid">Has Invalid</option>
            </select>
          </div>

          {showCategories && categories.length > 0 && (
            <div>
              <label htmlFor="category" className="block text-xs font-medium text-muted-foreground mb-2">
                Category
              </label>
              <select
                id="category"
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="w-full px-3 py-2 border border-border rounded-md text-sm bg-secondary text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/50"
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
          Showing {filteredItems.length} of {items.length} items
        </p>
      </div>

      {/* Item Grid */}
      {filteredItems.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-border rounded-lg overflow-hidden">
          {filteredItems.map((item) => {
            const accuracy = item.metrics.accuracy
            const accentClass =
              accuracy >= 70 ? 'border-l-emerald-500' : accuracy >= 50 ? 'border-l-amber-500' : 'border-l-red-500'

            const originalIdx = items.findIndex((i) => i.item_id === item.item_id)

            return (
              <Link
                key={item.item_id}
                href={`${basePath}/${originalIdx}`}
                className={`group bg-card border-l-2 ${accentClass} px-5 py-4 hover:bg-secondary/30 transition-colors`}
              >
                <p className="text-xs font-medium text-muted-foreground mb-2 truncate">{item.item_id}</p>
                <div className="flex items-baseline gap-3">
                  <span className="text-xl font-bold text-foreground tabular-nums">{accuracy.toFixed(1)}%</span>
                  <span className="text-xs text-muted-foreground">{item.metrics.correct}/{item.metrics.total} correct</span>
                </div>
              </Link>
            )
          })}
        </div>
      ) : (
        <div className="text-center py-12 bg-card border border-border rounded-lg">
          <p className="text-muted-foreground text-sm">No items match your filters</p>
        </div>
      )}
    </div>
  )
}
