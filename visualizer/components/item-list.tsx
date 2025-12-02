'use client'

import { useState, useMemo } from 'react'
import Link from 'next/link'
import type { ItemResult } from '@/lib/types'

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

  // Get all unique categories from the data
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

  // Filter items
  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      // Search filter
      if (searchTerm && !item.item_id.toLowerCase().includes(searchTerm.toLowerCase())) {
        return false
      }

      // Correctness filter
      if (correctnessFilter !== 'all') {
        const accuracy = item.metrics.accuracy
        const hasInvalid = item.metrics.invalid > 0
        if (correctnessFilter === 'invalid' && !hasInvalid) return false
        if (correctnessFilter === 'correct' && accuracy < 100) return false
        if (correctnessFilter === 'incorrect' && (accuracy >= 100 || hasInvalid)) return false
      }

      // Category filter
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
      <div className="bg-white border border-border rounded-lg p-6 shadow-sm mb-6">
        <h3 className="text-sm font-semibold text-foreground mb-4">Filters</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Search */}
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
              className="w-full px-3 py-2 border border-border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

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
              <option value="correct">Correct (100%)</option>
              <option value="incorrect">Incorrect</option>
              <option value="invalid">Has Invalid</option>
            </select>
          </div>

          {/* Category */}
          {showCategories && categories.length > 0 && (
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
          Showing {filteredItems.length} of {items.length} items
        </p>
      </div>

      {/* Item Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filteredItems.map((item, idx) => {
          const accuracy = item.metrics.accuracy
          const color = accuracy >= 70 ? '🟢' : accuracy >= 50 ? '🟡' : '🔴'
          const borderClass =
            accuracy >= 70
              ? 'border-l-4 border-green-600'
              : accuracy >= 50
              ? 'border-l-4 border-yellow-500'
              : 'border-l-4 border-red-600'
          const hoverClass =
            accuracy >= 70 ? 'hover:bg-green-50' : accuracy >= 50 ? 'hover:bg-yellow-50' : 'hover:bg-red-50'

          // Find original index
          const originalIdx = items.findIndex((i) => i.item_id === item.item_id)

          return (
            <Link
              key={item.item_id}
              href={`${basePath}/${originalIdx}`}
              className={`block bg-white border border-border rounded-lg shadow-sm transition-all ${borderClass} ${hoverClass}`}
            >
              <div className="p-6">
                <p className="text-lg font-semibold text-foreground mb-2">
                  {color} {item.item_id}
                </p>
                <div className="flex gap-6 items-center">
                  <div className="text-center">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Accuracy</p>
                    <p className="text-2xl font-bold text-foreground">{accuracy.toFixed(1)}%</p>
                  </div>
                  <div className="text-center">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">Correct</p>
                    <p className="text-xl font-semibold text-foreground">
                      {item.metrics.correct}/{item.metrics.total}
                    </p>
                  </div>
                </div>
              </div>
            </Link>
          )
        })}
      </div>

      {filteredItems.length === 0 && (
        <div className="text-center py-12 bg-white border border-border rounded-lg">
          <p className="text-muted-foreground">No items match your filters</p>
        </div>
      )}
    </div>
  )
}
