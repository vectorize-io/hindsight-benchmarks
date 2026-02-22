import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function getCategoryName(category: number | string): string {
  if (typeof category === 'string') return category

  const categories: Record<number, string> = {
    1: 'Multi-hop',
    2: 'Single-hop',
    3: 'Temporal',
    4: 'Open-domain',
  }
  return categories[category] || 'Unknown'
}
