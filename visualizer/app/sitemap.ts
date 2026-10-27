import type { MetadataRoute } from 'next'
import { BOARDS } from '@/lib/data'

export const dynamic = 'force-static'

const siteUrl = 'https://benchmarks.hindsight.vectorize.io'

export default function sitemap(): MetadataRoute.Sitemap {
  const pages = ['', '/benchmarks', '/leaderboards', ...BOARDS.map((b) => `/leaderboards/${b.id}`)]
  return pages.map((path) => ({
    url: `${siteUrl}${path}`,
    lastModified: new Date(),
    changeFrequency: 'weekly',
    priority: path === '' ? 1 : 0.8,
  }))
}
