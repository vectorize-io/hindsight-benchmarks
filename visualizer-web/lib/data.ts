import { readFileSync } from 'fs'
import { join } from 'path'
import { gunzipSync } from 'zlib'
import type { BenchmarkResult } from './types'

export type * from './types'

function loadJSONFile(filename: string): any | null {
  const resultsPath = join(process.cwd(), '..', 'results')

  try {
    // Try .json.gz first
    const gzPath = join(resultsPath, `${filename}.gz`)
    try {
      const compressed = readFileSync(gzPath)
      const decompressed = gunzipSync(compressed)
      return JSON.parse(decompressed.toString('utf-8'))
    } catch {}

    // Try regular .json
    const jsonPath = join(resultsPath, filename)
    const content = readFileSync(jsonPath, 'utf-8')
    return JSON.parse(content)
  } catch (error) {
    console.error(`Failed to load ${filename}:`, error)
    return null
  }
}

export function loadLongMemEval(): BenchmarkResult | null {
  return loadJSONFile('longmemeval.json')
}

export function loadLoComo(): BenchmarkResult | null {
  return loadJSONFile('locomo.json')
}

export { getCategoryName } from './utils'
