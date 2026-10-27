import type { Metadata } from 'next'
import { Navbar, themeScript } from '@/components/v2/Navbar'

export const metadata: Metadata = {
  title: 'Model Leaderboards',
  description: 'Which model to run behind each Hindsight operation: retain, recall, consolidation, reflect and knowledge pages.',
}

export default function V2Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="v2 min-h-screen font-body antialiased">
      <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      <Navbar />
      {children}
      <footer className="border-t border-border mt-24">
        <div className="max-w-6xl mx-auto px-4 py-10 flex flex-col sm:flex-row gap-4 sm:items-center justify-between text-sm text-muted-foreground">
          <span>© 2026 Vectorize. Hindsight is open source under the MIT license.</span>
          <div className="flex gap-5">
            <a className="hover:text-foreground" href="https://hindsight.vectorize.io/">Docs</a>
            <a className="hover:text-foreground" href="https://github.com/vectorize-io/hindsight-benchmarks">Benchmark code</a>
            <a className="hover:text-foreground" href="https://agentmemorybenchmark.ai">Agent Memory Benchmark</a>
          </div>
        </div>
      </footer>
    </div>
  )
}
