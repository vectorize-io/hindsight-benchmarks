import Link from 'next/link'
import { Button } from '@/components/ui/button'

export default function NotFound() {
  return (
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <div className="text-center py-12">
        <h1 className="text-4xl font-heading font-bold text-foreground mb-4">⚠️ Benchmark Results Not Found</h1>
        <p className="text-muted-foreground mb-6">The requested benchmark results are not available.</p>
        <Button asChild>
          <Link href="/">← Back to benchmarks</Link>
        </Button>
      </div>
    </main>
  )
}
