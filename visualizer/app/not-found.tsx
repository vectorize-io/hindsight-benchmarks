import Link from 'next/link'

export default function NotFound() {
  return (
    <main className="container mx-auto max-w-7xl px-4 py-8">
      <div className="text-center py-12">
        <h1 className="text-4xl font-bold text-foreground mb-4">⚠️ Benchmark Results Not Found</h1>
        <p className="text-muted-foreground mb-6">The requested benchmark results are not available.</p>
        <Link
          href="/"
          className="inline-flex items-center px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 text-sm font-medium"
        >
          ← Back to benchmarks
        </Link>
      </div>
    </main>
  )
}
