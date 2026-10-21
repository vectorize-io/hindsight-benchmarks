import type { Metadata } from 'next'
import { loadRerankerData } from '@/lib/reranker'
import RerankerTable from '@/components/leaderboard/RerankerTable'

export const metadata: Metadata = {
  title: 'Reranker Benchmark — Frozen Candidate Comparisons',
  description: 'Direct passage reranking on LoCoMo and SciFact: evidence retrieval quality, uncertainty, latency, and API usage.',
}

export default function RerankerLeaderboardPage() {
  const rerankers = loadRerankerData()
  return <main className="container mx-auto max-w-7xl px-4 py-8">
    <h1 className="text-4xl font-heading font-bold mb-3">Reranker Benchmark</h1>
    <p className="text-lg text-muted-foreground mb-6">Models rank identical frozen candidates using their own relevance scores.</p>
    <div className="rounded-lg border border-border bg-card p-5 mb-8 space-y-3 text-sm text-muted-foreground">
      <p>Quality, latency, and cost are reported separately. These are passage-ranking measurements; they do not measure Hindsight&apos;s extraction, graph retrieval, or answer generation.</p>
      <p>Both datasets are public and may have influenced model training or selection. Ettin explicitly selected checkpoints using NanoBEIR and MTEB. These results establish performance on the published fixtures, not unseen-data performance.</p>
      <p>The previous March 2026 metrics have been retired. They measured an unpinned recall pipeline that transformed provider scores and blended them with retrieval and recency signals, so they did not isolate reranker quality.</p>
      <a className="underline" href="https://github.com/vectorize-io/hindsight-benchmarks/blob/main/benchmark-runner/RERANKER_BENCHMARK.md">Methodology, limitations, raw scores, and reproduction commands</a>
    </div>
    {[
      { id: 'locomo', title: 'LoCoMo conversation evidence', description: 'ten conversations; original evidence IDs' },
      { id: 'scifact-evidence', title: 'SciFact annotated evidence', description: 'original development-set subset; explicit SUPPORT or CONTRADICT annotations; rankings reused from the full 300-query run' },
      { id: 'scifact', title: 'BEIR SciFact cited-document retrieval', description: 'BEIR test split, corresponding to the original SciFact development set; cited-document qrels' },
    ].map(({ id: dataset, title, description }) => {
      const rows = rerankers.filter(r => r.dataset === dataset)
      return <section key={dataset} className="mb-12">
        <h2 className="text-2xl font-heading font-bold mb-3">{title}</h2>
        <p className="text-sm text-muted-foreground mb-4">{rows.length ? `${rows[0].total_questions} queries; ${description}; 100 frozen candidates per query. Candidate relevance recall ceiling: ${(rows[0].retrieval_ceiling_recall * 100).toFixed(1)}%.` : 'No complete results published.'}</p>
        {rows.length > 0 && <RerankerTable rerankers={rows} />}
      </section>
    })}
    <section className="space-y-4 text-sm text-muted-foreground">
      <h2 className="text-2xl font-heading font-bold text-foreground">How to read these numbers</h2>
      <p><strong>nDCG@10</strong> rewards annotated evidence near the top, relative to the ideal ranking. <strong>MRR</strong> averages the reciprocal rank of the first annotated evidence passage. <strong>Recall@K</strong> is the fraction of all annotated evidence passages recovered in the first K; <strong>Hit@1</strong> is the fraction of queries whose first passage is annotated evidence.</p>
      <p>BM25 and pinned BGE-base embeddings produce a fixed top-100 list by reciprocal rank fusion. Gold evidence is never inserted into the candidate pool. Retrieval misses count as zero. LoCoMo evidence IDs receive only syntax normalization; missing annotations and unresolved references are listed in the fixture. LoCoMo labels may omit other useful passages.</p>
      <p>SciFact has two different relevance definitions. Standard BEIR qrels credit the papers cited by a claim&apos;s source sentence, including papers without annotated evidence. The evidence view instead credits only explicitly annotated supporting or contradicting abstracts and excludes all 112 queries with no such annotations. Both views use identical recorded rankings; neither is an unseen holdout.</p>
      <p>Local models run on an RTX PRO 5000 Blackwell GPU in bfloat16, with batch size 32 and three warmup batches. Hosted models run from the same machine. Latency covers the whole candidate pool and includes client queueing, pacing, and retries; Jev is evaluated both with independent pointwise Noul requests and with Hindsight&apos;s listwise Choice/tournament ranking, pruning disabled. Both configurations use bounded concurrency. This measures the recorded configuration, not each provider&apos;s maximum throughput.</p>
      <p>API cost uses reported usage and a verified list rate where available. Local compute and unverified rates are left unpriced. Confidence intervals resample whole conversations for LoCoMo and individual queries for SciFact; they describe sampling uncertainty and cannot account for incomplete labels or training contamination.</p>
    </section>
  </main>
}
