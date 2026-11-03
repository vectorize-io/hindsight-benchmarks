// Hindsight's published Agent Memory Benchmark scores, one per dataset split.
export const AMB_SITE = 'https://agentmemorybenchmark.ai'
const AMB_RUN = 'https://agentmemorybenchmark.ai/run'

export const AMB_RESULTS = [
  { benchmark: 'LongMemEval', split: 'S', score: 94.6, href: `${AMB_RUN}/outputs/longmemeval/hindsight/rag/s.json.gz` },
  { benchmark: 'LoComo', split: '10', score: 92.0, href: `${AMB_RUN}/outputs/locomo/locomo-hindsight/rag/locomo10.json.gz` },
  { benchmark: 'PersonaMem', split: '32K', score: 86.6, href: `${AMB_RUN}/outputs/personamem/hindsight/rag/32k.json.gz` },
  { benchmark: 'BEAM', split: '100K', score: 75.0, href: `${AMB_RUN}/outputs/beam/hindsight/single-query/100k.json` },
  { benchmark: 'BEAM', split: '1M', score: 73.9, href: `${AMB_RUN}/outputs/beam/hindsight/single-query/1m.json` },
  { benchmark: 'LifeBench', split: 'EN', score: 71.5, href: `${AMB_RUN}/outputs/lifebench/hindsight/rag/en.json.gz` },
  { benchmark: 'BEAM', split: '500K', score: 71.1, href: `${AMB_RUN}/outputs/beam/hindsight/single-query/500k.json` },
  { benchmark: 'BEAM', split: '10M', score: 64.1, href: `${AMB_RUN}/outputs/beam/hindsight/single-query/10m.json` },
]
