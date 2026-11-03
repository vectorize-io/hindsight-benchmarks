import type { Metadata } from 'next'
import { FAMILIES } from '@/lib/models'
import { ModelsPage } from '@/components/ModelsView'

export const metadata: Metadata = {
  title: 'Hindsight Models',
  description: "Hindsight's decision, embedding and reranking models on public industry benchmarks.",
}

export default function Page() {
  return <ModelsPage family={FAMILIES[0]} />
}
