import type { Metadata } from 'next'
import { notFound } from 'next/navigation'
import { FAMILIES } from '@/lib/models'
import { ModelsPage } from '@/components/ModelsView'

const byId = (id: string) => FAMILIES.find((f) => f.id === id)

export function generateStaticParams() {
  return FAMILIES.map((f) => ({ family: f.id }))
}

export async function generateMetadata({ params }: { params: Promise<{ family: string }> }): Promise<Metadata> {
  const f = byId((await params).family)
  return f ? { title: `${f.model} benchmarks`, description: f.short } : {}
}

export default async function FamilyPage({ params }: { params: Promise<{ family: string }> }) {
  const f = byId((await params).family)
  if (!f) notFound()
  return <ModelsPage family={f} />
}
