import { ModelWithResult } from '@/lib/types'

interface NonViableModelsProps {
  models: ModelWithResult[]
}

function ModelRow({ model }: { model: ModelWithResult }) {
  return (
    <li className="flex items-center gap-3 text-sm">
      <span className="text-red-400">✗</span>
      <span className="font-medium text-foreground">{model.config.model_name}</span>
    </li>
  )
}

export default function NonViableModels({ models }: NonViableModelsProps) {
  if (models.length === 0) return null

  return (
    <div className="mt-12 pt-8 border-t border-border">
      <h2 className="text-2xl font-heading font-bold text-foreground mb-2">Models Not Viable for This Task</h2>
      <p className="text-sm text-muted-foreground mb-6">
        These models were tested but scored 0 successes — they could not follow the required JSON schema.
      </p>

      <div className="border border-border rounded-lg overflow-hidden bg-card">
        <ul className="px-5 py-4 space-y-2">
          {models.map(m => <ModelRow key={`${m.config.provider_id}-${m.config.model_id}`} model={m} />)}
        </ul>
      </div>
    </div>
  )
}
