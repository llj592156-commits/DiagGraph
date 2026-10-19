import { useTranslation } from 'react-i18next'
import Button from '@/components/ui/Button'
import type { QueryWorkflow } from '@/types/assistant'

export default function QueryWorkflowPicker({
  value,
  onChange,
  disabled
}: {
  value: QueryWorkflow
  onChange: (value: QueryWorkflow) => void
  disabled: boolean
}) {
  const { t } = useTranslation()
  return (
    <div className="flex flex-wrap items-center gap-2">
      <div role="group" aria-label={t('assistant.workflow')} className="flex gap-1">
        {(['assistant', 'retrieval'] as const).map((workflow) => (
          <Button
            key={workflow}
            type="button"
            size="sm"
            variant={workflow === value ? 'default' : 'outline'}
            aria-pressed={workflow === value}
            disabled={disabled}
            onClick={() => onChange(workflow)}
          >
            {t(`assistant.workflows.${workflow}`)}
          </Button>
        ))}
      </div>
      {value === 'assistant' && (
        <p className="text-muted-foreground text-xs">{t('assistant.description')}</p>
      )}
    </div>
  )
}
