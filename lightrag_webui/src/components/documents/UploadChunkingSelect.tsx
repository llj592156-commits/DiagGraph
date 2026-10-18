import { useTranslation } from 'react-i18next'
import type { UploadChunkingStrategy } from '@/lib/uploadChunking'

interface Props {
  value: UploadChunkingStrategy
  onChange: (value: UploadChunkingStrategy) => void
  disabled?: boolean
}

export default function UploadChunkingSelect({ value, onChange, disabled }: Props) {
  const { t } = useTranslation()
  return (
    <div className="space-y-2">
      <label htmlFor="upload-chunking" className="text-sm font-medium">
        {t('documentPanel.uploadDocuments.chunking.label')}
      </label>
      <select
        id="upload-chunking"
        value={value}
        disabled={disabled}
        onChange={event => onChange(event.target.value as UploadChunkingStrategy)}
        className="w-full rounded-md border bg-background px-3 py-2 text-sm"
      >
        {(['auto', 'F', 'R', 'V', 'P', 'C', 'H'] as const).map(strategy => (
          <option key={strategy} value={strategy}>
            {t(`documentPanel.uploadDocuments.chunking.${strategy}`)}
          </option>
        ))}
      </select>
      {value === 'H' && <p className="text-sm text-muted-foreground">
        {t('documentPanel.uploadDocuments.chunking.parentChildHelp')}
      </p>}
    </div>
  )
}
