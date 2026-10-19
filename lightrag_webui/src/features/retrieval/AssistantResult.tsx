import { useTranslation } from 'react-i18next'
import type { AssistantOutcome } from '@/types/assistant'
import { formatSourcePages } from './assistantCitations'

export default function AssistantResult({ result }: { result: AssistantOutcome }) {
  const { t } = useTranslation()
  return (
    <div className="mt-2 space-y-2 text-xs">
      <p role="status" className="text-muted-foreground">
        {t(`assistant.status.${result.status}`)}
      </p>
      {result.historyTruncated && (
        <p className="text-muted-foreground">{t('assistant.historyTruncated')}</p>
      )}
      {result.citations.length > 0 && (
        <details className="rounded-md border p-2">
          <summary className="cursor-pointer font-medium">{t('assistant.sources')}</summary>
          <div className="mt-2 space-y-2">
            {result.citations.map((citation) => (
              <details
                key={`${citation.reference_id}-${citation.evidence_id}`}
                className="rounded border p-2"
              >
                <summary className="cursor-pointer break-words">
                  <span>[{citation.reference_id}] {citation.file_path}</span>
                  {citation.location?.heading_path.length ? (
                    <span className="text-muted-foreground mt-1 block">
                      {citation.location.heading_path.join(' › ')}
                    </span>
                  ) : null}
                  {citation.location?.pages.length ? (
                    <span className="text-muted-foreground mt-1 block">
                      {t('assistant.sourcePages', { pages: formatSourcePages(citation.location.pages) })}
                    </span>
                  ) : null}
                  {!citation.location && (
                    <span className="text-muted-foreground mt-1 block">
                      {t('assistant.sourceLocationUnavailable')}
                    </span>
                  )}
                </summary>
                <p className="mt-2 text-sm break-words whitespace-pre-wrap">
                  {citation.content || t('assistant.sourceUnavailable')}
                </p>
                {!!citation.related_locations?.length && (
                  <div className="text-muted-foreground mt-3 space-y-1 border-t pt-2">
                    <p className="font-medium">{t('assistant.relatedSections')}</p>
                    {citation.related_locations.map((location, index) => (
                      <p key={index} className="break-words">
                        <span>{location.heading_path.join(' › ')}</span>
                        {location.heading_path.length > 0 && location.pages.length > 0 ? ' · ' : ''}
                        {location.pages.length > 0 && (
                          <span>{t('assistant.sourcePages', { pages: formatSourcePages(location.pages) })}</span>
                        )}
                      </p>
                    ))}
                  </div>
                )}
              </details>
            ))}
          </div>
        </details>
      )}
      {result.status === 'failed' && result.error?.code && (
        <details>
          <summary className="cursor-pointer">{t('assistant.errorDetails')}</summary>
          <p className="mt-1 break-words">
            {result.error.code} · {result.error.stage}
          </p>
        </details>
      )}
    </div>
  )
}
