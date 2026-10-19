import type {
  AssistantReadableCitation,
  AssistantResponse,
  AssistantSourceLocation
} from '@/types/assistant'

function sourceLocation(metadata: Record<string, unknown>): AssistantSourceLocation | undefined {
  const heading_path = Array.isArray(metadata.heading_path)
    ? metadata.heading_path.filter((h): h is string => typeof h === 'string' && !!h.trim()).map((h) => h.trim())
    : []
  const pages = Array.isArray(metadata.pages)
    ? [...new Set(metadata.pages.filter((p): p is number => Number.isSafeInteger(p) && p > 0))].sort((a, b) => a - b)
    : []
  return heading_path.length || pages.length ? { heading_path, pages } : undefined
}

/** Persist only readable locations from the final evidence, keeping main and prerequisite chapters separate. */
export function buildAssistantCitations(
  response: Pick<AssistantResponse, 'citations' | 'evidence'>
): AssistantReadableCitation[] {
  const evidence = new Map(response.evidence.map((item) => [item.evidence_id, item]))
  return response.citations.map((citation) => {
    const item = evidence.get(citation.evidence_id)
    const primary = item?.parents?.find((parent) => parent.parent_id === item.primary_parent_id)
    const related_locations = (item?.parents || [])
      .filter((parent) => parent.parent_id !== item?.primary_parent_id)
      .map((parent) => sourceLocation(parent.metadata))
      .filter((location): location is AssistantSourceLocation => !!location)
    return {
      ...citation,
      content: item?.content,
      location: primary ? sourceLocation(primary.metadata) : undefined,
      related_locations
    }
  })
}

export function formatSourcePages(pages: number[]): string {
  const sorted = [...new Set(pages)].sort((a, b) => a - b)
  const ranges: string[] = []
  for (let i = 0; i < sorted.length; i++) {
    const start = sorted[i]
    let end = start
    while (i + 1 < sorted.length && sorted[i + 1] === end + 1) end = sorted[++i]
    ranges.push(start === end ? String(start) : `${start}–${end}`)
  }
  return ranges.join(', ')
}
