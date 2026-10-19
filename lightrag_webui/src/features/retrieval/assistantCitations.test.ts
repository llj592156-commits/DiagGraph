import { describe, expect, test } from 'bun:test'
import type { AssistantResponse } from '@/types/assistant'
import { buildAssistantCitations, formatSourcePages } from './assistantCitations'

describe('assistant citation metadata', () => {
  test('uses the matching evidence primary parent and separates prerequisite locations', () => {
    const input: Pick<AssistantResponse, 'citations' | 'evidence'> = {
      citations: [{
        reference_id: '1', evidence_id: 'ev1', doc_id: 'd',
        file_path: 'manual.pdf', source_chunk_ids: ['child']
      }],
      evidence: [{
        evidence_id: 'ev1', content: 'Actual final evidence text.',
        primary_parent_id: 'main',
        parents: [
          { parent_id: 'prerequisite', metadata: { heading_path: ['Safety'], pages: [3] } },
          { parent_id: 'main', metadata: {
            heading_path: [' Maintenance ', '', 'Nozzle check'], pages: [85, 84, 84, -1, 0, 2.5, '86']
          } }
        ]
      }]
    }
    const snapshot = JSON.stringify(input)
    const [citation] = buildAssistantCitations(input)
    expect(citation.content).toBe('Actual final evidence text.')
    expect(citation.location).toEqual({ heading_path: ['Maintenance', 'Nozzle check'], pages: [84, 85] })
    expect(citation.related_locations).toEqual([{ heading_path: ['Safety'], pages: [3] }])
    expect(JSON.stringify(input)).toBe(snapshot)
  })

  test('does not mislabel a prerequisite as the main chapter when primary metadata is missing', () => {
    const [citation] = buildAssistantCitations({
      citations: [{
        reference_id: '1', evidence_id: 'ev1', doc_id: 'd',
        file_path: 'manual.pdf', source_chunk_ids: ['child']
      }],
      evidence: [{
        evidence_id: 'ev1', content: 'Final source.', primary_parent_id: 'missing',
        parents: [{ parent_id: 'prerequisite', metadata: { heading_path: ['Safety'], pages: [3] } }]
      }]
    })
    expect(citation.location).toBeUndefined()
    expect(citation.related_locations).toEqual([{ heading_path: ['Safety'], pages: [3] }])
  })

  test('missing evidence keeps citation identity without borrowing another source', () => {
    const [citation] = buildAssistantCitations({
      citations: [{
        reference_id: '1', evidence_id: 'missing', doc_id: 'd',
        file_path: 'manual.pdf', source_chunk_ids: ['child']
      }],
      evidence: [{ evidence_id: 'other', content: 'Unrelated text.' }]
    })
    expect(citation.evidence_id).toBe('missing')
    expect(citation.content).toBeUndefined()
    expect(citation.location).toBeUndefined()
  })

  test('compresses consecutive pages without hiding gaps', () => {
    expect(formatSourcePages([90, 84, 85, 86, 90])).toBe('84–86, 90')
    expect(formatSourcePages([10, 12])).toBe('10, 12')
    expect(formatSourcePages([84])).toBe('84')
    expect(formatSourcePages([])).toBe('')
  })
})
