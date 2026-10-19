import { describe, expect, test } from 'bun:test'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/render'
import type { AssistantOutcome } from '@/types/assistant'
import AssistantResult from './AssistantResult'

const result = (): AssistantOutcome => ({
  status: 'answered',
  strategy: 'retrieve_once',
  citations: [
    {
      reference_id: '1', evidence_id: 'ev1', doc_id: 'doc',
      file_path: 'L4263.pdf', source_chunk_ids: ['c1'],
      content: 'Choose Maintenance, then Nozzle Check.',
      location: {
        heading_path: ['Maintenance', 'Checking and cleaning the print head'],
        pages: [84, 85, 86]
      },
      related_locations: [{ heading_path: ['Safety precautions'], pages: [10, 12] }]
    },
    {
      reference_id: '2', evidence_id: 'ev2', doc_id: 'doc',
      file_path: 'L4263.pdf', source_chunk_ids: ['c2'],
      content: 'Visually check the ink levels.',
      location: { heading_path: ['Maintenance', 'Checking ink levels'], pages: [84] }
    }
  ]
})

describe('assistant source locations', () => {
  test('distinguishes two citations from the same file by chapter and pages', async () => {
    renderWithProviders(<AssistantResult result={result()} />)
    const user = userEvent.setup()
    await user.click(screen.getByText('Sources'))
    expect(screen.getByText('Maintenance › Checking and cleaning the print head').textContent)
      .toBe('Maintenance › Checking and cleaning the print head')
    expect(screen.queryAllByText('Source pages: 84–86')).toHaveLength(1)
    expect(screen.queryAllByText('Maintenance › Checking ink levels')).toHaveLength(1)
    expect(screen.queryAllByText('Source pages: 84')).toHaveLength(1)
    await user.click(screen.getByText('[1] L4263.pdf'))
    expect(screen.queryAllByText('Choose Maintenance, then Nozzle Check.')).toHaveLength(1)
    expect(screen.queryAllByText('Related sections')).toHaveLength(1)
    expect(screen.queryAllByText('Safety precautions')).toHaveLength(1)
    expect(screen.queryAllByText('Source pages: 10, 12')).toHaveLength(1)
  })

  test('old saved citations still show their original text without invented locations', async () => {
    const old = result()
    old.citations = [{
      reference_id: '1', evidence_id: 'old', doc_id: 'doc', file_path: 'old.pdf',
      source_chunk_ids: ['old'], content: 'Previously saved source text.'
    }]
    renderWithProviders(<AssistantResult result={old} />)
    const user = userEvent.setup()
    await user.click(screen.getByText('Sources'))
    expect(screen.queryAllByText('Chapter and page information was not provided.')).toHaveLength(1)
    await user.click(screen.getByText('[1] old.pdf'))
    expect(screen.queryAllByText('Previously saved source text.')).toHaveLength(1)
    expect(screen.queryAllByText(/Source pages:/)).toHaveLength(0)
  })
})
