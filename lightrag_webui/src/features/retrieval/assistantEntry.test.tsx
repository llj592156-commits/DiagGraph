import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import type { ComponentType } from 'react'
import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/render'
import { resetCustomization, seedCustomization } from '@/test/customization'
import {
  captureProcessState,
  restoreProcessState,
  type ProcessStateSnapshot
} from '@/test/processState'
import { useSettingsStore } from '@/stores/settings'
import { defaultQuerySettings, useQuerySettingsStore } from '@/stores/querySettings'
import { useWebuiRetrievalHistoryStore } from '@/stores/webuiRetrievalHistory'
import { useWorkspaceRetrievalHistoryStore } from '@/stores/workspaceRetrievalHistory'
import { AuthenticationRequiredError } from '@/api/errors'
import type { AssistantRequest, AssistantResponse } from '@/types/assistant'

const stores = [
  useSettingsStore,
  useQuerySettingsStore,
  useWebuiRetrievalHistoryStore,
  useWorkspaceRetrievalHistoryStore
]
const requests: AssistantRequest[] = []
const legacy = mock(async () => ({ response: 'legacy answer' }))
let realApi: Record<string, unknown>
let entries: Record<string, ComponentType>
let snapshot: ProcessStateSnapshot
let respond: (request: AssistantRequest, signal?: AbortSignal) => Promise<AssistantResponse>

const response = (updates: Partial<AssistantResponse> = {}): AssistantResponse => ({
  status: 'answered',
  answer: 'Use 002 ink. [1]',
  route: { strategy: 'retrieve_once' },
  complexity: 'B',
  metadata: { elapsed_ms: 1200 },
  error: null,
  citations: [
    {
      reference_id: '1',
      evidence_id: 'ev1',
      doc_id: 'doc1',
      file_path: 'L4263.pdf',
      source_chunk_ids: ['c1']
    }
  ],
  evidence: [{ evidence_id: 'ev1', content: 'BK, C, M and Y: 002.' }],
  ...updates
})

beforeAll(async () => {
  realApi = { ...(await import('@/api/lightrag')) }
  mock.module('@/api/lightrag', () => ({
    ...realApi,
    queryAssistant: mock(async (request: AssistantRequest, signal?: AbortSignal) => {
      requests.push(request)
      return respond(request, signal)
    }),
    queryText: legacy,
    queryTextStream: legacy
  }))
  entries = {
    admin: (await import('@/features/RetrievalView')).default,
    workspace: (await import('@/features/workspace/WorkspaceQueryView')).default
  }
})
afterAll(() => {
  mock.module('@/api/lightrag', () => realApi)
})
beforeEach(() => {
  snapshot = captureProcessState(stores)
  seedCustomization()
  requests.length = 0
  legacy.mockClear()
  respond = async () => response()
  useWebuiRetrievalHistoryStore.setState({ history: [] })
  useWorkspaceRetrievalHistoryStore.setState({ history: [] })
  useQuerySettingsStore.setState({
    querySettings: {
      ...defaultQuerySettings,
      stream: true,
      only_need_prompt: true,
      user_prompt: 'debug setting must not leak'
    }
  })
})
afterEach(() => {
  cleanup()
  resetCustomization()
  act(() => restoreProcessState(stores, snapshot))
})

const send = async (text: string) => {
  const input = within(screen.getByRole('search')).getByRole('textbox')
  fireEvent.change(input, { target: { value: text } })
  fireEvent.submit(screen.getByRole('search'))
  await act(async () => {})
}

describe('assistant UI integration', () => {
  test('restored assistant turns keep their sources and remain available to the next question', async () => {
    const Entry = entries.workspace
    respond = async () => response({ evidence: [{
      evidence_id: 'ev1', content: 'BK, C, M and Y: 002.', primary_parent_id: 'parent1',
      parents: [{ parent_id: 'parent1', metadata: { heading_path: ['Ink bottles'], pages: [140] } }]
    }] })
    renderWithProviders(<Entry />)
    await send('L4263 ink?')
    await waitFor(() => expect(screen.queryAllByText('Sources')).toHaveLength(1))
    cleanup()
    renderWithProviders(<Entry />)
    expect(screen.queryAllByText('Sources')).toHaveLength(1)
    const user = userEvent.setup()
    await user.click(screen.getByText('Sources'))
    expect(screen.queryAllByText('Ink bottles')).toHaveLength(1)
    expect(screen.queryAllByText('Source pages: 140')).toHaveLength(1)
    await send('Is that the same for all colours?')
    await waitFor(() => expect(requests).toHaveLength(2))
    expect(requests[1].history).toEqual([
      { role: 'user', content: 'L4263 ink?' },
      { role: 'assistant', content: 'Use 002 ink. [1]' }
    ])
  })

  test('an oversized question stays in the composer and never reaches the API', async () => {
    const Entry = entries.workspace
    renderWithProviders(<Entry />)
    const text = 'x'.repeat(8193)
    await send(text)
    expect(screen.getByRole('alert').textContent).toContain('8,192')
    expect(
      (within(screen.getByRole('search')).getByRole('textbox') as HTMLInputElement).value
    ).toBe(text)
    expect(requests).toHaveLength(0)
  })

  for (const entry of ['admin', 'workspace']) {
    test(`${entry} defaults to assistant, shows sources and carries the clarification turn`, async () => {
      const Entry = entries[entry]
      respond = async () =>
        response({
          status: 'needs_clarification',
          answer: 'Which printer model?',
          citations: [],
          evidence: []
        })
      renderWithProviders(<Entry />)
      await send('My printer has a problem')
      await waitFor(() => expect(screen.queryAllByText('Which printer model?')).toHaveLength(1))
      expect(requests[0]).toEqual({ query: 'My printer has a problem', history: [] })
      expect(screen.getByRole('button', { name: 'Assistant' }).getAttribute('aria-pressed')).toBe(
        'true'
      )
      respond = async () => response()
      await send('It is L4263. Which ink does it use?')
      await waitFor(() => expect(screen.queryAllByText('Use 002 ink. [1]')).toHaveLength(1))
      expect(requests[1].history).toEqual([
        { role: 'user', content: 'My printer has a problem' },
        { role: 'assistant', content: 'Which printer model?' }
      ])
      expect(Object.keys(requests[1]).sort()).toEqual(['history', 'query'])
      const user = userEvent.setup()
      await user.click(screen.getByText('Sources'))
      await user.click(screen.getByText('[1] L4263.pdf'))
      expect(screen.queryAllByText('BK, C, M and Y: 002.')).toHaveLength(1)
      expect(legacy.mock.calls).toHaveLength(0)
      const ownStore =
        entry === 'admin' ? useWebuiRetrievalHistoryStore : useWorkspaceRetrievalHistoryStore
      const otherStore =
        entry === 'admin' ? useWorkspaceRetrievalHistoryStore : useWebuiRetrievalHistoryStore
      expect(ownStore.getState().history).toHaveLength(4)
      expect(otherStore.getState().history).toHaveLength(0)
    })
  }
  test('HTTP 200 business failure is rendered as failure without exposing a draft', async () => {
    const Entry = entries.workspace
    respond = async () =>
      response({
        status: 'failed',
        answer: 'unverified draft',
        error: { code: 'model_timeout', stage: 'routing' }
      })
    renderWithProviders(<Entry />)
    await send('L4263 ink?')
    await waitFor(() => expect(screen.queryAllByText('Request failed')).toHaveLength(1))
    expect(screen.queryAllByText('unverified draft')).toHaveLength(0)
    expect(screen.queryAllByText('Sources')).toHaveLength(0)
    expect(useWorkspaceRetrievalHistoryStore.getState().history[1].isError).toBe(true)
  })
  test('authentication loss rolls back the optimistic pair', async () => {
    const Entry = entries.workspace
    respond = async () => {
      throw new AuthenticationRequiredError()
    }
    renderWithProviders(<Entry />)
    await send('L4263 ink?')
    await waitFor(() => expect(screen.queryAllByText('L4263 ink?')).toHaveLength(0))
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toHaveLength(0)
  })
  test('stop and clear prevent a late assistant response from restoring the old conversation', async () => {
    const Entry = entries.workspace
    let resolve!: (value: AssistantResponse) => void
    let signal: AbortSignal | undefined
    respond = async (_request, currentSignal) => {
      signal = currentSignal
      return await new Promise<AssistantResponse>((done) => {
        resolve = done
      })
    }
    renderWithProviders(<Entry />)
    await send('L4263 ink?')
    expect(
      screen.queryAllByText('Analysing your question and checking the available information…')
    ).toHaveLength(1)
    await waitFor(() =>
      expect((screen.getByRole('button', { name: 'Stop' }) as HTMLButtonElement).disabled).toBe(
        false
      )
    )
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Stop' }))
    await user.click(screen.getByRole('button', { name: 'Clear' }))
    await act(async () => {
      resolve(response())
    })
    expect(signal?.aborted).toBe(true)
    expect(screen.queryAllByText('Use 002 ink. [1]')).toHaveLength(0)
    expect(useWorkspaceRetrievalHistoryStore.getState().history).toHaveLength(0)
  })
  for (const status of ['escalated', 'insufficient_evidence'] as const) {
    test(`displays ${status} as a normal business outcome`, async () => {
      const Entry = entries.workspace
      respond = async () =>
        response({ status, answer: `Outcome: ${status}`, citations: [], evidence: [] })
      renderWithProviders(<Entry />)
      await send('Question')
      await waitFor(() => expect(screen.queryAllByText(`Outcome: ${status}`)).toHaveLength(1))
      expect(useWorkspaceRetrievalHistoryStore.getState().history[1].isError).toBe(false)
    })
  }
})
