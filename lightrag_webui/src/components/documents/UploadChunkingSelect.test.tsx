import { describe, expect, mock, test } from 'bun:test'
import { fireEvent, screen } from '@testing-library/react'
import { renderWithProviders } from '@/test/render'
import UploadChunkingSelect from './UploadChunkingSelect'

describe('upload chunking choices', () => {
  test('parent-child is offered alongside existing strategies and can be selected', () => {
    const changed = mock()
    renderWithProviders(<UploadChunkingSelect value="auto" onChange={changed} />)
    expect(screen.getAllByRole('option').map(option => (option as HTMLOptionElement).value))
      .toEqual(['auto', 'F', 'R', 'V', 'P', 'C', 'H'])
    fireEvent.change(screen.getByRole('combobox', { name: 'Chunking strategy' }), { target: { value: 'H' } })
    expect(changed).toHaveBeenCalledWith('H')
  })
  test('H explains PDF and current knowledge-base behavior', () => {
    renderWithProviders(<UploadChunkingSelect value="H" onChange={() => {}} disabled />)
    expect((screen.getByRole('combobox') as HTMLSelectElement).disabled).toBe(true)
    expect(screen.getByText(/PDF only/).textContent).toContain('current knowledge base')
  })
})
