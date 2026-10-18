import { describe, expect, test } from 'bun:test'
import { fileNameForChunking } from './uploadChunking'

describe('upload chunking selection', () => {
  test('automatic mode preserves existing hints', () => {
    expect(fileNameForChunking('L3266.[docling-R(chunk_ts=400)].pdf', 'auto'))
      .toBe('L3266.[docling-R(chunk_ts=400)].pdf')
  })
  test('parent-child uses Docling without changing the canonical document name', () => {
    expect(fileNameForChunking('L3266_L5298.pdf', 'H')).toBe('L3266_L5298.[docling-H].pdf')
    expect(fileNameForChunking('guide.[docling(do_ocr=false)-!R].pdf', 'H'))
      .toBe('guide.[docling(do_ocr=false)-!H].pdf')
  })
  test('changing strategy preserves engine parameters and non-chunk flags', () => {
    expect(fileNameForChunking('guide.[docling(do_ocr=false)-!R(chunk_ts=400)].pdf', 'F'))
      .toBe('guide.[docling(do_ocr=false)-!F].pdf')
  })
  test('H rejects files without PDF structure and incompatible analysis flags', () => {
    expect(() => fileNameForChunking('guide.txt', 'H')).toThrow('requires a PDF')
    expect(() => fileNameForChunking('guide.[docling-itR].pdf', 'H')).toThrow('cannot combine')
  })
})
