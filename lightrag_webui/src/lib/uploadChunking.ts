export type UploadChunkingStrategy = 'auto' | 'F' | 'R' | 'V' | 'P' | 'C' | 'H'

/** Preserve the canonical source name while using the server's existing hint channel. */
export const fileNameForChunking = (name: string, strategy: UploadChunkingStrategy): string => {
  if (strategy === 'auto') return name
  const match = name.match(/\.\[([^\]]*)\](\.[^.]+)$/)
  const clean = match ? name.slice(0, match.index) + match[2] : name
  const extension = clean.match(/\.[^.]+$/)?.[0]
  if (!extension) throw new Error('A file extension is required')
  if (strategy === 'H' && extension.toLowerCase() !== '.pdf') {
    throw new Error('Parent-child chunking requires a PDF')
  }
  let engine = ''
  let flags = ''
  if (match) {
    let depth = 0
    let split = -1
    for (let i = 0; i < match[1].length; i++) {
      const ch = match[1][i]
      if (ch === '(') depth++
      if (ch === ')') depth--
      if (ch === '-' && depth === 0) { split = i; break }
    }
    engine = split < 0 ? match[1] : match[1].slice(0, split)
    depth = 0
    for (const ch of split < 0 ? '' : match[1].slice(split + 1)) {
      if (ch === '(') depth++
      else if (ch === ')') depth--
      else if (depth === 0 && 'ite!'.includes(ch)) flags += ch
    }
  }
  if (strategy === 'H') {
    if (/[ite]/.test(flags)) throw new Error('Parent-child chunking cannot combine with separate image/table/equation analysis')
    if (!/^docling(?:\(|$)/i.test(engine)) engine = 'docling'
  }
  return `${clean.slice(0, -extension.length)}.[${engine}-${flags}${strategy}]${extension}`
}
