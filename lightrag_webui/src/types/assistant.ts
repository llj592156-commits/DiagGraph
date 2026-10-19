export type AssistantStatus =
  'answered' | 'needs_clarification' | 'escalated' | 'insufficient_evidence' | 'failed'

export type AssistantStrategy = 'direct' | 'retrieve_once' | 'diagnose' | 'clarify' | 'escalate'
export type QueryWorkflow = 'assistant' | 'retrieval'

export interface AssistantRequest {
  query: string
  history: { role: 'user' | 'assistant'; content: string }[]
}

export interface AssistantCitation {
  reference_id: string
  evidence_id: string
  file_path: string
  doc_id: string
  source_chunk_ids: string[]
}

export interface AssistantSourceLocation {
  heading_path: string[]
  pages: number[]
}

export interface AssistantReadableCitation extends AssistantCitation {
  content?: string
  location?: AssistantSourceLocation
  related_locations?: AssistantSourceLocation[]
}

export interface AssistantEvidence {
  evidence_id: string
  content: string
  primary_parent_id?: string | null
  parents?: { parent_id: string; metadata: Record<string, unknown> }[]
}

export interface AssistantResponse {
  status: AssistantStatus
  answer: string
  route: { strategy: AssistantStrategy } | null
  complexity: string | null
  citations: AssistantCitation[]
  evidence: AssistantEvidence[]
  metadata: { elapsed_ms?: number; [key: string]: unknown }
  error: { code?: string; stage?: string; message?: string } | null
}

/** Keep readable sources in local history without duplicating indexing metadata. */
export interface AssistantOutcome {
  status: AssistantStatus
  strategy: AssistantStrategy | null
  citations: AssistantReadableCitation[]
  error?: AssistantResponse['error']
  historyTruncated?: boolean
}
