import type { AssistantRequest } from '@/types/assistant'
import type { MessageWithError } from '@/types/retrieval'

const length = (text: string) => Array.from(text).length

/** Match the server's Unicode character count and spaced JSON input budget. */
function serializedLength(request: AssistantRequest): number {
  const history = request.history
    .map((m) => `{"role": ${JSON.stringify(m.role)}, "content": ${JSON.stringify(m.content)}}`)
    .join(', ')
  return length(`{"query": ${JSON.stringify(request.query)}, "history": [${history}]}`)
}

export function serializeAssistantRequest(query: string, messages: MessageWithError[]) {
  const request: AssistantRequest = { query: query.trim(), history: [] }
  const pairs: AssistantRequest['history'][] = []
  for (let i = 1; i < messages.length; i++) {
    const user = messages[i - 1]
    const answer = messages[i]
    if (
      user.role === 'user' &&
      user.workflow === 'assistant' &&
      answer.role === 'assistant' &&
      answer.workflow === 'assistant' &&
      answer.assistantOutcome &&
      answer.assistantOutcome.status !== 'failed' &&
      !answer.isError &&
      !answer.isAborted &&
      user.content.trim() &&
      answer.content.trim()
    ) {
      pairs.push([
        { role: 'user', content: user.content },
        { role: 'assistant', content: answer.content }
      ])
    }
  }
  let kept = 0
  for (let i = pairs.length - 1; i >= 0; i--) {
    const pair = pairs[i]
    if (pair.some((m) => length(m.content) > 4096) || request.history.length >= 32) break
    const candidate = { ...request, history: [...pair, ...request.history] }
    if (serializedLength(candidate) > 24000) break
    request.history = candidate.history
    kept++
  }
  return { request, historyTruncated: kept < pairs.length }
}
