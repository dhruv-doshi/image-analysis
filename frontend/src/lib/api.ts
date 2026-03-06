import type { AnalyseResponse, AnalyseStreamEvent, CompositionScores, ExifData, QualityTier, TechnicalScores } from '@/types/api'

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export async function analyseImage(file: File, features = 'full'): Promise<AnalyseResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('features', features)

  const res = await fetch(`${API_URL}/analyse`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(body.detail ?? `Request failed with status ${res.status}`)
  }

  return res.json() as Promise<AnalyseResponse>
}

export type MetricsPayload = {
  exif: ExifData
  quality_tier: QualityTier
  technical: TechnicalScores
  composition: CompositionScores
}

export async function analyseImageStream(
  file: File,
  features = 'full',
  onMetrics: (data: MetricsPayload) => void,
  onChunk: (text: string) => void,
  onReport: (report: import('@/types/api').AnalysisReport) => void,
  onDone: () => void,
): Promise<void> {
  const form = new FormData()
  form.append('file', file)
  form.append('features', features)

  const res = await fetch(`${API_URL}/analyse/stream`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(body.detail ?? `Request failed with status ${res.status}`)
  }

  const reader = res.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parts = buffer.split('\n\n')
    buffer = parts.pop() ?? ''
    for (const part of parts) {
      if (!part.startsWith('data: ')) continue
      // Parse the SSE frame — skip silently if malformed (e.g. keepalive pings)
      let event: AnalyseStreamEvent
      try {
        event = JSON.parse(part.slice(6)) as AnalyseStreamEvent
      } catch (e) {
        if (e instanceof SyntaxError) continue
        throw e
      }
      // Process event outside the frame-parse catch so callback errors propagate
      if (event.type === 'metrics') {
        const { type: _t, ...metrics } = event
        onMetrics(metrics as MetricsPayload)
      } else if (event.type === 'chunk') {
        onChunk(event.text)
      } else if (event.type === 'report') {
        onReport(event.report)
      } else if (event.type === 'done') {
        onDone()
      } else if (event.type === 'error') {
        throw new Error(event.message)
      }
    }
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`, { cache: 'no-store' })
    if (!res.ok) return false
    const body = await res.json()
    return body.models_loaded === true
  } catch {
    return false
  }
}
