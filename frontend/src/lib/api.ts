import type { AnalyseResponse } from '@/types/api'

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
