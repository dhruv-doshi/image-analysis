'use client'

import { useEffect, useRef, useState } from 'react'
import { analyseImageStream, checkHealth } from '@/lib/api'
import type { AnalyseResponse, AnalysisReport } from '@/types/api'
import type { MetricsPayload } from '@/lib/api'
import AnalysisResult from '@/components/AnalysisResult'

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<AnalyseResponse | null>(null)
  const [partialMetrics, setPartialMetrics] = useState<MetricsPayload | null>(null)
  const [streamingReport, setStreamingReport] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [backendReady, setBackendReady] = useState<boolean | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const reportBufferRef = useRef('')
  const partialMetricsRef = useRef<MetricsPayload | null>(null)

  useEffect(() => {
    checkHealth().then(setBackendReady)
  }, [])

  function handleFile(f: File) {
    setFile(f)
    setResult(null)
    setPartialMetrics(null)
    setStreamingReport(false)
    setError(null)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0]
    if (f) handleFile(f)
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) handleFile(f)
  }

  function onDragOver(e: React.DragEvent) {
    e.preventDefault()
    setDragging(true)
  }

  function onDragLeave() {
    setDragging(false)
  }

  async function onAnalyse() {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    setPartialMetrics(null)
    partialMetricsRef.current = null
    setStreamingReport(false)
    reportBufferRef.current = ''

    try {
      await analyseImageStream(
        file,
        'full',
        (metrics) => {
          partialMetricsRef.current = metrics
          setPartialMetrics(metrics)
          setLoading(false)
          setStreamingReport(true)
        },
        (chunk) => {
          reportBufferRef.current += chunk
        },
        () => {
          setStreamingReport(false)
          try {
            let buf = reportBufferRef.current.trim()
            if (buf.startsWith('```')) {
              const lines = buf.split('\n')
              lines.shift()
              const ci = lines.lastIndexOf('```')
              if (ci !== -1) lines.splice(ci)
              buf = lines.join('\n').trim()
            }
            // JSON forbids leading-plus numbers (+15 → 15)
            buf = buf.replace(/:\s*\+(\d)/g, ': $1')
            // Missing comma after ] or } before next "key" field
            buf = buf.replace(/([}\]])\s*\n(\s*"[a-z_]+")/g, '$1,\n$2')
            // Trailing commas before closing brace/bracket
            buf = buf.replace(/,\s*([}\]])/g, '$1')
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            let parsed: any
            try {
              parsed = JSON.parse(buf)
            } catch (e) {
              console.error('JSON parse failed:', e, '\nBuffer:', buf.slice(0, 500))
              throw e
            }
            // LLM sometimes returns arrays or objects for string fields — coerce to string
            const reportKeys = ['summary', 'composition', 'aesthetics', 'technical', 'improvements', 'editing', 'inspiration'] as const
            for (const key of reportKeys) {
              if (Array.isArray(parsed[key])) parsed[key] = parsed[key].join('\n')
              else if (parsed[key] !== null && typeof parsed[key] === 'object') parsed[key] = Object.entries(parsed[key]).map(([k, v]) => `${k}: ${v}`).join('\n')
            }
            const report = parsed as AnalysisReport
            const m = partialMetricsRef.current
            if (m) setResult({ ...m, report })
          } catch {
            setError('Failed to parse analysis report')
          }
        },
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
      setLoading(false)
      setStreamingReport(false)
    }
  }

  // Build the display result: full result if done, or partial with stub report while streaming
  const displayResult: AnalyseResponse | null = result ?? (
    partialMetrics
      ? {
          ...partialMetrics,
          report: {
            summary: streamingReport ? 'Generating report…' : 'Parsing report…',
          },
        }
      : null
  )

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">FrameIQ</h1>
          <p className="text-xs text-zinc-500 mt-0.5">AI-powered photo analysis</p>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span
            className={`inline-block w-2 h-2 rounded-full ${
              backendReady === null
                ? 'bg-zinc-600'
                : backendReady
                ? 'bg-emerald-500'
                : 'bg-red-500'
            }`}
          />
          <span className="text-zinc-400">
            {backendReady === null ? 'Checking…' : backendReady ? 'Backend ready' : 'Backend offline'}
          </span>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-10 space-y-8">
        {/* Upload zone */}
        <div
          className={`relative border-2 border-dashed rounded-xl transition-colors cursor-pointer ${
            dragging
              ? 'border-indigo-500 bg-indigo-950/30'
              : 'border-zinc-700 hover:border-zinc-500 bg-zinc-900/50'
          }`}
          onClick={() => inputRef.current?.click()}
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            className="hidden"
            onChange={onInputChange}
          />

          {preview ? (
            <div className="p-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={preview}
                alt="Preview"
                className="max-h-80 mx-auto rounded-lg object-contain"
              />
              <p className="text-center text-xs text-zinc-500 mt-2">
                {file?.name} — click or drop to change
              </p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-16 px-4 text-center">
              <svg
                className="w-12 h-12 text-zinc-600 mb-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
                />
              </svg>
              <p className="text-zinc-300 font-medium">Drop a photo here</p>
              <p className="text-zinc-500 text-sm mt-1">or click to browse — JPEG / PNG, max 20 MB</p>
            </div>
          )}
        </div>

        {/* Analyse button */}
        <div className="flex justify-center">
          <button
            onClick={onAnalyse}
            disabled={!file || loading}
            className="px-8 py-3 rounded-lg font-medium text-sm transition-all
              bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-800 disabled:text-zinc-500
              disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8v8H4z"
                  />
                </svg>
                Analysing…
              </span>
            ) : (
              'Analyse'
            )}
          </button>
        </div>

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}

        {/* Results */}
        {displayResult && preview && <AnalysisResult result={displayResult} imageUrl={preview} />}
      </main>
    </div>
  )
}
