'use client'

import { useState } from 'react'
import type { ReactNode } from 'react'
import type { AnalyseResponse, QualityTier } from '@/types/api'
import OverlayViewer from '@/components/OverlayViewer'

type Tab = 'composition' | 'technical' | 'report'

const METRIC_INFO: Record<string, { label: string; description: string }> = {
  // Technical
  brisque:        { label: 'Overall sharpness quality', description: 'Detects blur and compression artefacts — lower is better (0 = pristine).' },
  sharpness:      { label: 'Edge sharpness',            description: 'Measures how crisp the edges and fine details are.' },
  noise:          { label: 'Noise level',               description: 'Amount of grain/noise in the image — lower means cleaner.' },
  highlights:     { label: 'Blown highlights',          description: 'Percentage of pixels that are pure white with no detail.' },
  shadows:        { label: 'Crushed shadows',           description: 'Percentage of pixels that are pure black with no detail.' },
  dynamic_range:  { label: 'Tonal range',               description: 'Gap between the brightest and darkest areas, in exposure stops.' },
  contrast:       { label: 'Contrast',                  description: 'How much variation there is in brightness across the image.' },
  histogram_mean: { label: 'Average brightness',        description: 'Overall exposure level — 128 is a balanced mid-tone.' },
  nima:           { label: 'Aesthetic appeal',          description: 'AI estimate of how visually pleasing the image is (0–10).' },
  clip_iqa:       { label: 'Perceptual quality',        description: 'How natural and high-quality the image looks to an AI visual model (0–1).' },
  musiq:          { label: 'Overall image quality',     description: 'Holistic quality score trained on human ratings (0–100).' },
  niqe:           { label: 'Distortion / artefacts',    description: 'Detects compression artefacts and distortions — lower is better.' },
  // Composition
  best_alignment: { label: 'Subject placement',         description: 'How well the main subject aligns to classic compositional grids.' },
  negative_space: { label: 'Breathing room',            description: 'How much empty space surrounds the subject.' },
  line_pattern:   { label: 'Line type',                 description: 'Dominant type of lines detected (e.g. converging, parallel, curved).' },
  visual_weight:  { label: 'Balance',                   description: 'How evenly the visual mass is distributed across the frame.' },
  leading_lines:  { label: 'Lines lead to subject',     description: 'Whether the detected lines guide the eye toward the main subject.' },
  dominant_angles:{ label: 'Line angles',               description: 'The primary directions of lines in the image (in degrees).' },
  horizon_tilt:   { label: 'Horizon tilt',              description: 'How many degrees the horizon is off-level.' },
  color_harmony:  { label: 'Colour harmony',            description: 'How well the colours in the image work together.' },
}

function MetricLabel({ id }: { id: string }) {
  const [open, setOpen] = useState(false)
  const info = METRIC_INFO[id]
  if (!info) return null
  return (
    <span className="flex items-center gap-1">
      <span>{info.label}</span>
      <span className="relative inline-flex group">
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          className="w-4 h-4 rounded-full bg-zinc-700 text-zinc-400 text-[9px] font-bold inline-flex items-center justify-center hover:bg-indigo-600 hover:text-white transition-colors flex-shrink-0"
          aria-label={info.description}
        >
          i
        </button>
        <span
          className={`absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-52 p-2 text-xs text-zinc-200 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl z-10 transition-opacity pointer-events-none ${
            open ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
          }`}
        >
          {info.description}
        </span>
      </span>
    </span>
  )
}

const TIER_COLORS: Record<QualityTier['overall'], string> = {
  excellent: 'bg-emerald-900/60 text-emerald-300 border-emerald-700',
  good: 'bg-teal-900/60 text-teal-300 border-teal-700',
  average: 'bg-yellow-900/60 text-yellow-300 border-yellow-700',
  poor: 'bg-orange-900/60 text-orange-300 border-orange-700',
  terrible: 'bg-red-900/60 text-red-300 border-red-700',
}

function TierBadge({ tier }: { tier: QualityTier['overall'] }) {
  return (
    <span className={`inline-block px-2 py-0.5 text-xs font-medium rounded border capitalize ${TIER_COLORS[tier]}`}>
      {tier}
    </span>
  )
}

function ProgressBar({ value, label }: { value: number; label: string }) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100)
  return (
    <div>
      <div className="flex justify-between text-xs text-zinc-400 mb-1">
        <span>{label}</span>
        <span>{pct}%</span>
      </div>
      <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-indigo-500 rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

function MetricCard({
  label,
  value,
  tier,
}: {
  label: ReactNode
  value: string
  tier?: QualityTier['overall']
}) {
  return (
    <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-3 space-y-1">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="text-sm font-mono font-medium">{value}</p>
      {tier && <TierBadge tier={tier} />}
    </div>
  )
}

function ReportSection({ title, content }: { title: string; content: string }) {
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-zinc-300 capitalize">{title}</h3>
      <div className="text-sm text-zinc-400 space-y-1">
        {String(content).split('\n').map((line, i) => (
          <p key={i}>{line}</p>
        ))}
      </div>
    </div>
  )
}

export default function AnalysisResult({ result, imageUrl }: { result: AnalyseResponse; imageUrl: string }) {
  const [tab, setTab] = useState<Tab>('report')
  const { exif, quality_tier, technical, composition, report } = result

  const exifChips: { label: string; value: string }[] = [
    exif.camera_make && exif.camera_model
      ? { label: 'Camera', value: `${exif.camera_make} ${exif.camera_model}` }
      : null,
    exif.lens_model ? { label: 'Lens', value: exif.lens_model } : null,
    exif.iso != null ? { label: 'ISO', value: String(exif.iso) } : null,
    exif.shutter_speed ? { label: 'Shutter', value: exif.shutter_speed } : null,
    exif.aperture != null ? { label: 'f/', value: String(exif.aperture) } : null,
    exif.focal_length != null ? { label: 'FL', value: `${exif.focal_length}mm` } : null,
  ].filter(Boolean) as { label: string; value: string }[]

  const tabs: { id: Tab; label: string }[] = [
    { id: 'report', label: 'Report' },
    { id: 'composition', label: 'Composition' },
    { id: 'technical', label: 'Technical' },
  ]

  return (
    <div className="space-y-6">
      {/* Overlay viewer */}
      <OverlayViewer imageUrl={imageUrl} comp={composition} />

      {/* Header: quality tier + summary */}
      <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-5 space-y-3">
        <div className="flex items-center gap-3 flex-wrap">
          <TierBadge tier={quality_tier.overall} />
          <span className="text-xs text-zinc-500">Overall quality</span>
          {quality_tier.composition_tier && (
            <>
              <TierBadge tier={quality_tier.composition_tier} />
              <span className="text-xs text-zinc-500">Composition</span>
            </>
          )}
        </div>
        <p className="text-sm text-zinc-300 leading-relaxed">{report.summary}</p>
      </div>

      {/* EXIF strip */}
      {exifChips.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {exifChips.map(({ label, value }) => (
            <span
              key={label}
              className="inline-flex items-center gap-1 px-2.5 py-1 bg-zinc-800 border border-zinc-700 rounded-full text-xs"
            >
              <span className="text-zinc-500">{label}</span>
              <span className="text-zinc-200">{value}</span>
            </span>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div>
        <div className="flex gap-1 border-b border-zinc-800 mb-5">
          {tabs.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`px-4 py-2 text-sm font-medium transition-colors border-b-2 -mb-px ${
                tab === id
                  ? 'border-indigo-500 text-indigo-400'
                  : 'border-transparent text-zinc-500 hover:text-zinc-300'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Composition tab */}
        {tab === 'composition' && (
          <div className="space-y-5">
            {/* Scene type badge */}
            {composition.scene_type && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-zinc-500">Scene</span>
                <span className="inline-block px-2 py-0.5 text-xs font-medium rounded border border-indigo-700 bg-indigo-900/40 text-indigo-300 capitalize">
                  {composition.scene_type}
                </span>
              </div>
            )}
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <MetricCard
                label={<MetricLabel id="best_alignment" />}
                value={composition.best_alignment === 'rule_of_thirds' ? 'Rule of Thirds' : 'Golden Ratio'}
              />
              <MetricCard
                label={<MetricLabel id="negative_space" />}
                value={composition.negative_space_ratio != null ? `${Math.round(composition.negative_space_ratio * 100)}%` : '—'}
              />
              <MetricCard
                label={<MetricLabel id="line_pattern" />}
                value={composition.line_pattern}
              />
              <MetricCard
                label={<MetricLabel id="visual_weight" />}
                value={composition.visual_weight_balance != null ? composition.visual_weight_balance.toFixed(2) : '—'}
              />
              <MetricCard
                label={<MetricLabel id="leading_lines" />}
                value={composition.leading_lines_converge_to_subject ? 'Yes' : 'No'}
              />
              {composition.dominant_line_angles != null && composition.dominant_line_angles.length > 0 && (
                <MetricCard
                  label={<MetricLabel id="dominant_angles" />}
                  value={composition.dominant_line_angles.map(a => `${Math.round(a)}°`).join(', ')}
                />
              )}
              {composition.horizon_tilt_degrees != null ? (
                <MetricCard
                  label={<MetricLabel id="horizon_tilt" />}
                  value={`${composition.horizon_tilt_degrees >= 0 ? '+' : ''}${composition.horizon_tilt_degrees.toFixed(1)}°`}
                />
              ) : (
                <MetricCard label={<MetricLabel id="horizon_tilt" />} value="—" />
              )}
              {composition.color_harmony_type && (
                <MetricCard
                  label={<MetricLabel id="color_harmony" />}
                  value={composition.color_harmony_type}
                />
              )}
            </div>
            {composition.color_harmony_score != null && composition.color_harmony_type && composition.color_harmony_type !== 'complex' && (
              <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-4">
                <ProgressBar value={composition.color_harmony_score} label={`${composition.color_harmony_type} harmony score`} />
              </div>
            )}

            <div className="space-y-3 bg-zinc-900 border border-zinc-800 rounded-xl p-4">
              <h3 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Symmetry</h3>
              <ProgressBar value={composition.symmetry_horizontal} label="Horizontal" />
              <ProgressBar value={composition.symmetry_vertical} label="Vertical" />
            </div>
          </div>
        )}

        {/* Technical tab */}
        {tab === 'technical' && (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <MetricCard
              label={<MetricLabel id="brisque" />}
              value={technical.brisque != null ? technical.brisque.toFixed(1) : '—'}
              tier={quality_tier.brisque_tier}
            />
            <MetricCard
              label={<MetricLabel id="sharpness" />}
              value={technical.sharpness_laplacian != null ? technical.sharpness_laplacian.toFixed(1) : '—'}
              tier={quality_tier.sharpness_tier}
            />
            <MetricCard
              label={<MetricLabel id="noise" />}
              value={technical.noise_sigma != null ? technical.noise_sigma.toFixed(2) : '—'}
              tier={quality_tier.noise_tier}
            />
            <MetricCard
              label={<MetricLabel id="highlights" />}
              value={technical.exposure_clipped_highlights_pct != null ? `${technical.exposure_clipped_highlights_pct.toFixed(2)}%` : '—'}
              tier={quality_tier.exposure_tier}
            />
            <MetricCard
              label={<MetricLabel id="shadows" />}
              value={technical.exposure_clipped_shadows_pct != null ? `${technical.exposure_clipped_shadows_pct.toFixed(2)}%` : '—'}
            />
            <MetricCard
              label={<MetricLabel id="dynamic_range" />}
              value={technical.dynamic_range_stops != null ? `${technical.dynamic_range_stops.toFixed(1)} stops` : '—'}
            />
            <MetricCard
              label={<MetricLabel id="contrast" />}
              value={technical.contrast_rms != null ? technical.contrast_rms.toFixed(3) : '—'}
            />
            <MetricCard
              label={<MetricLabel id="histogram_mean" />}
              value={technical.histogram_mean != null ? technical.histogram_mean.toFixed(1) : '—'}
            />
            {technical.nima_aesthetic != null && (
              <MetricCard
                label={<MetricLabel id="nima" />}
                value={technical.nima_aesthetic.toFixed(2)}
              />
            )}
            {technical.clip_iqa != null && (
              <MetricCard
                label={<MetricLabel id="clip_iqa" />}
                value={technical.clip_iqa.toFixed(3)}
              />
            )}
            {technical.musiq != null && (
              <MetricCard
                label={<MetricLabel id="musiq" />}
                value={technical.musiq.toFixed(1)}
              />
            )}
            {technical.niqe != null && (
              <MetricCard
                label={<MetricLabel id="niqe" />}
                value={technical.niqe.toFixed(2)}
              />
            )}
          </div>
        )}

        {/* Report tab */}
        {tab === 'report' && (
          <div className="space-y-6">
            {report.composition && (
              <ReportSection title="Composition" content={report.composition} />
            )}
            {report.aesthetics && (
              <ReportSection title="Aesthetics" content={report.aesthetics} />
            )}
            {report.technical && (
              <ReportSection title="Technical" content={report.technical} />
            )}
            {report.improvements && (
              <ReportSection title="Improvements" content={report.improvements} />
            )}
            {report.editing && (
              <ReportSection title="Editing" content={report.editing} />
            )}
            {report.inspiration && (
              <ReportSection title="Inspiration" content={report.inspiration} />
            )}
          </div>
        )}
      </div>
    </div>
  )
}
