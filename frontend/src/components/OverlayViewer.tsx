'use client'

import { useEffect, useRef, useState } from 'react'
import type { CompositionScores } from '@/types/api'

type OverlayType = 'original' | 'rot' | 'gr' | 'dynamics' | 'lines'

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

function centroidColor(score: number): string {
  if (score < 0.25) return 'rgba(0, 220, 0, 0.9)'
  if (score <= 0.5) return 'rgba(255, 200, 0, 0.9)'
  return 'rgba(220, 40, 40, 0.9)'
}

function drawCrosshair(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  color: string,
  arm = 15,
  radius = 8,
  lineWidth = 2,
) {
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = lineWidth
  ctx.beginPath()
  ctx.moveTo(x - arm, y)
  ctx.lineTo(x + arm, y)
  ctx.stroke()
  ctx.beginPath()
  ctx.moveTo(x, y - arm)
  ctx.lineTo(x, y + arm)
  ctx.stroke()
  ctx.beginPath()
  ctx.arc(x, y, radius, 0, Math.PI * 2)
  ctx.stroke()
  ctx.restore()
}

function drawRotGrid(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  cx: number,
  cy: number,
  score: number,
) {
  ctx.save()
  ctx.strokeStyle = 'rgba(0, 255, 255, 0.6)'
  ctx.lineWidth = 2
  for (const frac of [1 / 3, 2 / 3]) {
    ctx.beginPath(); ctx.moveTo(w * frac, 0); ctx.lineTo(w * frac, h); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(0, h * frac); ctx.lineTo(w, h * frac); ctx.stroke()
  }
  ctx.strokeStyle = 'rgba(0, 255, 255, 0.78)'
  for (const fx of [1 / 3, 2 / 3]) {
    for (const fy of [1 / 3, 2 / 3]) {
      ctx.beginPath(); ctx.arc(w * fx, h * fy, 8, 0, Math.PI * 2); ctx.stroke()
    }
  }
  ctx.restore()
  drawCrosshair(ctx, cx, cy, centroidColor(score))
}

function drawGrGrid(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  cx: number,
  cy: number,
  score: number,
) {
  const phi = 0.618
  ctx.save()
  ctx.strokeStyle = 'rgba(255, 0, 255, 0.6)'
  ctx.lineWidth = 2
  for (const frac of [1 - phi, phi]) {
    ctx.beginPath(); ctx.moveTo(w * frac, 0); ctx.lineTo(w * frac, h); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(0, h * frac); ctx.lineTo(w, h * frac); ctx.stroke()
  }
  ctx.strokeStyle = 'rgba(255, 0, 255, 0.78)'
  for (const fx of [1 - phi, phi]) {
    for (const fy of [1 - phi, phi]) {
      ctx.beginPath(); ctx.arc(w * fx, h * fy, 8, 0, Math.PI * 2); ctx.stroke()
    }
  }
  ctx.restore()
  drawCrosshair(ctx, cx, cy, centroidColor(score))
}

function drawDynamics(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  cx: number,
  cy: number,
  quadrants: CompositionScores['visual_weight_quadrants'],
) {
  const mw = Math.floor(w / 2)
  const mh = Math.floor(h / 2)
  const regions: Record<keyof typeof quadrants, [number, number, number, number]> = {
    top_left: [0, 0, mw, mh],
    top_right: [mw, 0, w, mh],
    bottom_left: [0, mh, mw, h],
    bottom_right: [mw, mh, w, h],
  }
  const vals = Object.values(quadrants)
  const wmin = Math.min(...vals)
  const wmax = Math.max(...vals)
  const range = wmax !== wmin ? wmax - wmin : 1

  ctx.save()
  for (const key of Object.keys(regions) as (keyof typeof quadrants)[]) {
    const [x0, y0, x1, y1] = regions[key]
    const t = (quadrants[key] - wmin) / range
    const r = Math.round(t * 255)
    const b = Math.round((1 - t) * 255)
    ctx.fillStyle = `rgba(${r}, 0, ${b}, 0.3)`
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)'
    ctx.lineWidth = 1
    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0)
  }
  ctx.restore()
  drawCrosshair(ctx, cx, cy, 'rgba(255, 255, 255, 0.9)', 15, 12, 3)
}

function drawLines(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  cx: number,
  cy: number,
  angles: number[],
  converges: boolean,
) {
  if (!angles.length) return
  const color = converges ? 'rgba(255, 0, 255, 0.86)' : 'rgba(0, 255, 255, 0.86)'
  const far = Math.max(w, h) * 2

  ctx.save()
  ctx.beginPath(); ctx.rect(0, 0, w, h); ctx.clip()
  ctx.strokeStyle = color
  ctx.lineWidth = 3
  for (const angleDeg of angles) {
    const rad = (angleDeg * Math.PI) / 180
    const dx = Math.cos(rad)
    const dy = Math.sin(rad)
    ctx.beginPath()
    ctx.moveTo(cx - dx * far, cy - dy * far)
    ctx.lineTo(cx + dx * far, cy + dy * far)
    ctx.stroke()
  }
  ctx.restore()
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const TABS: { id: OverlayType; label: string }[] = [
  { id: 'original', label: 'Original' },
  { id: 'rot', label: 'Rule of Thirds' },
  { id: 'gr', label: 'Golden Ratio' },
  { id: 'dynamics', label: 'Subject Dynamics' },
  { id: 'lines', label: 'Leading Lines' },
]

interface Props {
  imageUrl: string
  comp: CompositionScores
}

export default function OverlayViewer({ imageUrl, comp }: Props) {
  const [active, setActive] = useState<OverlayType>('original')
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [img, setImg] = useState<HTMLImageElement | null>(null)

  // Load image once (or when URL changes)
  useEffect(() => {
    const image = new window.Image()
    image.onload = () => setImg(image)
    image.src = imageUrl
    return () => { image.onload = null }
  }, [imageUrl])

  // Redraw whenever tab or image changes
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    ctx.drawImage(img, 0, 0)

    if (active === 'original') return

    const w = canvas.width
    const h = canvas.height
    const cx = comp.saliency_centroid_x * w
    const cy = comp.saliency_centroid_y * h

    if (active === 'rot') drawRotGrid(ctx, w, h, cx, cy, comp.rot_alignment_score)
    else if (active === 'gr') drawGrGrid(ctx, w, h, cx, cy, comp.golden_ratio_alignment_score)
    else if (active === 'dynamics') drawDynamics(ctx, w, h, cx, cy, comp.visual_weight_quadrants)
    else if (active === 'lines') drawLines(ctx, w, h, cx, cy, comp.dominant_line_angles, comp.leading_lines_converge_to_subject)
  }, [img, active, comp])

  return (
    <div className="space-y-3">
      {/* Tab buttons */}
      <div className="flex gap-1.5 flex-wrap">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setActive(id)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              active === id
                ? 'bg-indigo-600 text-white'
                : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Canvas (always rendered — draws base image for Original tab too) */}
      <canvas
        ref={canvasRef}
        className="w-full rounded-lg object-contain max-h-[70vh]"
        style={{ display: img ? 'block' : 'none' }}
      />

      {/* Legend */}
      {active === 'dynamics' && (
        <p className="text-xs text-zinc-500">
          Blue = low visual weight · Red = high visual weight · White crosshair = subject centroid
        </p>
      )}
      {(active === 'rot' || active === 'gr') && (
        <p className="text-xs text-zinc-500">
          Dot colour: <span className="text-green-400">green</span> = well-placed ·{' '}
          <span className="text-yellow-400">yellow</span> = fair ·{' '}
          <span className="text-red-400">red</span> = off power points
        </p>
      )}
      {active === 'lines' && (
        <p className="text-xs text-zinc-500">
          {comp.leading_lines_converge_to_subject
            ? 'Magenta lines converge toward the subject'
            : 'Cyan lines — dominant angles detected (not converging)'}
        </p>
      )}
    </div>
  )
}
