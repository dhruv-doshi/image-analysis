# FrameIQ API Contract v1

This is the single source of truth for the FastAPI backend and Next.js frontend.
**Neither agent may deviate from this spec.** Changes must be coordinated through the main terminal.

---

## Base URL

| Environment | URL |
|---|---|
| Local (backend dev) | `http://localhost:8000` |
| Production | `https://frameiq-api.fly.dev` (set via `NEXT_PUBLIC_API_URL` env var in Vercel) |

---

## Endpoints

### `GET /health`

Used by Fly.io health checks and the frontend to confirm the backend is ready.

**Response `200`**
```json
{
  "status": "ok",
  "models_loaded": true
}
```

---

### `POST /analyse`

**Request** — `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | binary (JPEG/PNG) | yes | Max 20MB |
| `features` | string | no | Comma-separated. Default: `"full"`. Options: `composition`, `aesthetics`, `technical`, `improvements`, `editing`, `inspiration`, `full` |

**Response `200`** — `application/json`

```json
{
  "exif": {
    "camera_make": "Canon",
    "camera_model": "EOS R5",
    "iso": 400,
    "shutter_speed": "1/125",
    "aperture": 2.8,
    "focal_length": 50.0,
    "lens_model": "RF 50mm f/1.2L USM",
    "image_width": 4000,
    "image_height": 6000
  },
  "quality_tier": {
    "overall": "good",
    "brisque_tier": "excellent",
    "sharpness_tier": "average",
    "noise_tier": "good",
    "exposure_tier": "excellent"
  },
  "technical": {
    "brisque": 42.3,
    "nima_aesthetic": 5.6,
    "clip_iqa": 0.72,
    "sharpness_laplacian": 312.4,
    "sharpness_regional": {
      "top_left": 280.1,
      "top_right": 340.2,
      "bottom_left": 290.5,
      "bottom_right": 330.0
    },
    "noise_sigma": 4.2,
    "exposure_clipped_highlights_pct": 1.2,
    "exposure_clipped_shadows_pct": 0.8,
    "histogram_mean": 128.5,
    "histogram_std": 45.2,
    "dynamic_range_stops": 4.1,
    "contrast_rms": 0.35
  },
  "composition": {
    "saliency_centroid_x": 0.33,
    "saliency_centroid_y": 0.40,
    "rot_alignment_score": 0.12,
    "golden_ratio_alignment_score": 0.18,
    "best_alignment": "rule_of_thirds",
    "negative_space_ratio": 0.65,
    "visual_weight_quadrants": {
      "top_left": 0.25,
      "top_right": 0.35,
      "bottom_left": 0.20,
      "bottom_right": 0.20
    },
    "visual_weight_balance": 1.75,
    "symmetry_horizontal": 0.82,
    "symmetry_vertical": 0.71,
    "dominant_line_angles": [45.0, 135.0],
    "leading_lines_converge_to_subject": true,
    "line_pattern": "diagonal"
  },
  "report": {
    "summary": "2–3 sentence overall assessment, always present.",
    "composition": "Composition critique. null if not requested.",
    "aesthetics": "Mood, colour harmony, visual impact. null if not requested.",
    "technical": "Technical quality with EXIF context. null if not requested.",
    "improvements": "3–5 ranked shooting/compositional tips. null if not requested.",
    "editing": "Lightroom/Capture One/Darktable adjustments. null if not requested.",
    "inspiration": "2–3 reference photographers or movements. null if not requested."
  }
}
```

**All `exif` fields are nullable** — omitted if not present in the image.
**All `report` fields except `summary` are nullable** — omitted if not in `features`.
**`nima_aesthetic` and `clip_iqa` in `technical` are nullable** — model may fail gracefully.

---

### Error Responses

| Status | When |
|---|---|
| `400` | File is not JPEG or PNG, or exceeds 20MB |
| `422` | Missing required `file` field |
| `500` | Pipeline failure (e.g. missing API key, model error) |

```json
{ "detail": "Human-readable error message" }
```

---

## CORS

Backend must allow:
- `http://localhost:3000` (Next.js dev)
- The production Vercel URL (set via `ALLOWED_ORIGINS` env var, comma-separated)

---

## TypeScript types (for the frontend agent)

```typescript
// Paste this into frontend/src/types/api.ts

export interface ExifData {
  camera_make?: string
  camera_model?: string
  iso?: number
  shutter_speed?: string
  aperture?: number
  focal_length?: number
  lens_model?: string
  image_width?: number
  image_height?: number
}

export interface QualityTier {
  overall: 'excellent' | 'good' | 'average' | 'poor' | 'terrible'
  brisque_tier: 'excellent' | 'good' | 'average' | 'poor' | 'terrible'
  sharpness_tier: 'excellent' | 'good' | 'average' | 'poor' | 'terrible'
  noise_tier: 'excellent' | 'good' | 'average' | 'poor' | 'terrible'
  exposure_tier: 'excellent' | 'good' | 'average' | 'poor' | 'terrible'
}

export interface TechnicalScores {
  brisque: number
  nima_aesthetic?: number
  clip_iqa?: number
  sharpness_laplacian: number
  sharpness_regional: { top_left: number; top_right: number; bottom_left: number; bottom_right: number }
  noise_sigma: number
  exposure_clipped_highlights_pct: number
  exposure_clipped_shadows_pct: number
  histogram_mean: number
  histogram_std: number
  dynamic_range_stops: number
  contrast_rms: number
}

export interface CompositionScores {
  saliency_centroid_x: number
  saliency_centroid_y: number
  rot_alignment_score: number
  golden_ratio_alignment_score: number
  best_alignment: 'rule_of_thirds' | 'golden_ratio'
  negative_space_ratio: number
  visual_weight_quadrants: { top_left: number; top_right: number; bottom_left: number; bottom_right: number }
  visual_weight_balance: number
  symmetry_horizontal: number
  symmetry_vertical: number
  dominant_line_angles: number[]
  leading_lines_converge_to_subject: boolean
  line_pattern: 'diagonal' | 'horizontal' | 'vertical' | 'mixed' | 'none'
}

export interface AnalysisReport {
  summary: string
  composition?: string
  aesthetics?: string
  technical?: string
  improvements?: string
  editing?: string
  inspiration?: string
}

export interface AnalyseResponse {
  exif: ExifData
  quality_tier: QualityTier
  technical: TechnicalScores
  composition: CompositionScores
  report: AnalysisReport
}
```
