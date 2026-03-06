# FrameIQ — Architecture Reference

## 1. Overview

FrameIQ is a FastAPI + Next.js web application that accepts an uploaded photograph and returns an AI-powered analysis covering composition, aesthetics, technical quality, improvement tips, and photographer/style recommendations.

The backend is a **three-layer pipeline**:

1. **Layer 1 — Technical Analysis**: learned IQA metrics (pyiqa) + classical computer-vision metrics (OpenCV, scikit-image).
2. **Layer 2 — Composition Analysis**: salient-object detection (rembg U²-Net) → geometric and perceptual composition scores.
3. **Layer 3 — LLM Synthesis**: all numeric scores serialised into an annotated JSON payload, sent to Claude via OpenRouter; structured `AnalysisReport` returned.

---

## 2. Pipeline Diagram

```
                          ┌─────────────────────────────────────────────────────────────────────┐
                          │                       FastAPI  (api.py)                              │
                          │                                                                      │
  HTTP POST /analyse ────►│  Image bytes                                                         │
  (multipart upload)      │      │                                                               │
                          │      ▼                                                               │
                          │  ┌───────────────────────┐                                          │
                          │  │   Image Loader         │  PIL → RGB → resize (≤1024px) →         │
                          │  │   src/utils/loader.py  │  BGR ndarray + float32 tensor            │
                          │  │                        │  EXIF metadata extracted                 │
                          │  └──────────┬────────────┘                                          │
                          │             │                                                        │
                          │             ▼                                                        │
                          │  ┌───────────────────────┐                                          │
                          │  │  Layer 1 — Technical   │  pyiqa learned metrics                  │
                          │  │  src/analysis/         │  BRISQUE · NIMA · CLIP-IQA+ · MUSIQ     │
                          │  │    technical.py        │  + classical CV sharpness / noise /      │
                          │  │                        │    exposure / dynamic range / contrast   │
                          │  │  Output: TechnicalScores│                                         │
                          │  └──────────┬────────────┘                                          │
                          │             │                                                        │
                          │             ▼                                                        │
                          │  ┌───────────────────────┐                                          │
                          │  │  Layer 2 — Composition │  rembg saliency → centroid              │
                          │  │  src/analysis/         │  RoT / GR alignment · neg. space        │
                          │  │    composition.py      │  symmetry · leading lines               │
                          │  │                        │  horizon tilt · scene class             │
                          │  │  Output: CompositionScores  color harmony                        │
                          │  └──────────┬────────────┘                                          │
                          │             │                                                        │
                          │             ▼                                                        │
                          │  ┌───────────────────────┐                                          │
                          │  │  Layer 3 — LLM Synth   │  Annotated JSON payload →               │
                          │  │  src/llm/              │  Claude (OpenRouter)                    │
                          │  │    synthesizer.py      │  Structured AnalysisReport JSON         │
                          │  │    client.py           │  parsed back                            │
                          │  │  Output: AnalysisReport│                                         │
                          │  └──────────┬────────────┘                                          │
                          │             │                                                        │
                          │             ▼                                                        │
  JSON response ◄─────────│  AnalyseResponse { exif, quality_tier, technical,                   │
                          │                    composition, report }                             │
                          └─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Image Loading (`src/utils/loader.py`)

### `load_image(path) → (ndarray, tensor)`

| Step | Detail |
|------|--------|
| Open | `PIL.Image.open()` → convert to RGB |
| Resize | Long edge capped at `_MAX_ANALYSIS_DIM = 1024 px` using Lanczos resampling |
| BGR array | `np.array(img)[:, :, ::-1]` — OpenCV-compatible BGR uint8 |
| Tensor | `float32` in range [0, 1], shape `(1, 3, H, W)` — pyiqa-compatible |

**Why 1024 px?** ViT-based models (MUSIQ, CLIP-IQA+) scale as O(patches²) in attention. A 16 MP image took 418 s before the cap was introduced.

### `extract_exif(path) → ExifData`

- Reads all EXIF tags via Pillow's `_getexif()`.
- For image dimensions, prefers `ExifImageWidth` / `ExifImageHeight` (original camera dimensions) over the post-resize PIL size.
- Extracts: camera make/model, ISO, shutter speed, aperture (f-number), focal length, lens model.

---

## 4. Layer 1 — Technical Analysis (`src/analysis/technical.py`)

All pyiqa models are instantiated **once at module import** as module-level singletons (`_brisque`, `_nima`, `_clip_iqa`, `_musiq`). Each metric call is wrapped in `try/except` — on failure the field is `NaN` / `None`.

### 4.1 Learned Metrics (pyiqa)

| Field | Model | How it works | Scale | Interpretation |
|-------|-------|--------------|-------|----------------|
| `brisque` | BRISQUE | DCT coefficient statistics fit to a Generalised Gaussian Distribution; deviation from "natural scene statistics" modelled with an SVR | 0–100 | Lower = better. <30 excellent, 30–50 good, 50–65 fair, 65–80 poor |
| `nima_aesthetic` | NIMA (aesthetic) | Inception-ResNet-V2 trained on the AVA dataset; predicts a distribution over 1–10 human ratings; returns the expected score | 1–10 | Higher = better. <5 below average, 7+ strong |
| `clip_iqa+` | CLIP-IQA+ | CLIP vision-language model prompted with quality/distortion text pairs; cosine similarity to "good quality" direction | 0–1 | Higher = better. <0.4 poor, >0.6 good |
| `musiq` | MUSIQ | Multi-scale ViT that patches the image at several resolutions; trained on multiple IQA datasets | 0–100 | Higher = better. Direct perceptual quality score |

### 4.2 Classical CV Metrics

| Field | Computation | Scale / Interpretation |
|-------|-------------|------------------------|
| `sharpness_laplacian` | `Var(Laplacian(grayscale))` — high variance means strong, well-defined edges | <200 blurry; 200–500 moderate; >500 sharp |
| `sharpness_regional` | Same Laplacian variance computed per-quadrant (`top_left`, `top_right`, `bottom_left`, `bottom_right`) | Reveals focus falloff or intentional selective focus |
| `noise_sigma` | `skimage.restoration.estimate_sigma(rgb)` — estimates Gaussian noise std per channel, averaged | <3 clean; 3–8 acceptable; >8 noisy |
| `exposure_clipped_highlights_pct` | % pixels ≥ 250 in grayscale | <2% acceptable; >5% overexposed |
| `exposure_clipped_shadows_pct` | % pixels ≤ 5 in grayscale | <2% acceptable; >5% underexposed |
| `histogram_mean` | Mean grayscale luminance (0–255) | <80 dark; 80–170 well exposed; >200 very bright |
| `histogram_std` | Std dev of grayscale histogram | Width of tonal range |
| `dynamic_range_stops` | `log₂(p99 / p1)` of luminance (percentile-based, avoids spike outliers) | <3 flat/low contrast; 3–5 normal; >5 rich |
| `contrast_rms` | `std(grayscale) / mean(grayscale)` — normalised RMS contrast | <0.2 flat; 0.2–0.5 normal; >0.5 punchy |

---

## 5. Layer 2 — Composition Analysis (`src/analysis/composition.py`)

### 5.1 Saliency Map

`_saliency_map(bgr_image) → float32 [0,1] H×W array`

- Calls **rembg** with U²-Net (`u2net`) to segment the salient foreground.
- Extracts the alpha channel of the RGBA output as the saliency weight.
- Falls back to a uniform map (all ones) on import failure or exception.

### 5.2 Centroid & Alignment

| Field | Computation |
|-------|-------------|
| `saliency_centroid_x/y` | Weighted centre of mass (image moment) of the saliency map; normalised to [0,1] |
| `rot_alignment_score` | Min Euclidean distance from centroid to the 4 Rule-of-Thirds power points `{(1/3,1/3), (1/3,2/3), (2/3,1/3), (2/3,2/3)}`, normalised by max possible distance. 0 = perfect alignment |
| `golden_ratio_alignment_score` | Same, using Golden Ratio power points at φ = 0.618 offsets |
| `best_alignment` | `"rule_of_thirds"` if `rot_score ≤ gr_score`, else `"golden_ratio"` |

### 5.3 Negative Space & Visual Weight

| Field | Computation |
|-------|-------------|
| `negative_space_ratio` | Fraction of pixels below 50% of the max saliency value. >0.5 = minimalist |
| `visual_weight_quadrants` | Per-quadrant sum of saliency divided by total saliency — how attention is distributed across four corners |
| `visual_weight_balance` | `max_quadrant_weight / min_quadrant_weight`. 1.0 = balanced; >3 = heavily concentrated. Capped at 99 to avoid JSON `Infinity` |

### 5.4 Symmetry

| Field | Computation |
|-------|-------------|
| `symmetry_horizontal` | Normalised Cross-Correlation (NCC) of the left half vs the horizontally-flipped right half of the saliency map. 1.0 = perfect left-right symmetry |
| `symmetry_vertical` | NCC of the top half vs the vertically-flipped bottom half. 1.0 = perfect top-bottom symmetry |

### 5.5 Leading Lines

1. **Canny** edge detection on the grayscale image.
2. **HoughLinesP** (`threshold=80`, `minLineLength=100px`) to find line segments.
3. Angles binned into 15° clusters; cluster medians reported.

| Field | Meaning |
|-------|---------|
| `dominant_line_angles` | List of representative angles (degrees) from each cluster |
| `leading_lines_converge_to_subject` | `True` if ≥30% of detected lines pass within 10% of image dimension from the saliency centroid |
| `line_pattern` | `diagonal` / `horizontal` / `vertical` / `mixed` / `none` based on angle distribution |

### 5.6 Horizon Tilt

- Reuses `_detect_lines()` output.
- Filters to **near-horizontal** lines (angle <20° or >160°).
- Takes the **median signed tilt** across those lines.

| Field | Meaning |
|-------|---------|
| `horizon_tilt_degrees` | Positive = clockwise tilt; negative = counter-clockwise; `None` if fewer than 2 horizontal lines detected |

### 5.7 Scene Classification

`scene_type` is one of: `portrait` · `landscape` · `architecture` · `macro` · `general`

| Class | Detection logic |
|-------|-----------------|
| `portrait` | OpenCV Haar cascade face detection; face bounding box > 5% of image area |
| `landscape` | No faces detected; top 20% of rows brighter than bottom 20% by ≥30 luminance; wide aspect ratio (width > height × 1.2) |
| `architecture` | >50% of detected lines are vertical (angle 70°–110°) |
| `macro` | No face detected; focal length >90 mm (from EXIF) OR small image with highly uneven per-quadrant sharpness (max quadrant > 2× mean quadrant) |
| `general` | Fallback when no other class matches |

### 5.8 Color Harmony

1. Downsample image to 100×100 px.
2. Convert to **CIELAB** color space.
3. **K-means** clustering (k=5) to find 5 dominant colors.
4. Convert each cluster center Lab → HSV to extract hue and saturation.

| Field | Meaning |
|-------|---------|
| `dominant_colors` | Top-5 cluster centers as `[L, a, b]` values |
| `color_harmony_type` | Classified by hue relationships (see table below) |
| `color_harmony_score` | 0–1 strength of match to the named template |

**Harmony classification rules:**

| Type | Rule |
|------|------|
| `monochromatic` | All cluster saturations < 20 (near-greyscale) |
| `analogous` | All pairwise hue differences < 30° |
| `complementary` | 2 clusters approximately 180° apart (±30° tolerance) |
| `triadic` | 3 clusters approximately 120° apart (±20° tolerance) |
| `split-complementary` | One hue + two others approximately 150° away (±20° tolerance) |
| `complex` | None of the above |

---

## 6. Layer 3 — LLM Synthesis (`src/llm/synthesizer.py` + `src/llm/client.py`)

### 6.1 Quality Tier Computation

Before calling the LLM, `_compute_quality_tier()` converts raw metrics to **ordinal scores** (0 = excellent … 4 = terrible):

| Metric | Ordinal thresholds |
|--------|--------------------|
| BRISQUE | <30→0, <50→1, <65→2, <80→3, else 4 |
| Sharpness | Adjusted: `adj = sharpness − noise² × 20`; >500→0, ≥200→2, ≥80→3, else 4 |
| Noise | <3→0, ≤8→2, ≤15→3, else 4 |
| Exposure | Counts severe (hl>15%, sh>20%, mean<40 or >230) and moderate thresholds |
| NIMA | <4→4, <5→3, <6→2, <7→1, else 0 |
| CLIP-IQA+ | <0.3→4, <0.4→3, <0.5→2, <0.6→1, else 0 |

**Weighted average**: `(BRISQUE×3 + sharpness×2 + noise×2 + exposure×1 + NIMA×2 + CLIP×1) / 11`

The resulting float maps to a quality tier string sent to the LLM for tone anchoring.

### 6.2 Payload Construction

`_build_payload()` serialises all scores into a JSON **user message**. Every numeric field is wrapped with both a `value` and a `scale` description string so the LLM receives human-readable interpretation bounds alongside each number.

Example snippet:
```json
{
  "brisque": {
    "value": 28.4,
    "scale": "0–100, lower is better; <30 excellent, 30–50 good, 50–65 fair, >65 poor"
  }
}
```

### 6.3 LLM Call

| Aspect | Detail |
|--------|--------|
| Client | OpenAI-compatible SDK pointed at OpenRouter base URL |
| Auth | `OPENROUTER_API_KEY` environment variable |
| Model | `LLM_MODEL` env var (default: `anthropic/claude-haiku-4-5-20251001`) |
| System prompt | `prompts/system.md` — expert photography critic persona, quality-tier tone anchoring table, genre-aware guidance sections (portrait / landscape / architecture / macro) |
| Sync call | `synthesise()` → returns complete `AnalysisReport` |
| Streaming | `synthesise_stream()` → SSE generator yielding text chunks; guards `if not chunk.choices: continue` to handle OpenRouter keepalive empty-choices chunks |

### 6.4 JSON Sanitisation (`_sanitise_llm_json`)

The LLM output is cleaned before `json.loads()`:

1. Strip markdown code fences (` ```json … ``` `).
2. Remove leading-plus signs from numbers (`+15` → `15`).
3. Insert missing commas between adjacent JSON fields.
4. Remove trailing commas before `}` or `]`.

---

## 7. API Entry Point (`api.py`)

| Aspect | Detail |
|--------|--------|
| Framework | FastAPI |
| CORS | All origins allowed (development default) |
| Lifespan | `@asynccontextmanager` startup handler pre-warms pyiqa models and rembg on first request |
| `POST /analyse` | Synchronous — returns complete `AnalyseResponse` JSON |
| `POST /analyse/stream` | SSE streaming — emits `metrics` event (JSON with exif + technical + composition), then streams LLM tokens as `report_chunk` events, ends with `done` |
| `_json_safe()` | Recursively replaces `nan` / `inf` in metric dicts before SSE emission (JSON spec does not allow these values) |

---

## 8. Data Models (`src/models.py`)

All models are **Pydantic** `BaseModel`.

### `ExifData`
Camera make, model, ISO, shutter speed, aperture (f-number), focal length, lens model, original image dimensions.

### `TechnicalScores` (13 fields)
| Group | Fields |
|-------|--------|
| Learned | `brisque`, `nima_aesthetic`, `clip_iqa_plus`, `musiq` |
| Sharpness | `sharpness_laplacian`, `sharpness_regional` |
| Noise | `noise_sigma` |
| Exposure | `exposure_clipped_highlights_pct`, `exposure_clipped_shadows_pct`, `histogram_mean`, `histogram_std` |
| Tonal | `dynamic_range_stops`, `contrast_rms` |

### `CompositionScores` (17 fields)
Centroid, alignment scores + best_alignment, negative_space_ratio, visual_weight_quadrants, visual_weight_balance, symmetry_horizontal, symmetry_vertical, dominant_line_angles, leading_lines_converge_to_subject, line_pattern, horizon_tilt_degrees, scene_type, dominant_colors, color_harmony_type, color_harmony_score.

### `AnalysisReport` (7 string fields)
`summary`, `composition_feedback`, `technical_feedback`, `aesthetic_feedback`, `improvement_tips`, `photographer_recommendations`, `style_references`.

### `AnalyseResponse`
Top-level response: `exif` + `quality_tier` + `technical` + `composition` + `report`.

---

## 9. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | — | (legacy; not used in current OpenRouter setup) |
| `OPENROUTER_API_KEY` | required | OpenRouter API key for LLM calls |
| `LLM_MODEL` | `anthropic/claude-haiku-4-5-20251001` | Model identifier passed to OpenRouter |

All variables are loaded from `.env` via `python-dotenv`. Never hard-code keys.

---

## 10. Key Conventions

- `uploads/` is gitignored — never commit user images.
- Model weights (pyiqa, rembg U²-Net) are downloaded at runtime — never commit weight files.
- `prompts/system.md` is read at synthesizer module import time — keep it on disk.
- pyiqa models are initialised once at module import; each metric call is wrapped in `try/except` and returns `NaN`/`None` on failure.
- rembg uses a lazy import inside `_saliency_map()` and falls back to a uniform map on failure.
