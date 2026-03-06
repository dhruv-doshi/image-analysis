# FrameIQ — Architecture Reference

## 1. Overview

FrameIQ is a FastAPI + Next.js web application that accepts an uploaded photograph and returns an AI-powered analysis covering composition, aesthetics, technical quality, improvement tips, and photographer/style recommendations.

The backend is a **three-layer pipeline**:

1. **Layer 1 — Technical Analysis**: learned IQA metrics (pyiqa) + classical computer-vision metrics (OpenCV, scikit-image).
2. **Layer 2 — Composition Analysis**: salient-object detection (rembg U²-Net) → geometric and perceptual composition scores.
3. **Layer 3 — LLM Synthesis**: all numeric scores serialised into an annotated JSON payload, sent to Claude via OpenRouter; structured `AnalysisReport` returned.

A **metrics configuration layer** (`src/config/metrics.py`) sits across Layers 1–3 and controls which metrics are active and how they contribute to the quality tier.

---

## 2. Pipeline Diagram

```
                          ┌─────────────────────────────────────────────────────────────────────┐
                          │                       FastAPI  (api.py)                              │
                          │                                                                      │
  HTTP POST /analyse ────►│  Image bytes (≤20 MB, JPEG/PNG)                                     │
  (multipart upload)      │      │                                                               │
                          │      ▼                                                               │
                          │  ┌───────────────────────┐                                          │
                          │  │   Image Loader         │  PIL → RGB → resize (≤1024px) →         │
                          │  │   src/utils/loader.py  │  pil_image + BGR ndarray + float32 tensor│
                          │  │                        │  EXIF metadata extracted                 │
                          │  └──────────┬────────────┘                                          │
                          │             │                                                        │
                          │             ▼                                                        │
                          │  ┌───────────────────────┐                                          │
                          │  │  Layer 1 — Technical   │  pyiqa learned metrics                  │
                          │  │  src/analysis/         │  BRISQUE · NIMA · CLIP-IQA+             │
                          │  │    technical.py        │  MUSIQ · NIQE                           │
                          │  │                        │  + classical CV sharpness / noise /      │
                          │  │  Output: TechnicalScores│   exposure / dynamic range / contrast  │
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

### `load_image(source) → (pil_image, bgr_array, tensor)`

`source` may be a file path (`str`/`Path`) or a file-like object (e.g. FastAPI `UploadFile`).

| Step | Detail |
|------|--------|
| Open | `PIL.Image.open()` → convert to RGB |
| Resize | Long edge capped at `_MAX_ANALYSIS_DIM = 1024 px` using Lanczos resampling |
| `pil_image` | RGB PIL Image with EXIF preserved in `.info` |
| `bgr_array` | `np.array(img)[:, :, ::-1]` — uint8 BGR for OpenCV |
| `tensor` | `float32` in range [0, 1], shape `(1, 3, H, W)` — pyiqa-compatible |

**Why 1024 px?** ViT-based models (MUSIQ, CLIP-IQA+) scale as O(patches²) in attention. A 16 MP image took 418 s before the cap was introduced.

### `extract_exif(pil_image) → ExifData`

- Reads all EXIF tags via Pillow's `_getexif()`.
- For image dimensions, prefers `ExifImageWidth` / `ExifImageHeight` (original camera dimensions) over the post-resize PIL size.
- Extracts: camera make/model, ISO, shutter speed, aperture (f-number), focal length, lens model.

---

## 4. Layer 1 — Technical Analysis (`src/analysis/technical.py`)

All pyiqa models are instantiated **once at module import** as module-level singletons (`_brisque`, `_nima`, `_clip_iqa`, `_musiq`, `_niqe`). Each metric call is wrapped in `try/except` — on failure the field is `NaN` / `None`.

Every metric is guarded by `is_enabled()` from the config layer (Section 10) before being computed or included in results.

### 4.1 Learned Metrics (pyiqa)

| Field | Model | How it works | Scale | Interpretation |
|-------|-------|--------------|-------|----------------|
| `brisque` | BRISQUE | DCT coefficient statistics fit to a Generalised Gaussian Distribution; deviation from "natural scene statistics" modelled with an SVR | 0–100 | Lower = better. <30 excellent, 30–50 good, 50–65 fair, 65–80 poor |
| `nima_aesthetic` | NIMA (aesthetic) | Inception-ResNet-V2 trained on the AVA dataset; predicts a distribution over 1–10 human ratings; returns the expected score | 1–10 | Higher = better. <5 below average, 7+ strong |
| `clip_iqa` | CLIP-IQA+ | CLIP vision-language model prompted with quality/distortion text pairs; cosine similarity to "good quality" direction | 0–1 | Higher = better. <0.4 poor, >0.6 good |
| `musiq` | MUSIQ | Multi-scale ViT that patches the image at several resolutions; trained on multiple IQA datasets | 0–100 | Higher = better. Direct perceptual quality score |
| `niqe` | NIQE | No-reference metric based on natural scene statistics; detects compression artefacts and unnatural distortions | lower = better | <3 excellent, 3–5 good, 5–8 average, 8–12 poor, ≥12 terrible |

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

Every sub-metric is guarded by `is_enabled()` before computation.

### 5.1 Saliency Map

`_saliency_map(bgr_image) → float32 [0,1] H×W array`

- Calls **rembg** with U²-Net (`u2net`) to segment the salient foreground.
- Extracts the alpha channel of the RGBA output as the saliency weight.
- Falls back to a uniform map (all ones) on import failure or exception.
- The resulting map is stored in `CompositionScores.saliency_map` (`exclude=True`) — available for visualisation within a request but not serialised in API responses.

> **Known inefficiency**: `_detect_lines()` is called once for horizon tilt / scene classification and `_leading_lines()` runs its own separate Canny + HoughLinesP pass internally — two full edge-detection passes over the same image.

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
3. Angles binned into 15° clusters; cluster medians reported (up to 12 clusters).

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
| `architecture` | >8 vertical lines (angle 70°–110°) and ≥50% of detected angles are vertical |
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

Before calling the LLM, `_compute_quality_tier()` converts raw metrics to **ordinal scores** (0 = excellent … 4 = terrible) and aggregates them into a weighted average. Weights are pulled from `get_weight()` in the config layer (Section 10), so disabling a metric automatically removes it from the tier calculation.

| Metric | Ordinal thresholds | Config weight |
|--------|--------------------|---------------|
| BRISQUE | <30→0, <50→1, <65→2, <80→3, else 4; NaN→2 | 3.0 |
| Sharpness | Noise-adjusted: `adj = sharpness − noise² × 20`; >500→0, ≥200→2, ≥80→3, else 4 | 2.0 |
| Noise | <3→0, ≤8→2, ≤15→3, else 4; NaN→2 | 2.0 |
| Exposure | Counts severe (hl>15%, sh>20%, mean<40 or >230) and moderate thresholds | 1.0 each |
| NIMA | ≥7→0, ≥6→1, ≥5→2, ≥4→3, <4→4; None/NaN→None (excluded) | 2.0 |
| CLIP-IQA+ | ≥0.60→0, ≥0.50→1, ≥0.40→2, ≥0.30→3, <0.30→4 | 1.0 |
| MUSIQ | ≥70→0, ≥55→1, ≥40→2, ≥25→3, <25→4 | 1.5 |
| NIQE | <3→0, <5→1, <8→2, <12→3, ≥12→4 | 1.0 |
| Composition | `_composition_ord()`: penalty based on `visual_weight_balance` and `rot_alignment_score` | 2.0 |

`_composition_ord()` converts the composition sub-scores into a single ordinal using visual weight balance (imbalanced > 3.0 → poor) and RoT alignment (>0.7 → poor).

The return value of `_compute_quality_tier()` is a plain `dict` that maps to the `QualityTier` Pydantic model (see Section 8), with individual tier strings for each enabled metric plus an `overall` tier.

### 6.2 Payload Construction

`_build_payload(tech, comp, exif, features)` serialises all scores into a JSON **user message**. Every numeric field is wrapped with both a `value` and a `scale` description string so the LLM receives human-readable interpretation bounds alongside each number. The payload also includes a `requested_features` list so the LLM knows which output sections to generate.

Example snippet:
```json
{
  "brisque": {
    "value": 28.4,
    "scale": "0–100, lower is better; <30 excellent, 30–50 good, 50–65 fair, >65 poor"
  },
  "niqe": {
    "value": 4.1,
    "scale": "lower is better; <3 excellent, 3–5 good, 5–8 average, >8 poor"
  },
  "requested_features": ["composition", "technical", "improvements"]
}
```

Metrics disabled in the config layer are omitted from the payload entirely.

### 6.3 LLM Call

| Aspect | Detail |
|--------|--------|
| Client | OpenAI-compatible SDK pointed at OpenRouter base URL |
| Auth | `OPENROUTER_API_KEY` environment variable |
| Model | `LLM_MODEL` env var (default: `anthropic/claude-haiku-4-5-20251001`) |
| System prompt | `prompts/system.md` — expert photography critic persona, quality-tier tone anchoring table, genre-aware guidance sections (portrait / landscape / architecture / macro) |
| Max tokens | 4096 |
| Sync call | `synthesise()` → returns complete `AnalysisReport` |
| Streaming | `synthesise_stream()` → SSE generator yielding text chunks; guards `if not chunk.choices: continue` to handle OpenRouter keepalive empty-choices chunks |

**SSE event sequence for `/analyse/stream`:**

| Event type | Payload | When emitted |
|------------|---------|--------------|
| `metrics` | Full `{exif, quality_tier, technical, composition}` JSON | Immediately after L1+L2 complete |
| `chunk` | Raw LLM text delta | Each streamed token |
| `report` | Parsed `AnalysisReport` JSON | After LLM stream ends and JSON is sanitised |
| `done` | `{}` | End of stream |
| `error` | `{detail: string}` | On any exception |

### 6.4 JSON Sanitisation (`_sanitise_llm_json`)

The LLM output is cleaned before `json.loads()`:

1. Strip markdown code fences (` ```json … ``` `).
2. Extract from first `{` to last `}` (strips preamble / epilogue).
3. Remove leading-plus signs from numbers (`+15` → `15`).
4. Insert missing commas between adjacent JSON fields.
5. Remove trailing commas before `}` or `]`.

---

## 7. API Entry Point (`api.py`)

| Aspect | Detail |
|--------|--------|
| Framework | FastAPI |
| Max upload size | 20 MB |
| Allowed content types | `image/jpeg`, `image/png` |
| CORS | `http://localhost:3000` + comma-separated `ALLOWED_ORIGINS` env var |
| Log level | `LOG_LEVEL` env var (default: `INFO`) |
| Torch multiprocessing | `torch.multiprocessing.set_sharing_strategy("file_system")` at startup to avoid named-semaphore leaks on macOS |
| Lifespan | `@asynccontextmanager` startup handler calls `log_active_pipeline()`, then pre-warms pyiqa + rembg models |

### Endpoints

| Method | Path | Form params | Response |
|--------|------|-------------|----------|
| `GET` | `/health` | — | `{"status": "ok", "models_loaded": bool}` |
| `POST` | `/analyse` | `file` (UploadFile), `features` (str, default `"full"`) | `AnalyseResponse` JSON |
| `POST` | `/analyse/stream` | same | SSE stream (see Section 6.3) |

### Feature Flags (`_parse_features`)

The `features` form field accepts a comma-separated list (e.g. `"composition,technical"`) or `"full"`. It is parsed by `_parse_features()` into an `AnalysisFeature` flag (a bitfield). The flag controls which sections the LLM generates and which fields appear in the `AnalysisReport`.

`_FEATURE_MAP` in `api.py` maps string names to `AnalysisFeature` members: `composition`, `aesthetics`, `technical`, `improvements`, `editing`, `inspiration`, `full`.

### Helpers

- `_json_safe()` — Recursively replaces `nan`/`inf` in metric dicts before JSON serialisation (the JSON spec forbids these values).

---

## 8. Data Models (`src/models.py`)

All models are **Pydantic** `BaseModel`.

### `AnalysisFeature` (Flag enum)

Bitfield controlling which LLM output sections are requested:

| Member | Value |
|--------|-------|
| `COMPOSITION` | 1 |
| `AESTHETICS` | 2 |
| `TECHNICAL` | 4 |
| `IMPROVEMENTS` | 8 |
| `EDITING` | 16 |
| `INSPIRATION` | 32 |
| `FULL` | all flags OR'd (63) |

### `ExifData`
Camera make, model, ISO, shutter speed, aperture (f-number), focal length, lens model, original image dimensions (`image_width`, `image_height`).

### `TechnicalScores` (14 fields)

| Group | Fields |
|-------|--------|
| Learned | `brisque`, `nima_aesthetic`, `clip_iqa`, `musiq`, `niqe` |
| Sharpness | `sharpness_laplacian`, `sharpness_regional` |
| Noise | `noise_sigma` |
| Exposure | `exposure_clipped_highlights_pct`, `exposure_clipped_shadows_pct`, `histogram_mean`, `histogram_std` |
| Tonal | `dynamic_range_stops`, `contrast_rms` |

### `CompositionScores`

All centroid, alignment, negative-space, visual-weight, symmetry, leading-lines, horizon-tilt, scene-type, and color-harmony fields described in Section 5.

`saliency_map: Any | None` — declared with `exclude=True`; never appears in serialised API responses but is available within the request lifecycle for visualisation.

### `AnalysisReport` (7 string fields)

| Field | Present when |
|-------|-------------|
| `summary` | Always |
| `composition` | `AnalysisFeature.COMPOSITION` requested |
| `aesthetics` | `AnalysisFeature.AESTHETICS` requested |
| `technical` | `AnalysisFeature.TECHNICAL` requested |
| `improvements` | `AnalysisFeature.IMPROVEMENTS` requested |
| `editing` | `AnalysisFeature.EDITING` requested |
| `inspiration` | `AnalysisFeature.INSPIRATION` requested |

### `QualityTier`

Pydantic `BaseModel` with `overall` (tier string) plus per-metric tier fields:

`overall`, `brisque_tier`, `sharpness_tier`, `noise_tier`, `exposure_tier`, `composition_tier`, `nima_tier`, `clip_tier`, `musiq_tier`, `niqe_tier`

Each per-metric tier is `str | None` — `None` if the metric is disabled in config.

### `AnalyseResponse`
Top-level response: `exif` + `quality_tier` + `technical` + `composition` + `report`.

---

## 9. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | required | OpenRouter API key for LLM calls |
| `LLM_MODEL` | `anthropic/claude-haiku-4-5-20251001` | Model identifier passed to OpenRouter |
| `LOG_LEVEL` | `INFO` | Python logging level for the API server |
| `ALLOWED_ORIGINS` | _(none)_ | Comma-separated extra CORS origins (in addition to `localhost:3000`) |

All variables are loaded from `.env` via `python-dotenv`. Never hard-code keys.

---

## 10. Metrics Configuration (`src/config/metrics.py`)

The config layer is the single source of truth for which metrics are active and how much they contribute to the quality tier.

### `MetricConfig` (TypedDict)

```python
class MetricConfig(TypedDict):
    enabled: bool
    weight: float   # 0 = informational (not used in quality tier)
```

### `METRICS` dict

Hard-coded dict of 28 metrics (`str → MetricConfig`). Covers all Layer-1 and Layer-2 metrics. Selected defaults:

| Metric | Enabled | Weight | Notes |
|--------|---------|--------|-------|
| `brisque` | `True` | 3.0 | |
| `nima_aesthetic` | `True` | 2.0 | |
| `clip_iqa` | `True` | 1.0 | |
| `musiq` | `True` | 1.5 | |
| `niqe` | `True` | 1.0 | |
| `sharpness_laplacian` | `True` | 2.0 | |
| `sharpness_regional` | `True` | 0.0 | informational only |
| `noise_sigma` | `True` | 2.0 | |
| `rot_alignment_score` | `True` | 2.0 | drives `_composition_ord()` |
| _(all others)_ | `True` | 0.0 | informational only |

### Public API

| Function | Signature | Behaviour |
|----------|-----------|-----------|
| `is_enabled` | `(metric: str) → bool` | Returns `False` for unknown metrics |
| `get_weight` | `(metric: str) → float` | Returns `0.0` for unknown metrics |
| `log_active_pipeline` | `() → None` | Logs formatted summary of active metrics at startup |

`log_active_pipeline()` is called during the FastAPI lifespan startup hook.

---

## 11. Key Conventions

- `uploads/` is gitignored — never commit user images.
- Model weights (pyiqa, rembg U²-Net) are downloaded at runtime — never commit weight files.
- `prompts/system.md` is read at synthesizer module import time — keep it on disk.
- pyiqa models are initialised once at module import; each metric call is wrapped in `try/except` and returns `NaN`/`None` on failure.
- rembg uses a lazy import inside `_saliency_map()` and falls back to a uniform map on failure.
- `saliency_map` is excluded from all serialised output (`exclude=True` on the Pydantic field).
- Torch multiprocessing sharing strategy is set to `"file_system"` at startup to prevent named-semaphore leaks on macOS.
