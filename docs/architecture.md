# FrameIQ — Architecture Reference

## 1. Overview

FrameIQ is a FastAPI + Next.js web application that accepts an uploaded photograph and returns an AI-powered analysis covering composition, aesthetics, technical quality, improvement tips, and photographer/style recommendations.

The backend is a **three-layer pipeline**:

1. **Layer 1 — Technical Analysis**: learned IQA metrics (pyiqa) + classical computer-vision metrics (OpenCV, scikit-image). All five pyiqa scorers run in parallel via `ThreadPoolExecutor`.
2. **Layer 2 — Composition Analysis**: OpenCV spectral-residual saliency (fast default) or rembg U²-Net (opt-in heavy mode) → geometric and perceptual composition scores.
3. **Layer 3 — LLM Synthesis (multimodal)**: all numeric scores serialised into an annotated JSON payload **plus the photograph itself** (base64 JPEG), sent together to Claude via OpenRouter. The LLM uses the image as ground truth and the metrics as supporting evidence. Structured `AnalysisReport` returned.

A **metrics configuration layer** (`src/config/metrics.py`) sits across Layers 1–3 and controls which metrics are active and how they contribute to the quality tier.

All CPU-bound pipeline work (L1, L2, L3) is offloaded to a thread-pool executor via `asyncio.get_running_loop().run_in_executor()` so the FastAPI event loop is never blocked.

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
                          │  │   Pre-screening gate   │  _is_photograph() — CLIP zero-shot      │
                          │  │   src/analysis/        │  + statistical check                    │
                          │  │     technical.py       │  → HTTP 422 if not a photograph         │
                          │  └──────────┬────────────┘                                          │
                          │             │                                                        │
                          │             ▼                                                        │
                          │  ┌───────────────────────┐                                          │
                          │  │   Image Loader         │  PIL → RGB → resize (≤1024px) →         │
                          │  │   src/utils/loader.py  │  pil_image + BGR ndarray + float32 tensor│
                          │  │                        │  EXIF metadata extracted                 │
                          │  └──────────┬────────────┘                                          │
                          │             │                                                        │
                          │             ▼                                                        │
                          │  ┌───────────────────────┐  Parallel ThreadPoolExecutor             │
                          │  │  Layer 1 — Technical   │  BRISQUE · NIMA · CLIP-IQA+             │
                          │  │  src/analysis/         │  MUSIQ · NIQE  ←── all concurrent       │
                          │  │    technical.py        │  + classical CV sharpness / noise /      │
                          │  │                        │    exposure / dynamic range / contrast   │
                          │  │  Output: TechnicalScores│                                        │
                          │  └──────────┬────────────┘                                          │
                          │             │                                                        │
                          │             ▼                                                        │
                          │  ┌───────────────────────┐                                          │
                          │  │  Layer 2 — Composition │  OpenCV spectral residual saliency      │
                          │  │  src/analysis/         │  (rembg U²-Net opt-in via config)       │
                          │  │    composition.py      │  RoT / GR alignment · neg. space        │
                          │  │                        │  symmetry · leading lines               │
                          │  │                        │  horizon tilt · scene class             │
                          │  │  Output: CompositionScores  color harmony                        │
                          │  └──────────┬────────────┘                                          │
                          │             │                                                        │
                          │             ▼                                                        │
                          │  ┌────────────────────────────────────────┐                         │
                          │  │  Layer 3 — LLM Synthesis (multimodal)  │                         │
                          │  │  src/llm/synthesizer.py                │                         │
                          │  │  src/llm/client.py                     │                         │
                          │  │                                        │                         │
                          │  │  User message content:                 │                         │
                          │  │    [text]  annotated JSON payload      │                         │
                          │  │    [image] base64 JPEG of photo        │                         │
                          │  │                                        │                         │
                          │  │  → Claude (OpenRouter)                 │                         │
                          │  │  ← Structured AnalysisReport JSON      │                         │
                          │  │  Output: AnalysisReport                │                         │
                          │  └──────────┬─────────────────────────────┘                         │
                          │             │                                                        │
                          │             ▼                                                        │
  JSON response ◄─────────│  AnalyseResponse { exif, quality_tier, technical,                   │
                          │                    composition, report }                             │
                          └─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Image Loading (`src/utils/loader.py`)

### `load_image(source) → (pil_image, bgr_array, tensor)`

`source` may be a file path (`str`/`Path`) or a file-like object.

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

Every metric is guarded by `is_enabled()` from the config layer before being computed. All five pyiqa scorers run **concurrently** inside a `ThreadPoolExecutor` — wall-clock time is `max(model_latencies)` rather than `sum(model_latencies)`.

### 4.1 Pre-screening Gate (`_is_photograph`)

Before L1 runs, `_is_photograph(bgr_array, tensor)` performs a two-tier check:

- **Tier 1 (statistical)**: If `std(grayscale) < 8`, the image is near-blank/solid → immediate reject.
- **Tier 2 (CLIP)**: If `_clip_prescreener` is enabled, uses the loaded CLIP-IQA+ model to zero-shot classify the image as photograph vs. non-photograph.

Returns `(True, "")` or `(False, reason_string)`. On rejection the API returns HTTP 422. Controlled by the `FRAMEIQ_PRESCREENING` env var.

### 4.2 Learned Metrics (pyiqa)

| Field | Model | How it works | Scale | Interpretation |
|-------|-------|--------------|-------|----------------|
| `brisque` | BRISQUE | DCT coefficient statistics fit to a Generalised Gaussian Distribution; deviation from "natural scene statistics" modelled with an SVR | 0–100 | Lower = better. <30 excellent, 30–50 good, 50–65 fair, 65–80 poor |
| `nima_aesthetic` | NIMA (aesthetic) | Inception-ResNet-V2 trained on the AVA dataset; predicts a distribution over 1–10 human ratings; returns the expected score | 1–10 | Higher = better. <5 below average, 7+ strong |
| `clip_iqa` | CLIP-IQA+ | CLIP vision-language model prompted with quality/distortion text pairs; cosine similarity to "good quality" direction | 0–1 | Higher = better. <0.4 poor, >0.6 good |
| `musiq` | MUSIQ | Multi-scale ViT that patches the image at several resolutions; trained on multiple IQA datasets | 0–100 | Higher = better. Direct perceptual quality score |
| `niqe` | NIQE | No-reference metric based on natural scene statistics; detects compression artefacts and unnatural distortions | lower = better | <3 excellent, 3–5 good, 5–8 average, 8–12 poor, ≥12 terrible |

### 4.3 Classical CV Metrics

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

**Default (fast):** OpenCV spectral residual saliency (`cv2.saliency.StaticSaliencySpectralResidual_create()`). Runs in ~50–100 ms with no extra dependencies.

**Heavy mode (opt-in):** rembg U²-Net (`u2net`). Enabled by setting `use_heavy_saliency` to `True` in `src/config/metrics.py`. Takes 2–5 s and requires the ~200 MB U²-Net model. Falls back to a uniform map on failure.

The saliency map is stored in `CompositionScores.saliency_map` (`exclude=True`) — available within a request but not serialised in API responses.

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

Before calling the LLM, `_compute_quality_tier()` converts raw metrics to **ordinal scores** (0 = excellent … 4 = terrible) and aggregates them into a weighted average. Weights are pulled from `get_weight()` in the config layer, so disabling a metric automatically removes it from the tier calculation.

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

**Critical failure gates (applied after the weighted average):**

These prevent a single catastrophic failure from being averaged away by clean scores on unrelated metrics.

| Condition | Effect | Rationale |
|-----------|--------|-----------|
| Sharpness ordinal = 4 (terrible) | `overall ≥ poor` | A blurry image is at minimum "poor" regardless of clean noise or good BRISQUE |
| Exposure ordinal = 4 (terrible) | `overall ≥ poor` | Catastrophically under/over-exposed images cannot be "average" or better |
| NIMA ordinal ≥ 2 (average or below) | `overall ≤ good` | Technical perfection does not make an image aesthetically compelling; NIMA is the only metric trained on human aesthetic ratings — if it says average, the overall cannot be "excellent" |

### 6.2 Payload Construction

`_build_payload(tech, comp, exif, features)` serialises all scores into a JSON text block. Every numeric field is wrapped with both a `value` and a `scale` description string. The payload also includes a `requested_features` list so the LLM knows which output sections to generate.

Metrics disabled in the config layer are omitted from the payload entirely.

### 6.3 Multimodal LLM Input

`synthesise(tech, comp, exif, features, pil_image=None)` and `synthesise_stream(tech, comp, exif, features, pil_image=None)` accept an optional PIL image.

When `pil_image` is provided (as it always is from the API endpoints):
- The image is re-encoded as a JPEG at quality=85 and base64-encoded.
- The user message is a **content list**: `[{"type": "text", "text": payload}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]`.
- When `pil_image` is `None` (e.g. in tests), content is a plain string — the text-only path remains available for backward compatibility.

The LLM treats the **photograph as ground truth** and the metrics as supporting evidence. The system prompt (`prompts/system.md`) instructs the model to flag discrepancies between what it sees and what the metrics report.

### 6.4 System Prompt (`prompts/system.md`)

The system prompt establishes a **critical, fault-finding posture**:

- "Your visual judgement comes first." — metrics confirm or quantify; they do not conclude.
- "Every image has meaningful problems — find them." — no softening of faults.
- Metric names (BRISQUE, NIMA, etc.) must not appear in output prose; only human-readable descriptions of what was observed.
- `quality_tier` is explicitly ignored by the LLM — it forms its own overall assessment from the image.
- Genre-aware guidance for portrait, landscape, architecture, macro, and general scenes.
- `improvements` must be exactly 3, concrete, ranked by impact.

### 6.5 LLM Call

| Aspect | Detail |
|--------|--------|
| Client | OpenAI-compatible SDK pointed at OpenRouter base URL |
| Auth | `OPENROUTER_API_KEY` environment variable |
| Model | `LLM_MODEL` env var (default: `anthropic/claude-haiku-4-5-20251001`) |
| System prompt | `prompts/system.md` — critical photography critic persona, image-first stance, genre-aware guidance |
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

### 6.6 JSON Sanitisation (`_sanitise_llm_json`)

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
| `POST` | `/analyse/stream` | same | SSE stream (see Section 6.5) |

### Feature Flags (`_parse_features`)

The `features` form field accepts a comma-separated list (e.g. `"composition,technical"`) or `"full"`. Parsed by `_parse_features()` into an `AnalysisFeature` flag (a bitfield). The flag controls which sections the LLM generates and which fields appear in the `AnalysisReport`.

### Helpers

- `_json_safe()` — Recursively replaces `nan`/`inf` in metric dicts before JSON serialisation.

---

## 8. Data Models (`src/models.py`)

All models are **Pydantic** `BaseModel`.

### `AnalysisFeature` (Flag enum)

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

`saliency_map: Any | None` — declared with `exclude=True`; never appears in serialised API responses.

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

`overall`, `brisque_tier`, `sharpness_tier`, `noise_tier`, `exposure_tier`, `composition_tier`, `nima_tier`, `clip_tier`, `musiq_tier`, `niqe_tier` — each `str | None` (None if metric disabled).

---

## 9. Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | required | OpenRouter API key for LLM calls |
| `LLM_MODEL` | `anthropic/claude-haiku-4-5-20251001` | Model identifier passed to OpenRouter |
| `LOG_LEVEL` | `INFO` | Python logging level for the API server |
| `ALLOWED_ORIGINS` | _(none)_ | Comma-separated extra CORS origins |
| `FRAMEIQ_PRESCREENING` | `True` (code default) | Enable/disable pre-screening gate |

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

| Metric | Enabled | Weight | Notes |
|--------|---------|--------|-------|
| `brisque` | `True` | 3.0 | |
| `nima_aesthetic` | `True` | 2.0 | Also drives aesthetic gate |
| `clip_iqa` | `True` | 1.0 | Also used for pre-screening |
| `musiq` | `True` | 1.5 | |
| `niqe` | `True` | 1.0 | |
| `sharpness_laplacian` | `True` | 2.0 | Also drives sharpness gate |
| `sharpness_regional` | `True` | 0.0 | informational only |
| `noise_sigma` | `True` | 2.0 | |
| `rot_alignment_score` | `True` | 2.0 | drives `_composition_ord()` |
| `use_heavy_saliency` | `False` | 0.0 | Set `True` to use rembg U²-Net instead of OpenCV spectral residual |
| _(all others)_ | `True` | 0.0 | informational only |

### Public API

| Function | Signature | Behaviour |
|----------|-----------|-----------|
| `is_enabled` | `(metric: str) → bool` | Returns `False` for unknown metrics |
| `get_weight` | `(metric: str) → float` | Returns `0.0` for unknown metrics |
| `log_active_pipeline` | `() → None` | Logs formatted summary at startup |

---

## 11. LLM Strategy Experiment (`src/llm/comparison.py`)

A standalone comparison module used to validate the multimodal approach. Tested three strategies across 7 evaluation images (`tests/evaluation/images/`):

| Strategy | Input | Key finding |
|----------|-------|-------------|
| `metrics_only` | JSON payload only | Correctly measures technical issues but blind to subject identity; misattributes root causes (e.g. blamed a lens for blur caused by a blurred foreground person) |
| `metrics_and_photo` | JSON payload + image | Best across all images: correct scene understanding + precise technical grounding. Never hallucinated. Selected as the production strategy. |
| `photo_only` | Image only | Strong scene understanding but hallucinates on technical properties (rated a Laplacian=62 image as "sharp") |

Full results: `tests/evaluation/results/strategy_comparison_20260307.md`
Raw JSON per-image: `prompts/comparisons/`

Run a new comparison:
```bash
python scripts/compare_strategies.py path/to/photo.jpg
python scripts/compare_strategies.py path/to/photo.jpg --no-llm   # L1+L2 only
```

---

## 12. Key Conventions

- `uploads/` is gitignored — never commit user images.
- Model weights (pyiqa, rembg U²-Net) are downloaded at runtime — never commit weight files.
- `prompts/system.md` is read at synthesizer module import time — keep it on disk.
- pyiqa models are initialised once at module import; each metric call is wrapped in `try/except` and returns `NaN`/`None` on failure.
- rembg uses a lazy import inside `_saliency_map()` and falls back to a uniform map on failure.
- `saliency_map` is excluded from all serialised output (`exclude=True` on the Pydantic field).
- Torch multiprocessing sharing strategy is set to `"file_system"` at startup to prevent named-semaphore leaks on macOS.
- The LLM sees both the photograph and the numeric metrics. It is instructed to trust its visual assessment first and use metrics as supporting evidence.
