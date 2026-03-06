# FrameIQ — Pipeline Improvement Proposals

Prioritised architectural and algorithmic improvements. Prompt-level tweaks are excluded; every proposal here requires a code change.

Each entry follows the pattern: **Problem → Proposed solution → Impact → Complexity**.

---

## Priority summary

| # | Area | Problem | Impact | Complexity |
|---|------|---------|--------|------------|
| 1 | Performance | Layer 1 metrics run sequentially | High | Low |
| 2 | Performance | Layers 1+2 block the async event loop | High | Low |
| 3 | Performance | rembg U²-Net saliency is the heaviest single step | High | Medium |
| 4 | Architecture | `_detect_lines()` runs twice per request | Medium | Low |
| 5 | Metric quality | BRISQUE + NIQE are both NSS-based no-reference metrics | Medium | Low |
| 6 | Metric quality | Haar cascade face detector has poor recall/precision | Medium | Medium |
| 7 | Architecture | Metrics config is not environment-driven | Medium | Low |
| 8 | Architecture | `_compute_quality_tier()` returns a raw dict, not a `QualityTier` model | Low | Low |
| 9 | Metric quality | Color clustering uses a Lab→BGR→HSV round-trip | Low | Low |
| 10 | Metric quality | Three correlated tonal metrics add noise to the tier | Low | Low |
| 11 | Performance | Result cache for repeated uploads | Low | Low |
| 12 | Vision model | LLM receives only numeric metrics, not the image itself | High | Low |
| 13 | Vision model | Scene classification uses heuristics; CLIP is already loaded | Medium | Low |
| 14 | Vision model | Numeric metrics can't capture semantic/contextual quality | Medium | Medium |
| 15 | Vision model | Full pipeline runs even for unsuitable images (blank, corrupt, screenshots) | Medium | Low |
| 16 | Metric quality | Metric weights are hand-tuned constants with no feedback signal | High | Medium |

---

## 1. Parallel Layer-1 metric execution

**Problem.** `analyse_technical()` evaluates five pyiqa models sequentially: BRISQUE → NIMA → CLIP-IQA+ → MUSIQ → NIQE. MUSIQ (multi-scale ViT) dominates, taking 2–4 s on a 1024 px image even after the resize cap. The total Layer-1 wall-clock time is the sum of all five models.

**Proposed solution.** Run each `_score_*()` call in a `ThreadPoolExecutor`. PyTorch releases the GIL during inference (`torch.no_grad()` context), so CPU-bound models can run concurrently on separate threads. Each model is already a module-level singleton, so there is no init overhead inside the thread.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=5) as pool:
    futures = {
        pool.submit(_score_brisque, tensor): "brisque",
        pool.submit(_score_nima, tensor):    "nima",
        ...
    }
    results = {name: f.result() for f, name in futures.items()}
```

**Impact.** Wall-clock time for Layer 1 drops from `sum(model_latencies)` to roughly `max(model_latencies)` — a 50–70% reduction for typical images.

**Complexity.** Low. No new dependencies. Each scorer is already isolated and returns a plain `float | None`.

---

## 2. Offload CPU-bound layers to a thread pool executor

**Problem.** Both `/analyse` and `/analyse/stream` call `analyse_technical()` and `analyse_composition()` directly from async handlers, blocking the FastAPI event loop for the full duration of L1+L2 (~7–15 s). No other request can be served while a request is in flight.

**Proposed solution.** Wrap both blocking calls with `asyncio.get_event_loop().run_in_executor()`:

```python
loop = asyncio.get_event_loop()
tech = await loop.run_in_executor(None, analyse_technical, bgr_array, tensor)
comp = await loop.run_in_executor(None, analyse_composition, bgr_array, pil_image, exif)
```

This moves the work to the default `ThreadPoolExecutor` and yields the event loop back while computing.

**Impact.** Server throughput increases 3–5× under concurrent load. The improvement applies to both endpoints with minimal code change.

**Complexity.** Low. The two function signatures require no changes; only the call sites in `api.py` are modified.

---

## 3. Replace rembg U²-Net saliency with a lightweight alternative

**Problem.** rembg's U²-Net model is ~200 MB and takes 2–5 s per inference on CPU, making it the largest single contributor to Layer-2 latency. The model is overkill for composition analysis: only the alpha channel (salient-region mask) is used, not the full segmented foreground.

**Proposed solution.** Replace `_saliency_map()` with OpenCV's spectral residual saliency, which runs in ~50–100 ms with no extra dependencies:

```python
saliency_algo = cv2.saliency.StaticSaliencySpectralResidual_create()
_, saliency_map = saliency_algo.computeSaliency(bgr_array)
```

Add a config flag `USE_HEAVY_SALIENCY` (default `False`). When `True`, fall back to the current rembg path for users who need high-precision subject segmentation (e.g. fine-art critique mode).

**Impact.** Layer-2 latency drops from ~5 s to ~200 ms in the default case. For portrait and macro scene types, where accurate subject boundary matters most, the heavy-model flag can be re-enabled.

**Complexity.** Medium. The interface of `_saliency_map()` does not change (returns same float32 [0,1] array). Centroid, symmetry, and visual-weight callers are unaffected. Requires tuning the spectral residual output range to [0,1].

---

## 4. Deduplicate line detection (single Canny + HoughLinesP pass)

**Problem.** `_detect_lines(bgr_array)` is called from `_horizon_tilt()` and from scene-type classification, producing a list of raw angles. Separately, `_leading_lines()` runs its own full Canny + HoughLinesP internally, computing a second complete edge-detection pass over the same (already-resized) image.

**Proposed solution.** Compute lines once at the top of `analyse()` and thread the result through all consumers:

```python
raw_angles = _detect_lines(bgr_array)
horizon_tilt = _horizon_tilt(raw_angles)
scene_type, faces = _classify_scene(bgr_array, raw_angles, exif_data)
lines_result = _leading_lines(bgr_array, saliency_centroid, raw_angles)
```

Modify `_leading_lines()` to accept an optional pre-computed `raw_angles` argument; fall back to its own detection if `None`.

**Impact.** Removes ~100–200 ms of redundant edge detection. Also simplifies the call graph.

**Complexity.** Low. Internal refactor only; no public API changes.

---

## 5. Resolve BRISQUE / NIQE redundancy

**Problem.** BRISQUE and NIQE are both no-reference IQA metrics rooted in Natural Scene Statistics (NSS). Both measure deviation from statistical regularities found in undistorted natural images. On typical photographs they correlate strongly (Pearson r ≈ 0.75–0.85 on standard benchmarks), so both in the quality tier adds inference cost without proportionally more information.

**Recommendation.** Set NIQE `enabled = False` in `src/config/metrics.py` as the default, reducing Layer-1 latency by ~500 ms. NIQE is slightly more sensitive to compression artefacts, so re-enable it when the payload suggests a JPEG-compressed source (could be inferred from EXIF software tag or file extension). Alternatively, keep NIQE and disable BRISQUE — NIQE is parameter-free (no SVR), making it less likely to overfit.

**Impact.** ~500 ms latency reduction per request. Quality-tier accuracy is not materially degraded because CLIP-IQA+, NIMA, and MUSIQ already provide complementary signal.

**Complexity.** Low. One boolean change in `METRICS` dict; no code changes.

---

## 6. Replace Haar cascade face detector with OpenCV YuNet

**Problem.** The Haar cascade (`haarcascade_frontalface_default.xml`) used for portrait scene classification has well-known limitations: ~30–40% miss rate on rotated, occluded, or non-frontal faces; high false-positive rate on structured backgrounds. This directly affects scene classification accuracy and downstream LLM critique emphasis.

**Proposed solution.** Replace the Haar cascade with OpenCV's built-in YuNet DNN face detector (`cv2.FaceDetectorYN`), available since OpenCV 4.5.4:

```python
detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    score_threshold=0.6,
)
_, faces = detector.detect(bgr_resized)
```

YuNet runs in ~50 ms on CPU (vs. ~20 ms for Haar) but achieves ~90% recall on WIDER FACE compared to ~60% for Haar.

**Impact.** Portrait detection accuracy improves significantly, particularly for group photos, profile shots, and low-light faces. Scene classification quality improves across all categories (portrait mis-classifications were causing unnecessary landscape/architecture fallback).

**Complexity.** Medium. Requires bundling the ~1 MB ONNX model file or downloading it at startup (similar to pyiqa/rembg model download pattern).

---

## 7. Make metrics config environment-driven

**Problem.** `src/config/metrics.py` hard-codes enable/disable flags and weights in Python. Changing the active metric set requires a code edit and redeploy, even for operational toggles like "disable MUSIQ in the free tier".

**Proposed solution.** Read an optional `FRAMEIQ_METRICS_CONFIG` env var containing a JSON patch applied over the defaults:

```python
import json, os

_OVERRIDES = json.loads(os.getenv("FRAMEIQ_METRICS_CONFIG", "{}"))
for name, patch in _OVERRIDES.items():
    if name in METRICS:
        METRICS[name].update(patch)
```

Example deployment override:
```
FRAMEIQ_METRICS_CONFIG='{"musiq":{"enabled":false},"niqe":{"enabled":false}}'
```

**Impact.** Enables runtime control of metric sets without code changes or redeployment. Useful for cost tiering, A/B testing, or hardware-specific tuning.

**Complexity.** Low. ~10 lines in `metrics.py`; no downstream changes required.

---

## 8. Wire `_compute_quality_tier()` through the `QualityTier` Pydantic model

**Problem.** `_compute_quality_tier()` in `synthesizer.py` returns a plain `dict`. The `QualityTier` Pydantic model exists in `src/models.py` but is not used at the computation site — the dict is passed through to the response and the model is only applied later during serialisation. This means type errors in the dict silently produce invalid tier values that only fail at response time.

**Proposed solution.** Have `_compute_quality_tier()` return a `QualityTier` instance directly:

```python
def _compute_quality_tier(tech: TechnicalScores, comp: CompositionScores) -> QualityTier:
    ...
    return QualityTier(overall=overall, brisque_tier=brisque_t, ...)
```

**Impact.** Type errors in tier computation are caught at the point of construction, not serialisation. Makes the return type explicit in the function signature.

**Complexity.** Low. The dict keys already match the `QualityTier` field names. Change is a one-line return-type annotation plus the constructor call.

---

## 9. Eliminate the Lab → BGR → HSV round-trip in color harmony

**Problem.** In `_color_harmony()`, K-means is run in Lab space, then each cluster center is converted Lab → BGR → HSV to extract hue and saturation. The BGR intermediate step is unnecessary and introduces floating-point rounding error.

**Proposed solution.** Convert Lab → XYZ → RGB in one step, then RGB → HSV directly, or use `skimage.color.lab2rgb` followed by `skimage.color.rgb2hsv`. Both are single-step pipelines with no intermediate BGR conversion.

```python
from skimage.color import lab2rgb, rgb2hsv

rgb = lab2rgb(lab_center.reshape(1, 1, 3))[0, 0]
hsv = rgb2hsv(rgb.reshape(1, 1, 3))[0, 0]
hue, saturation = hsv[0] * 360, hsv[1]
```

**Impact.** Eliminates a small but unnecessary conversion error. Makes the computation path easier to follow.

**Complexity.** Low. Purely internal to `_color_harmony()`; no model or API changes.

---

## 10. Reduce correlated tonal metrics

**Problem.** `histogram_std`, `dynamic_range_stops` (log₂(p99/p1)), and `contrast_rms` (std/mean of grayscale) all measure tonal spread. They are strongly correlated on typical images. All three are currently informational (`weight=0`) but still computed and included in the LLM payload, increasing prompt length without adding proportional signal.

**Recommendation.** Remove `histogram_std` from the payload (it is the weakest signal — a direct std without the dynamic-range normalisation of `dynamic_range_stops` or the mean-normalisation of `contrast_rms`). Keep `dynamic_range_stops` (interpretable in photographic stops) and `contrast_rms` (normalised, scale-invariant). Update `METRICS` to set `histogram_std` `enabled = False`.

**Impact.** Slightly shorter LLM payload; marginal latency reduction from skipping the histogram std computation; cleaner payload for the LLM to interpret.

**Complexity.** Low. One boolean in `METRICS` dict; remove the histogram_std entry from `_build_payload()`.

---

## 11. In-memory result cache keyed on image hash

**Problem.** Repeated uploads of the same image (A/B retesting, frontend retries) re-run the full 10–20 s pipeline even though the result is deterministic.

**Proposed solution.** Compute `hashlib.sha256(raw_bytes).hexdigest()` before processing. Check an `LRU_CACHE` (e.g. `cachetools.LRUCache(maxsize=100)`) for a cached `AnalyseResponse`. If hit, return immediately. Cache entries expire after 1 hour via a `TTLCache`.

```python
from cachetools import TTLCache
_cache: TTLCache = TTLCache(maxsize=100, ttl=3600)

digest = hashlib.sha256(image_bytes).hexdigest()
if digest in _cache:
    return _cache[digest]
```

Gate the feature behind a `FRAMEIQ_ENABLE_CACHE=true` env var to make it opt-in.

**Impact.** Near-zero latency for repeat requests. Particularly useful during frontend development and for users who repeatedly tweak EXIF metadata comparisons.

**Complexity.** Low. Requires `cachetools` (lightweight, no server-side state). Thread-safe if `asyncio` access is serialised through the event loop.

---

## Vision model integration

The proposals below treat the LLM as a first-class vision model, not just a text synthesiser. They are grouped separately because they change the fundamental role of Layer 3 in the pipeline.

---

## 12. Send the image to the LLM alongside the numeric metrics (multimodal Layer 3)

**Problem.** `_build_payload()` currently sends only a JSON text message containing pre-computed numeric scores. The LLM never sees the photograph. This means it must trust the metrics unconditionally — it cannot catch cases where a metric is unreliable (e.g. BRISQUE mis-scoring HDR images, or CLIP-IQA+ giving high scores to technically sharp but compositionally empty shots).

**Proposed solution.** Extend `_build_payload()` to accept the PIL image and encode it as a base64 data URL. Pass it as a `image_url` content block alongside the JSON text using the OpenRouter vision API format (compatible with OpenAI's multimodal message structure):

```python
import base64, io

def _encode_image(pil_image: Image.Image, max_dim: int = 1024) -> str:
    """Resize and base64-encode for LLM vision input."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()

# In synthesise() / synthesise_stream():
user_content = [
    {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(pil_image)}"},
    },
    {
        "type": "text",
        "text": _build_payload(tech, comp, exif, features),
    },
]
```

The system prompt should be updated to instruct the model to use the image as ground truth and treat the numeric payload as supporting evidence, flagging discrepancies where appropriate.

**Required model change.** Switch `LLM_MODEL` to a vision-capable model. Any current Claude model (claude-haiku-4-5, claude-sonnet-4-6) and GPT-4o support multimodal input via OpenRouter. The default `anthropic/claude-haiku-4-5-20251001` already supports vision.

**What this unlocks:**
- The LLM can assess subject matter, emotional impact, and contextual appropriateness — things no numeric metric captures.
- It can cross-check suspicious metric values (e.g. "BRISQUE scores this as 22/excellent but I can see strong chromatic aberration on the edges").
- `inspiration` and `aesthetics` sections become substantially richer because the model can reference visible elements directly.
- Scene type reported by Layer 2 can be verified or overridden by the model's own visual understanding.

**Impact.** High. This is the single biggest qualitative improvement available without changing the numeric pipeline at all. The JSON metrics payload becomes an enriching context layer rather than the sole source of truth.

**Complexity.** Low. The OpenRouter/OpenAI SDK already handles multimodal messages. The change is entirely in `synthesizer.py` and `client.py` — no new dependencies, no pipeline restructuring. Base64 encoding a 1024 px JPEG adds ~300 KB to the API request (~50 ms network overhead).

**Caveats.** Token cost increases because vision tokens are priced separately. The `prompts/system.md` needs a short addition telling the model how to use the image vs. the metrics. Add a `FRAMEIQ_VISION_INPUT=true` env var to make this opt-in until it is tested at scale.

---

## 13. Replace heuristic scene classification with CLIP zero-shot

**Problem.** Scene classification in `_classify_scene()` relies on a Haar cascade for faces, geometric heuristics for landscape/architecture, and EXIF focal length for macro. Each heuristic has well-documented failure modes (Section 6 of this document). Meanwhile, the CLIP-IQA+ model (`_clip_iqa`) is already loaded at module import — its underlying ViT encoder can perform zero-shot image classification at negligible marginal cost.

**Proposed solution.** Add a `_classify_scene_clip(tensor)` function that queries the pre-loaded CLIP model with candidate text prompts using `pyiqa`'s lower-level API, or directly via the `transformers` CLIP processor:

```python
from transformers import CLIPModel, CLIPProcessor

_SCENE_PROMPTS = [
    "a portrait photograph of a person",
    "a landscape photograph of nature or scenery",
    "an architectural photograph of a building",
    "a macro photograph of a small subject up close",
    "a general photograph",
]

def _classify_scene_clip(pil_image: Image.Image) -> str:
    inputs = _clip_processor(
        text=_SCENE_PROMPTS, images=pil_image, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = _clip_model(**inputs).logits_per_image[0]
    probs = logits.softmax(dim=0)
    return ["portrait", "landscape", "architecture", "macro", "general"][probs.argmax()]
```

Run the CLIP classifier in parallel with the Haar cascade check; use the Haar result only to break ties when CLIP confidence is below 0.5.

**Impact.** Medium. Scene classification accuracy improves on ambiguous images (e.g. environmental portraits, urban landscapes with people, architectural details). The improvement flows directly into genre-aware critique from the system prompt.

**Complexity.** Low if the `transformers` CLIP model is already available (it is, via `pyiqa`'s CLIP-IQA+ dependency). Adds ~50 ms per request since the model is pre-loaded. No new pip dependencies.

---

## 14. Add a semantic quality layer using the vision model

**Problem.** The numeric pipeline measures low-level technical and geometric properties but cannot assess:
- Whether the subject is in focus vs. intentionally soft (creative choice vs. mistake)
- Whether an underexposed image is a moody night scene or a failed exposure
- Whether a high-noise result is film grain (artistic) or sensor noise (defect)
- Whether leading lines guide the eye toward something meaningful

These ambiguities cause the LLM to produce hedged critiques ("this may be intentional…") because it is reasoning from numbers, not pixels.

**Proposed solution.** When vision input is enabled (Proposal 12), extend the system prompt with a **semantic verification block** that asks the model to make explicit judgements before generating the report:

```
Before writing the report, assess the image directly on these dimensions:
1. Is the primary subject in sharp focus? (yes / no / intentionally soft)
2. Does the exposure level appear intentional for the scene?
3. Is visible noise/grain a stylistic choice or an artefact?
4. Do the leading lines guide the eye to a clear subject?

Use these assessments to contextualise the numeric metrics. Where your visual
assessment contradicts a metric, trust your visual assessment and note the discrepancy.
```

This costs no additional API call — it is a prompt addition to the existing Layer-3 call.

**Impact.** Medium. The most noticeable improvement is in the `improvements` and `technical` sections, which currently sometimes flag deliberate creative choices as defects. The semantic layer lets the model distinguish intent from error.

**Complexity.** Medium. Requires Proposal 12 (vision input) as a prerequisite. The prompt addition is straightforward, but the `AnalysisReport` model may need a `visual_assessment` field to surface the model's raw judgements to the frontend.

---

## 15. Pre-screen images with a lightweight vision model before running the full pipeline

**Problem.** The full pipeline (L1 + L2 + L3) takes 10–20 s and costs API tokens on every upload, including images that are unsuitable for photography critique: screenshots, memes, blank images, document scans, or images where the subject is unrecognisable. Currently `api.py` only validates MIME type and file size — it has no semantic content check.

**Proposed solution.** Add a fast pre-screening step using CLIP zero-shot classification before invoking the pipeline. The classifier runs on the already-loaded `_clip_iqa` model (no new cost) and takes ~50 ms:

```python
_REJECT_PROMPTS = [
    "a photograph of a real scene or subject",  # accept
    "a screenshot of a computer screen",        # reject
    "a blank or solid-color image",             # reject
    "a scanned document or text page",          # reject
]

def _is_photograph(tensor: torch.Tensor) -> tuple[bool, str]:
    """Returns (is_photo, reason). Fast CLIP zero-shot check."""
    ...
    if accept_prob < 0.4:
        return False, "Image does not appear to be a photograph"
    return True, ""
```

Return HTTP 422 with a clear error message if the image is rejected, before any expensive computation runs.

**Impact.** Medium. Eliminates wasted pipeline runs on unsuitable content. Also improves user experience by giving an immediate, specific rejection reason instead of a confusing low-quality analysis.

**Complexity.** Low. CLIP is already loaded. The check is a single forward pass. No new dependencies. The rejection threshold (0.4) should be tuned on a small validation set.

---

## 16. Feedback loop for metric weight optimisation

**Problem.** The per-metric weights in `src/config/metrics.py` are hand-tuned constants (e.g. `nima: weight=0.35`, `clip_iqa: weight=0.25`). They were set heuristically and never updated. There is no signal about whether the weights produce quality tiers that match human perception, and no mechanism to improve them without manual inspection and re-deployment.

**Proposed solution.** Introduce a three-part feedback system:

### Part A — Collect explicit ratings

Add a `POST /feedback` endpoint that accepts a structured rating against a prior analysis:

```python
class FeedbackPayload(BaseModel):
    image_hash: str          # SHA-256 of the uploaded image
    overall_rating: int      # 1–5 stars from the user
    tier_agreement: bool     # did the quality tier match the user's expectation?
    metric_flags: dict[str, bool] | None  # optional: per-metric "was this useful?"
```

Store ratings in a lightweight append-only SQLite table (`feedback.db`) via `aiosqlite`. Gate behind `FRAMEIQ_FEEDBACK_ENABLED=true`.

### Part B — Offline weight optimisation

Provide a CLI script `scripts/optimise_weights.py` that reads the feedback table and runs Bayesian optimisation (via `scikit-optimize` or `optuna`) to find weights that minimise the mean squared error between the computed `overall_quality_score` and the collected `overall_rating`:

```python
# Objective: minimise MSE(predicted_tier_score, human_rating) over weight space
def objective(weights: list[float]) -> float:
    scores = [recompute_tier(row, weights) for row in feedback_rows]
    return mean_squared_error(human_ratings, scores)
```

Weights are constrained to sum to 1.0 and each `>= 0`. The optimised weights are written back to a JSON file (`config/weights_optimised.json`) which `metrics.py` can load at startup via the existing `FRAMEIQ_METRICS_CONFIG` env var pattern (Proposal 7).

### Part C — Implicit signal from LLM critique

When vision input is enabled (Proposal 12), extract the LLM's own quality judgement from its response. If the LLM consistently rates an image higher or lower than the computed tier, treat the delta as a soft training signal. Log `(image_hash, computed_tier_score, llm_inferred_score)` to the same feedback table as an implicit row with lower weight than explicit user ratings.

```python
# In synthesizer.py after parsing AnalysisReport:
if report.overall_score and computed_tier.overall:
    delta = report.overall_score - computed_tier.overall
    if abs(delta) > 0.15:   # only log meaningful disagreements
        _log_implicit_feedback(image_hash, computed_tier.overall, report.overall_score)
```

**Impact.** High over time. The quality tier is the primary signal the LLM uses to calibrate its critique tone. Better-calibrated weights produce more accurate tiers, reducing over-praise of mediocre images and over-criticism of strong ones. The improvement compounds: more accurate tiers → better LLM critiques → higher user trust → more feedback submitted.

**Complexity.** Medium. Part A (endpoint + SQLite logging) is low complexity. Part B (optimisation script) requires `optuna` or `scikit-optimize` as a new dev dependency and ~150 lines of script code. Part C depends on Proposal 12 and requires a consistent `overall_score` field in `AnalysisReport`. The optimisation script runs offline and does not affect the hot path.
