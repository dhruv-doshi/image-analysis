# FrameIQ — Pipeline Improvement Proposals

Prioritised architectural and algorithmic improvements. Prompt-level tweaks are excluded; every proposal here requires a code change.

Each entry follows the pattern: **Problem → Proposed solution → Impact → Complexity → Status**.

---

## Priority summary

| # | Area | Problem | Impact | Complexity | Status |
|---|------|---------|--------|------------|--------|
| 1 | Performance | Layer 1 metrics run sequentially | High | Low | ✅ Done |
| 2 | Performance | Layers 1+2+3 block the async event loop | High | Low | ✅ Done |
| 3 | Performance | rembg U²-Net saliency is the heaviest single step | High | Medium | ✅ Done |
| 4 | Architecture | `_detect_lines()` runs twice per request | Medium | Low | Open |
| 5 | Metric quality | BRISQUE + NIQE are both NSS-based no-reference metrics | Medium | Low | Open |
| 6 | Metric quality | Haar cascade face detector has poor recall/precision | Medium | Medium | Open |
| 7 | Architecture | Metrics config is not environment-driven | Medium | Low | Open |
| 8 | Architecture | `_compute_quality_tier()` returns a raw dict, not a model | Low | Low | Open |
| 9 | Metric quality | Color clustering uses a Lab→BGR→HSV round-trip | Low | Low | Open |
| 10 | Metric quality | Three correlated tonal metrics add noise to the tier | Low | Low | Open |
| 11 | Performance | Result cache for repeated uploads | Low | Low | Open |
| 12 | Vision model | LLM receives only numeric metrics, not the image itself | High | Low | ✅ Done |
| 13 | Vision model | Scene classification uses heuristics; CLIP is already loaded | Medium | Low | Open |
| 14 | Vision model | Numeric metrics can't capture semantic/contextual quality | Medium | Medium | Open |
| 15 | Vision model | Full pipeline runs even for unsuitable images | Medium | Low | ✅ Done |
| 16 | Metric quality | Metric weights are hand-tuned constants with no feedback | High | Medium | Open |
| 17 | Quality tier | Single terrible metric averaged away by clean unrelated scores | High | Low | ✅ Done |
| 18 | LLM prompt | System prompt metric-centric; LLM over-relies on numbers | High | Low | ✅ Done |
| 19 | Evaluation | No systematic comparison of LLM input strategies | Medium | Low | ✅ Done |

---

## 1. Parallel Layer-1 metric execution ✅ Done

**Problem.** `analyse_technical()` evaluated five pyiqa models sequentially. MUSIQ dominates at 2–4 s.

**Solution.** All five `_score_*()` calls run concurrently in a `ThreadPoolExecutor(max_workers=5)`. PyTorch releases the GIL during inference so they run truly in parallel.

**Impact.** L1 wall-clock time dropped from `sum(latencies)` to `max(latencies)` — a 50–70% reduction.

---

## 2. Offload CPU-bound layers to executor ✅ Done

**Problem.** L1, L2, and L3 blocked the FastAPI event loop.

**Solution.** All three layers are wrapped with `asyncio.get_running_loop().run_in_executor(None, fn, *args)` in both `/analyse` and `/analyse/stream`.

**Impact.** Server can handle concurrent requests. 3–5× throughput improvement under load.

---

## 3. Replace rembg saliency with OpenCV spectral residual ✅ Done

**Problem.** rembg U²-Net (~200 MB) took 2–5 s per image.

**Solution.** Default saliency uses `cv2.saliency.StaticSaliencySpectralResidual_create()` (~50–100 ms). rembg is preserved behind `use_heavy_saliency` config flag (default `False`).

**Impact.** L2 latency reduced from ~5 s to ~200 ms in default mode.

---

## 4. Deduplicate line detection (single Canny + HoughLinesP pass)

**Problem.** `_detect_lines()` runs from both `_horizon_tilt()` / scene classification and separately inside `_leading_lines()` — two full edge-detection passes per request.

**Proposed solution.** Compute lines once at the top of `analyse()` and pass the result to all consumers.

**Impact.** ~100–200 ms latency reduction. Simpler call graph.

**Complexity.** Low.

---

## 5. Resolve BRISQUE / NIQE redundancy

**Problem.** Both are Natural Scene Statistics metrics; they correlate strongly (r ≈ 0.75–0.85).

**Recommendation.** Set NIQE `enabled = False` as default, saving ~500 ms. Re-enable for JPEG-heavy workloads where NIQE's artefact sensitivity adds signal.

**Complexity.** Low. One boolean in `METRICS`.

---

## 6. Replace Haar cascade face detector with OpenCV YuNet

**Problem.** Haar cascade has ~30–40% miss rate on rotated, occluded, or profile faces — directly degrading scene classification.

**Proposed solution.** `cv2.FaceDetectorYN` (YuNet DNN), available since OpenCV 4.5.4. ~90% recall on WIDER FACE vs ~60% for Haar. Requires ~1 MB ONNX model at startup.

**Complexity.** Medium.

---

## 7. Make metrics config environment-driven

**Problem.** Enable/disable flags and weights are hard-coded in Python.

**Proposed solution.** Read `FRAMEIQ_METRICS_CONFIG` env var containing a JSON patch applied over the defaults.

**Complexity.** Low. ~10 lines in `metrics.py`.

---

## 8. Wire `_compute_quality_tier()` through the `QualityTier` Pydantic model

**Problem.** Returns a raw `dict`; type errors surface only at serialisation time.

**Proposed solution.** Change return type to `QualityTier` directly.

**Complexity.** Low.

---

## 9. Eliminate the Lab → BGR → HSV round-trip in color harmony

**Problem.** Unnecessary intermediate BGR conversion introduces floating-point error.

**Proposed solution.** Use `skimage.color.lab2rgb` + `skimage.color.rgb2hsv` directly.

**Complexity.** Low. Purely internal to `_color_harmony()`.

---

## 10. Reduce correlated tonal metrics

**Problem.** `histogram_std`, `dynamic_range_stops`, and `contrast_rms` all measure tonal spread — all weight=0, yet sent in the LLM payload.

**Recommendation.** Disable `histogram_std` (`enabled = False`); keep `dynamic_range_stops` and `contrast_rms`. Shorter payload, no information loss.

**Complexity.** Low.

---

## 11. In-memory result cache keyed on image hash

**Problem.** Repeated uploads of the same image re-run the full pipeline.

**Proposed solution.** `hashlib.sha256(raw_bytes)` → check `cachetools.TTLCache(maxsize=100, ttl=3600)`. Gate behind `FRAMEIQ_ENABLE_CACHE=true`.

**Complexity.** Low.

---

## 12. Send the image to the LLM alongside the numeric metrics ✅ Done

**Problem.** The LLM received only numeric scores and couldn't detect focus failures, misattributed causes (e.g. blamed lens optics for a blurred foreground subject), and couldn't assess emotional impact or contextual appropriateness.

**Solution.** `synthesise()` and `synthesise_stream()` accept an optional `pil_image` argument. When provided (always from API endpoints), the PIL image is re-encoded as JPEG at quality=85 and sent as an `image_url` content block alongside the JSON text payload. The system prompt instructs the LLM to treat the photograph as ground truth and flag discrepancies with the metrics.

**Validated by experiment** (`scripts/compare_strategies.py` across 7 evaluation images — see `tests/evaluation/results/strategy_comparison_20260307.md`). The `metrics_and_photo` strategy was clearly superior: it never hallucinated on technical properties and correctly identified subjects, focus failures, and contextual issues that the metrics-only path missed.

**Impact.** Highest qualitative improvement in the pipeline. Root-cause accuracy dramatically improved.

---

## 13. Replace heuristic scene classification with CLIP zero-shot

**Problem.** Haar cascade + geometric heuristics for scene classification have well-documented failure modes (Section 6).

**Proposed solution.** Use the pre-loaded CLIP-IQA+ model for zero-shot classification with candidate text prompts. Run in parallel with the existing Haar check; use Haar only to break ties.

**Complexity.** Low. No new dependencies — CLIP is already loaded.

---

## 14. Add a semantic quality layer using the vision model

**Problem.** Numeric metrics cannot determine whether blur is creative intent or a mistake, whether underexposure is deliberate moodiness, or whether leading lines guide the eye to something meaningful.

**Proposed solution.** Extend the system prompt with a semantic verification block (now that vision input is live — Proposal 12 complete). Ask the model to make explicit judgements on focus intent, exposure intent, and grain vs. artefact before generating the report.

**Complexity.** Low (prompt addition only, no new API call). Depends on Proposal 12 ✅.

---

## 15. Pre-screen images with a lightweight vision model ✅ Done

**Problem.** Screenshots, blank images, and document scans burned the full pipeline.

**Solution.** `_is_photograph(bgr_array, tensor)` runs a two-tier check before L1:
1. **Statistical tier**: `std(grayscale) < 8` → immediate reject.
2. **CLIP tier**: optional CLIP zero-shot classification (controlled by `_clip_prescreener` flag).

Returns HTTP 422 with a descriptive reason on rejection. Controlled by `FRAMEIQ_PRESCREENING` env var.

**Impact.** Unsuitable images are rejected in <100 ms with no LLM cost.

---

## 16. Feedback loop for metric weight optimisation

**Problem.** Per-metric weights are hand-tuned constants with no feedback signal.

**Proposed solution (3 parts):**
- **Part A**: `POST /feedback` endpoint storing per-analysis user ratings in SQLite via `aiosqlite`.
- **Part B**: Offline `scripts/optimise_weights.py` using Bayesian optimisation (optuna) to minimise MSE between computed tier scores and human ratings.
- **Part C**: When vision input is enabled, log implicit feedback when the LLM's qualitative assessment diverges significantly from the computed tier.

**Impact.** High over time — better-calibrated weights → more accurate tiers → better LLM tone calibration.

**Complexity.** Medium. Part A is low; Part B needs optuna; Part C depends on Proposal 12 ✅.

---

## 17. Quality tier critical failure gates ✅ Done

**Problem.** A single catastrophic metric (e.g. Laplacian=49, terrible blur) was being averaged away by clean-but-irrelevant scores (BRISQUE=31, excellent; noise σ=0.37, excellent), producing "average" for a fundamentally broken image.

**Solution.** Three hard gates applied after the weighted average in `_compute_quality_tier()`:

| Condition | Effect |
|-----------|--------|
| Sharpness ordinal = 4 (terrible) | `overall = max(overall, 3)` — at least "poor" |
| Exposure ordinal = 4 (terrible) | `overall = max(overall, 3)` — at least "poor" |
| NIMA ordinal ≥ 2 (average or below) | `overall = max(overall, 1)` — at most "good"; technical perfection cannot produce "excellent" if humans rate it aesthetically average |

**Impact.** Quality tier now correctly reflects capture-critical and aesthetic failures. Two new tests in `tests/test_llm.py` lock in the gate behaviour.

---

## 18. System prompt overhaul — image-first, critical stance ✅ Done

**Problem.** `prompts/system.md` was metric-centric: it used `quality_tier` to set the LLM's tone (softening critique for "good" images), instructed the model to "analyse from quantitative metrics," and allowed metric names (BRISQUE, NIMA) to appear in output prose.

**Solution.** Prompt rewritten with three key changes:
1. **Image-first**: "Your visual judgement comes first. Use metrics to confirm or quantify what you see; never cite a metric number as a substitute for describing a real visual problem."
2. **Removed the `quality_tier` tone-anchor table**: The LLM now forms its own overall assessment from the photograph, not from a pre-computed tier.
3. **Critical default posture**: "Every image has meaningful problems — find them." No softening. Summary must open with the dominant flaw. `improvements` is exactly 3, ranked by impact, each concretely actionable.

**Impact.** LLM critiques are substantially more honest, contextually grounded, and useful. The mismatch between a harsh written critique and a "good" or "excellent" tier label is resolved at both ends.

---

## 19. LLM input strategy comparison experiment ✅ Done

**Problem.** No empirical basis for choosing between metrics-only, multimodal, or photo-only LLM input.

**Solution.** `scripts/compare_strategies.py` runs all three strategies in parallel (via `ThreadPoolExecutor`) on a local image and saves results to `prompts/comparisons/`. `src/llm/comparison.py` implements the three synthesisers against a lean comparison system prompt (`prompts/compare_system.md`).

Tested on 7 representative images across architecture, concert, snapshot, lifestyle, product, wildlife, and kitchen scenes. Key findings:
- `metrics_and_photo` was superior on every image: accurate scene understanding + precise technical grounding, no hallucinations.
- `photo_only` hallucinated on technical properties (rated a Laplacian=62 image "sharp").
- `metrics_only` correctly measured but misattributed causes (blamed a lens for blur caused by a blurred foreground subject).

Full report: `tests/evaluation/results/strategy_comparison_20260307.md`
