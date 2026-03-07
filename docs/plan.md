# FrameIQ — Implementation Plan

Top proposals selected from `improvements.md`, ranked by impact/complexity ratio.

---

## Completed

All five original proposals plus three additional improvements have shipped.

### ✅ Proposal 2 — Offload blocking layers to executor

All three layers (L1, L2, L3) wrapped with `run_in_executor()` in both `/analyse` and `/analyse/stream`. Event loop never blocked. 3–5× throughput improvement under concurrent load.

**Files changed:** `api.py`

---

### ✅ Proposal 1 — Parallel Layer-1 metric execution

Five pyiqa scorers (BRISQUE, NIMA, CLIP-IQA+, MUSIQ, NIQE) run concurrently in a `ThreadPoolExecutor(max_workers=5)`. L1 wall-clock time is now `max(latencies)` rather than `sum(latencies)`.

**Files changed:** `src/analysis/technical.py`

---

### ✅ Proposal 3 — Replace rembg saliency with OpenCV spectral residual

Default saliency uses OpenCV spectral residual (~100 ms). rembg U²-Net preserved behind `use_heavy_saliency` config flag. L2 latency dropped from ~5 s to ~200 ms.

**Files changed:** `src/analysis/composition.py`, `src/config/metrics.py`

---

### ✅ Proposal 15 — Pre-screen images with CLIP

`_is_photograph(bgr_array, tensor)` performs a two-tier check (statistical std + optional CLIP zero-shot). Unsuitable images rejected with HTTP 422 in <100 ms before any expensive computation runs.

**Files changed:** `src/analysis/technical.py`, `api.py`

---

### ✅ Proposal 12 — Send image to LLM (multimodal Layer 3)

`synthesise()` and `synthesise_stream()` now accept `pil_image` and build a multimodal content list (JSON metrics text + base64 JPEG). Both API endpoints pass the PIL image through. The system prompt instructs the LLM to treat the photograph as ground truth.

Validated by a systematic three-strategy comparison experiment across 7 evaluation images before shipping (see `tests/evaluation/results/strategy_comparison_20260307.md`). The `metrics_and_photo` strategy was clearly superior on every image.

**Files changed:** `src/llm/synthesizer.py`, `src/llm/client.py`, `api.py`, `prompts/system.md`

---

### ✅ Proposal 17 — Quality tier critical failure gates

Three hard gates added to `_compute_quality_tier()` to prevent catastrophic single-metric failures from being averaged away:
- Terrible sharpness → overall at least "poor"
- Terrible exposure → overall at least "poor"
- NIMA average or below → overall at most "good"

**Files changed:** `src/llm/synthesizer.py`, `tests/test_llm.py`

---

### ✅ Proposal 18 — System prompt overhaul

`prompts/system.md` rewritten: image-first posture, removed `quality_tier` tone anchor, critical default stance, no metric names in output prose, exactly 3 concrete improvements.

**Files changed:** `prompts/system.md`

---

### ✅ Proposal 19 — LLM input strategy comparison experiment

`scripts/compare_strategies.py` + `src/llm/comparison.py` + `prompts/compare_system.md` added. Provides a reproducible framework for testing LLM input strategy changes against real images. Results saved to `prompts/comparisons/` and `tests/evaluation/results/`.

**Files changed:** `scripts/compare_strategies.py`, `src/llm/comparison.py`, `prompts/compare_system.md`

---

## Actual cumulative impact (measured)

| After step | L1 latency | L2 latency | L3 | Total (typical) | Throughput | Output quality |
|------------|-----------|-----------|-----|----------------|------------|----------------|
| Baseline | ~6 s (sequential) | ~5 s (rembg) | ~5 s | ~16 s | 1 req at a time | Metrics-only LLM |
| + #2 executor | ~6 s | ~5 s | ~5 s | ~16 s | 3–5× concurrent | No change |
| + #1 parallel L1 | ~2–3 s | ~5 s | ~5 s | ~12 s | 3–5× concurrent | No change |
| + #3 fast saliency | ~2–3 s | ~0.2 s | ~5 s | ~7 s | 3–5× concurrent | No change |
| + #15 pre-screen | ~2–3 s | ~0.2 s | ~5 s | ~7 s (unsuitable: <0.1 s) | 3–5× concurrent | Rejects non-photos |
| + #12 multimodal | ~2–3 s | ~0.2 s | ~6 s | ~8 s | 3–5× concurrent | Substantially richer, contextually accurate |
| + #17 tier gates | — | — | — | — | — | Tier labels match critique severity |
| + #18 prompt | — | — | — | — | — | More critical, honest, actionable |

---

## Open proposals (prioritised)

| # | Proposal | Why next |
|---|----------|----------|
| 14 | Semantic quality layer (prompt addition) | Proposal 12 is live — add the semantic verification block to the system prompt now that the model sees the image |
| 6 | Replace Haar cascade with YuNet | Scene classification accuracy directly affects LLM genre emphasis; medium complexity |
| 13 | CLIP zero-shot scene classification | Lower complexity than #6, uses already-loaded model |
| 16 | Feedback loop for weight optimisation | Highest long-term impact; needs user data first |
| 4 | Deduplicate line detection | ~100–200 ms saving; low complexity; good cleanup |
| 7 | Env-driven metrics config | Operational nicety for cost tiering / A/B testing |
| 11 | Result cache | Useful for dev iteration speed |

---

## Skipped proposals (still open, lower priority)

| Proposal | Reason |
|----------|--------|
| #5 — Disable NIQE | Saves ~500 ms; pipeline is already fast enough |
| #8 — QualityTier return type | Low risk; defer until model layer is otherwise touched |
| #9 — Lab→HSV round-trip | Correctness improvement only; no user impact |
| #10 — Reduce tonal metrics | Minor payload reduction; low priority |
