# FrameIQ — Implementation Plan

Top 5 proposals selected from `improvements.md`, ranked by impact/complexity ratio.

**Recommended implementation order:** 2 → 1 → 3 → 15 → 12

The performance proposals (2, 1, 3) stabilise the pipeline first; pre-screening (15) protects it; multimodal input (12) then transforms output quality on a now-fast foundation.

---

## 1. Proposal 12 — Send image to LLM (multimodal Layer 3)

**Impact: High | Complexity: Low**

The single biggest qualitative leap. The LLM currently reasons entirely from numbers — it can't detect chromatic aberration, assess emotional impact, or distinguish creative intent from technical failure. Sending the image fixes all of that with no new dependencies and no pipeline restructuring. Everything else in the pipeline feeds into this improvement.

**File changes:** `src/llm/synthesizer.py`, `src/llm/client.py`, `prompts/system.md`

**Gate behind:** `FRAMEIQ_VISION_INPUT=true`

---

## 2. Proposal 2 — Offload blocking layers to executor

**Impact: High | Complexity: Low**

Currently a single request blocks all other requests for 10–20 s. Two lines of `run_in_executor()` in `api.py` give a 3–5× throughput improvement under concurrent load. No signature changes, no new dependencies — the highest ROI code change in the list.

**File changes:** `api.py` only

---

## 3. Proposal 1 — Parallel Layer-1 metric execution

**Impact: High | Complexity: Low**

Pairs directly with Proposal 2. L1 wall-clock time drops from `sum(latencies)` to `max(latencies)` — a 50–70% reduction. PyTorch already releases the GIL during inference so thread safety is not a concern. Models are already module-level singletons.

**File changes:** `src/analysis/technical.py` only

---

## 4. Proposal 3 — Replace rembg saliency with OpenCV spectral residual

**Impact: High | Complexity: Medium**

rembg is the single heaviest step in the entire pipeline (2–5 s, 200 MB model). The interface of `_saliency_map()` doesn't change; only the internals swap. Keeping rembg behind `USE_HEAVY_SALIENCY=true` preserves the option for high-precision use cases. Combined with Proposals 1 and 2, total latency could drop from ~15 s to ~3–4 s.

**File changes:** `src/analysis/composition.py`, `.env.example`

---

## 5. Proposal 15 — Pre-screen images with CLIP

**Impact: Medium | Complexity: Low**

Guards the pipeline entry point. Screenshots, blank images, and document scans currently burn 10–20 s and an API call before producing a meaningless result. CLIP is already loaded; this is a single forward pass (~50 ms) returning HTTP 422 before any expensive computation starts.

**File changes:** `src/analysis/technical.py` (add `_is_photograph()`), `api.py` (call before pipeline)

---

## Skipped proposals and rationale

| Proposal | Reason skipped |
|----------|----------------|
| #5 — Disable NIQE | Saves 500 ms against a 15 s pipeline; minor relative to above |
| #7 — Env-driven metrics config | Operational nicety, not a user-facing improvement |
| #13 — CLIP scene classification | Good, but scene accuracy has lower downstream impact than the top 5 |
| #14 — Semantic quality layer | Depends on #12; implement as a follow-on once #12 ships |
| #16 — Feedback loop | High long-term value but needs user data first; revisit after above ship |

---

## Expected cumulative impact

| After step | Latency (typical image) | Throughput | Output quality |
|------------|------------------------|------------|----------------|
| Baseline | ~15 s | 1 req at a time | Numbers-only LLM input |
| + #2 (executor) | ~15 s | 3–5× concurrent | No change |
| + #1 (parallel L1) | ~10 s | 3–5× concurrent | No change |
| + #3 (rembg swap) | ~3–4 s | 3–5× concurrent | No change |
| + #15 (pre-screen) | ~3–4 s (unsuitable: <100 ms) | 3–5× concurrent | No change |
| + #12 (multimodal) | ~4–5 s (+ base64 encode) | 3–5× concurrent | Substantially richer |
