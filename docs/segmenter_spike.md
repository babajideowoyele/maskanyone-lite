# Segmenter CPU Spike — EdgeTAM vs. MobileSAM

**Date:** 2026-04-23
**Goal:** decide between MobileSAM (plan §3 default) and EdgeTAM (Apache-2.0,
video-native) for maskanyone-lite's Precision mode. Numbers, not vibes.
**Outcome:** EdgeTAM wins on latency; recommendation below.

---

## Setup

| | |
|---|---|
| Machine | Windows 11, 16-thread CPU (Anaconda Python 3.9 via conda env `maskedpiper-smoke`) |
| PyTorch | 2.4.1+cpu (CPU wheels) |
| Transformers | 4.57.6 |
| Torch threads | 15 (`cpu_count() - 1`) |
| dtype | float32 (bfloat16 lacks HW accel on most x86 CPUs without AVX-512-BF16) |
| Test clip | `ted_kid.mp4` (1920×1080, 11 MB, from Masked-Piper's `Input_Videos/`) |
| Protocol | 30 contiguous leading frames, single positive center-point prompt re-applied per frame (no tracking) |

**Reproduce:**

```bash
cd experiments/edgetam_spike/
python bench_edgetam.py <video.mp4> 30
python bench_mobilesam.py <video.mp4> 30
```

Weights: MobileSAM `mobile_sam.pt` from the official GitHub (39 MB, not
checked in — see `.gitignore`). EdgeTAM pulled from HF
(`yonigozlan/EdgeTAM-hf`, cached under `~/.cache/huggingface/`).

---

## Results

| Metric | EdgeTAM-image | MobileSAM (vit_t) | Winner |
|---|---:|---:|---|
| First-frame ms | 299 | 348 | EdgeTAM |
| Mean ms/frame (ex. first) | **248** | **319** | **EdgeTAM (-22%)** |
| Median ms/frame | 246 | 313 | EdgeTAM |
| p95 ms/frame | 278 | 343 | EdgeTAM |
| Total wall for 30 frames | 9.87 s | 11.43 s | EdgeTAM |
| Load time | 1.8 s | 0.1 s | MobileSAM |
| Peak RSS | 1090 MB | **874 MB** | MobileSAM (-20%) |
| RSS growth (30 frames) | +598 MB | +336 MB | MobileSAM |
| Steady-state RSS reached | frame ~5 | frame ~3 | MobileSAM (faster) |

Raw per-frame logs: `result_edgetam.json`, `result_mobilesam.json`.

### Quality note (subjective)

MobileSAM reports a per-frame IoU score (0.885 → 0.903 across this clip as the
subject stabilizes). EdgeTAM-hf via HF doesn't expose a directly comparable
score, but its mask area tracked the subject's movement across frames without
visible drift. A real quality eval would need ground-truth masks on ≥3 clips;
**this spike didn't do that** — mask-area stability is a weak proxy.

---

## Key finding: the video-model path is blocked

EdgeTAM's headline architectural win — **video-native cross-frame tracking
with memory attention** — was **not measurable** here. Two reasons:

1. The HF-native video model `yonigozlan/edgetam-video-1` currently returns
   **HTTP 401** (private/gated). Only the image variant `EdgeTAM-hf` is
   publicly accessible right now.
2. The official repo [facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM)
   distributes `edgetam.pt` checkpoint + a simple API
   (`build_sam2_video_predictor` → `init_state` → `propagate_in_video`) but
   **README documents only CUDA**. Install is `pip install -e .` which compiles
   a CUDA kernel. CPU path is possible (most SAM 2 ops work on CPU) but
   requires editing the install + patching CUDA-only paths.

So these numbers compare **image-level inference cost only**. The real-world
EdgeTAM advantage (skip full re-encoding of every frame, propagate from memory
bank instead) is unmeasured.

---

## Interpretation

At image-only (per-frame re-encode) on this CPU:

- EdgeTAM is **~22% faster** than MobileSAM. Not a landslide, but meaningful.
- Both sit at **3–4 FPS at 1080p**. Both miss the plan's "1-min clip in
  1–3 min" target at native resolution; **downsampling to 720p should roughly
  double throughput** (encoder cost scales ~linearly with pixel count).
- MobileSAM uses ~20% less RAM and starts in 0.1 s vs. EdgeTAM's 1.8 s —
  small deployment wins.
- Both are Apache 2.0 (EdgeTAM) / Apache 2.0 (MobileSAM). No license
  differentiator.

If the video model becomes accessible (or we do the CPU port work), EdgeTAM's
real advantage kicks in: no re-encoding per frame. That's when the "22×
faster than SAM 2" narrative matters. Without it, EdgeTAM is just a
moderately-faster image SAM.

---

## Recommendation

**Short term (Phase 2, immediate):**
Update [LIGHTWEIGHT_PLAN.md](../LIGHTWEIGHT_PLAN.md) §3 to make **EdgeTAM
(image variant, `yonigozlan/EdgeTAM-hf`) the primary segmenter**, with
MobileSAM retained as a fallback. §7.1 (tracking strategy) stays open — both
candidates are image-only per-frame in this config, so hand-rolled IoU
propagation is still needed.

**Medium term (Phase 3 or dedicated follow-up spike):**
Attempt a CPU port of [facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM)
to unlock video tracking. Estimated ~1 day of work: adapt `pip install -e .`
to skip the CUDA kernel build, add `device="cpu"` plumbing, validate mask
parity against HF image model. If this lands, it replaces the IoU tracker in
§7.1 and becomes the headline architectural simplification.

**Skip:**
Sapiens2 — even the small variants aren't competitive on CPU at sensible
resolutions, and the non-commercial license adds adoption friction for (c)
the wide-adoption research-toolkit positioning.

---

## Risks / caveats this spike did NOT address

- **Single clip, single prompt location.** `ted_kid.mp4` has one clearly
  central subject. Prompt quality matters — edge-case prompts (small object,
  near frame boundary, occluded) could change the relative standings.
- **No quality benchmark.** Both segmenters produced plausible masks; we have
  no IoU-vs-ground-truth measurement.
- **Windows-specific timing.** Linux/Docker CPU performance may differ (in
  practice usually within ±10%).
- **Frame size.** 1080p is the native res of the test clip. At 720p / 480p
  (what a lightweight web app would likely use), numbers will be different.
- **Model warmup.** The 30-frame run came right after a 3-frame dry run, so
  filesystem cache + Python bytecode cache were warm. Cold-start may be
  +100–500 ms on the first `from_pretrained`.

## Follow-up work (not done here)

- [ ] Resolution sweep: benchmark at 480p, 720p, 1080p for both.
- [ ] Quality eval: masks-on-video side-by-side, plus IoU vs. a manually-drawn
      gt mask on 3 clips.
- [ ] CPU port of `facebookresearch/EdgeTAM` video model.
- [ ] Revisit when `yonigozlan/edgetam-video-1` becomes public.
