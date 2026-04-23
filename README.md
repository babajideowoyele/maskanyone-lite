# maskanyone-lite

CPU-friendly video de-identification toolbox. A lightweight research variant of
[MaskAnyone](https://github.com/MaskAnyone/MaskAnyone) that runs on any modern
laptop/desktop with Docker, without requiring a GPU.

**Status: scaffolding — not yet functional.** See [LIGHTWEIGHT_PLAN.md](LIGHTWEIGHT_PLAN.md)
for the design document and [phase breakdown](LIGHTWEIGHT_PLAN.md#6-deliverables--implementation-order).

## What this is

- A self-contained web app + worker pipeline for masking faces, bodies, and
  hands in video for research de-identification.
- CPU-only by design. Targets ~1-minute clip in ~1–3 minutes on an 8-core laptop.
- Two masking modes:
  - **Quick** — zero-prompt person segmentation via MediaPipe Selfie Segmentation.
  - **Precision** — point-prompt segmentation via a lightweight SAM-compatible
    model (MobileSAM or EdgeTAM — see spike notes).
- Produces pose/face/hand overlays and sidecar landmark timeseries.

## What this is not

- A matchless-quality segmenter. Use upstream [MaskAnyone](https://github.com/MaskAnyone/MaskAnyone)
  if you have a GPU and need SAM 2 quality.
- A commercial product. License is Apache 2.0; weights may carry their own licenses.

## Hardware targets

- Minimum: 4-core CPU, 8 GB RAM, 10 GB disk.
- Recommended: 8-core CPU, 16 GB RAM, Docker Desktop or native Docker Engine.

## License

Apache 2.0. See [LICENSE](LICENSE). Model weights may carry separate licenses —
see the output manifest for any processed video to audit dependencies.
