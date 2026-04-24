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

## Platform notes

Docker is the primary distribution path, but **Docker Desktop on Windows
runs containers inside a Hyper-V / WSL2 VM that taxes CPU-bound ML
inference by roughly 10x** in our measurements (EdgeTAM precision mode:
~2 min native vs ~24 min in Docker Desktop on the same 16-thread CPU).
Linux hosts and native WSL2-Docker are unaffected.

If you're on Windows and need real throughput, you have two options:

1. **Use Docker inside WSL2 directly** (`wsl --install`, install Docker
   Engine inside Ubuntu, skip Docker Desktop). Recovers most of the
   performance.
2. **Bypass Docker for development** using `scripts/native_dev.sh` —
   creates a venv, installs masker deps, lets you run
   `python worker/masker.py <input> <output> blur precision` directly.
   Fastest iteration on Windows, but **not the distributed UX** — don't
   rely on this for reproducibility or deployment.

## License

Apache 2.0. See [LICENSE](LICENSE). Model weights may carry separate licenses —
see the output manifest for any processed video to audit dependencies.
