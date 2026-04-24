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

Recommended order by performance + ease:

1. **Linux + Docker Engine** — simplest and fastest. `apt install docker.io`
   + `docker compose up -d`. No VM overhead; matches what you'd run on a
   research cluster.
2. **Windows + WSL2-native Docker** — `wsl --install`, install Docker
   Engine inside Ubuntu, skip Docker Desktop. Recovers most of the
   performance lost to Docker Desktop's VM.
3. **Windows + Docker Desktop** — works out of the box; precision mode
   is slow but quick mode is fine. Good enough for a single-clip user.
4. **Native Python via `scripts/native_dev.sh`** — fastest iteration on
   Windows, **dev-only**. Not the distributed UX — don't rely on this
   for reproducibility.
5. **macOS + Docker Desktop** — similar VM-overhead caveats to Windows
   Docker Desktop. For Apple Silicon, perf may vary further.

## License

Apache 2.0. See [LICENSE](LICENSE). Model weights may carry separate licenses —
see the output manifest for any processed video to audit dependencies.
