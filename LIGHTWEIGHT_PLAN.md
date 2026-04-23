# MaskAnyone-Lite: CPU-Friendly De-Identification Toolbox

**Purpose of this document:** briefing to bootstrap a new repository derived from MaskAnyone, targeting researchers and practitioners who need robust video de-identification **without a consumer GPU**.

---

## 1. Motivation

The current MaskAnyone pipeline (upstream: https://github.com/MaskAnyone/MaskAnyone, branch `samhack`) depends on SAM2, RTMPose, and OpenPose — all of which assume an NVIDIA GPU with substantial VRAM. On CPU:

- SAM2 runs at ~1–2 s per image (unusable for anything longer than a toy clip)
- RTMPose and OpenPose are impractical
- FFmpeg h264_nvenc unavailable; libx264 is fine but slower

For most academic users, a GPU is not available. A de-identification tool that requires a 5090 cannot be adopted broadly. This repo is the CPU-first variant.

## 2. Scope

### In scope

- A working web app + worker pipeline that runs on **any modern laptop/desktop with Docker**.
- Two masking modes:
  1. **Quick mode** — no prompts, person-only segmentation via MediaPipe Selfie Segmentation. Near-real-time.
  2. **Precision mode** — point-prompt segmentation via MobileSAM, SAM-compatible API.
- MediaPipe-based pose/face/hand overlays (already CPU-native in upstream).
- FFmpeg with libx264 encoding (CPU).
- Docker Compose setup that starts in <2 minutes on a laptop, no GPU dependencies.
- Permissive licensing end-to-end — **no AGPL dependencies**.

### Out of scope

- SAM2, RTMPose, OpenPose (upstream MaskAnyone covers the GPU path).
- YOLO-based object detection (AGPL). Replace with MediaPipe or RTMDet-lite if a detection helper is needed.
- Keycloak auth (upstream's profile-gated feature; not needed for a lightweight local tool).
- GPU code paths, CUDA toolchain, NVENC.

### Non-goals

- Matching SAM2 output quality. Precision mode is "good enough" for common de-identification needs, not state-of-the-art.
- Processing hour-long videos in minutes. Realistic target: a 1-minute clip processes in 1–3 minutes on an 8-core laptop.

## 3. Architectural Decisions (already made — do not re-litigate)

| Decision | Chosen | Reason |
|---|---|---|
| Primary segmenter | **EdgeTAM-image** (`yonigozlan/EdgeTAM-hf`, Apache 2.0) | Per-frame inference ~22% faster than MobileSAM on the same CPU (248 vs 319 ms at 1080p, float32). MobileSAM kept as fallback. Evidence: [docs/segmenter_spike.md](docs/segmenter_spike.md). Future spike: adapt [facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM) to CPU to unlock built-in video tracking. |
| Secondary segmenter (fallback) | **MobileSAM** (Apache 2.0) | Smaller RAM footprint (874 vs 1090 MB peak), 18× faster cold start; useful for low-RAM targets or if EdgeTAM-hf becomes unavailable |
| Zero-prompt fallback | **MediaPipe Selfie Segmentation** (Apache 2.0) | Already in upstream's worker image; ~20 ms/frame on CPU |
| Pose estimation | **MediaPipe** pose/face/hand landmarkers | Already CPU-native in upstream |
| Object detection (replacing YOLO) | **MediaPipe Object Detector** or **RTMDet-nano** | Both Apache 2.0; avoids AGPL from ultralytics |
| Video encoding | **libx264** via FFmpeg | Already available, no GPU needed |
| Service orchestration | **Docker Compose** with a single default profile | No profiles needed — every service is CPU |
| Backend framework | **FastAPI** (inherited from upstream) | Mature, well-understood |
| Frontend | **React + MUI** (inherited from upstream) | Reuse upstream's UI components; swap API endpoints |
| DB | **PostgreSQL** (inherited) | Job queue + metadata, unchanged from upstream |
| License | **Apache 2.0** for all new code; verify all runtime deps are Apache/MIT/BSD | Enables unrestricted research use and redistribution |

## 4. Relationship to Upstream MaskAnyone

This is a **new repository**, not a fork of the `samhack` branch. Reasons:

1. Upstream is actively developed for GPU; we don't want to carry its SAM2/RTMPose/OpenPose code forever.
2. The architectural shape is different: one segmenter service instead of three, no GPU capabilities block, no CUDA images.
3. AGPL hygiene — starting fresh lets us audit every dep from day one.

**What to copy from upstream:**

- Frontend UI scaffolding (`frontend/` — React components, routing, video player, prompt overlay) — but strip SAM2-specific UI.
- Backend skeleton (`backend/` — FastAPI app, models, routers, worker endpoints) — but replace the segmentation router.
- Postgres schema (`docker/postgres/docker-entrypoint-initdb.d/prototype.sql`).
- Job queue pattern from upstream's `worker/worker.py` and `worker/processing/worker_process.py` — proven, simple.
- FFmpeg conversion logic (`worker/masking/ffmpeg_converter.py`) — strip the `h264_nvenc` branch.
- MediaPipe landmarker code (`worker/masking/media_pipe_landmarker.py`, `media_pipe_pose_masker.py`).
- Mask renderer and pose renderer (`worker/masking/mask_renderer.py`, `pose_renderer.py`).

**What NOT to copy:**

- `sam2/` directory (the SAM2 service).
- `rtmpose/`, `openpose/` directories.
- `docker/sam2/`, `docker/rtmpose/`, `docker/openpose/` build contexts.
- Anything under `worker/` that imports `ultralytics` (it's YOLO/AGPL).
- GPU `deploy.resources.reservations.devices` blocks in docker-compose.
- `docker-compose-cli.yml`.

## 5. Target Architecture

```
┌──────────┐   ┌──────────┐   ┌──────────┐
│  nginx   │   │   yarn   │   │ pgadmin  │
│  (TLS)   │   │  (dev)   │   │  (opt)   │
└────┬─────┘   └──────────┘   └─────┬────┘
     │                              │
     ▼                              ▼
┌──────────┐                   ┌──────────┐
│  python  │──────────────────▶│ postgres │
│ (FastAPI)│                   │          │
└────┬─────┘                   └──────────┘
     │
     │ HTTP
     ▼
┌──────────┐       ┌──────────────────┐
│  worker  │──────▶│  segmenter       │
│          │       │  (MobileSAM +    │
│          │       │   MediaPipe      │
│          │       │   Selfie)        │
└──────────┘       └──────────────────┘
```

**Service count: 6** (vs upstream's 9+). One segmenter service replaces three GPU services.

## 6. Deliverables — Implementation Order

### Phase 1: Scaffolding (Day 1)

- [ ] Initialize new repo `maskanyone-lite` with Apache 2.0 LICENSE and README stub.
- [ ] Copy upstream's frontend, backend, worker, nginx, postgres, pgadmin directories as described above. Commit as "initial import".
- [ ] Strip GPU references from `docker-compose.yml`: no `deploy.resources.reservations.devices` blocks, no NVIDIA env vars.
- [ ] Remove the `sam2`, `rtmpose`, `openpose` services entirely from compose.
- [ ] Remove ultralytics/YOLO imports from the worker; remove `docker/worker/Dockerfile` lines that download yolo11x-pose.pt.
- [ ] Verify the stack comes up with `docker compose up -d`. Basic mediapipe masking should already work (upstream's `basic_masking` flow).
- [ ] Add healthchecks to `postgres` and `python` (see upstream commit that introduced these — use `pg_isready` and `wget http://localhost:8000/platform/mode`).

**Phase 1 exit criteria:** `docker compose up -d` succeeds on a Linux host with no NVIDIA drivers; upstream's `basic_masking` mode runs a 10-second test clip end-to-end.

### Phase 2: Segmenter Service (Days 2–3)

- [ ] Create `docker/segmenter/Dockerfile` based on `python:3.11-slim`. Install: `torch` (CPU wheel), `transformers`, `opencv-python-headless`, `mediapipe`, `fastapi`, `uvicorn`, `pillow`.
- [ ] Create `segmenter/main.py` with FastAPI endpoints:
  - `POST /segmenter/selfie` — takes video bytes, returns NPZ of per-frame person masks (MediaPipe Selfie).
  - `POST /segmenter/mobilesam/image` — takes image + point prompts, returns mask.
  - `POST /segmenter/mobilesam/video` — takes video + prompts, returns per-frame masks (frame-by-frame with optional tracker).
- [ ] Pre-download MobileSAM checkpoint (`mobile_sam.pt`, ~40 MB) in the Dockerfile so first run doesn't pay download latency.
- [ ] Add `@app.on_event("startup")` that loads MobileSAM into memory once (model is small, ~15 MB RAM resident).
- [ ] Add `GET /segmenter/health` endpoint for compose healthcheck.
- [ ] Wire into `docker-compose.yml` with a healthcheck.

**Phase 2 exit criteria:** curl against `/segmenter/selfie` with a 5-second clip returns masks in <15 seconds on 8-core CPU; curl against `/segmenter/mobilesam/image` with a single point prompt returns a mask in <500 ms.

### Phase 3: Worker Integration (Days 3–4)

- [ ] In `worker/processing/worker_process.py`, introduce two new job types: `selfie_masking` and `mobilesam_masking`. Keep upstream's `basic_masking` (MediaPipe pose) as the simplest fallback.
- [ ] New module `worker/masking/selfie_masker.py` — mirrors structure of upstream's `sam2_pose_masker.py` but calls `/segmenter/selfie`.
- [ ] New module `worker/masking/mobilesam_masker.py` — frame-by-frame MobileSAM. Include a simple IoU-based frame-to-frame tracker so prompts from frame 0 follow the subject for short clips.
- [ ] Re-use `worker/masking/mask_renderer.py` from upstream unchanged.
- [ ] Strip SAM2Client, OpenposeClient, RtmposeClient from worker imports.

**Phase 3 exit criteria:** a video uploaded via the UI with "Selfie quick mask" or "Prompted precision mask" selected processes end-to-end on CPU and produces a playable blurred output.

### Phase 4: Backend & UI (Days 4–5)

- [ ] Update backend's `platform_router.py` to include a `/platform/capabilities` endpoint that returns `{"segmenters": ["selfie", "mobilesam"]}`.
- [ ] Frontend: update the masking editor to show only two modes — Quick and Precision. Remove SAM2 model variant selector, RTMPose/OpenPose overlay options.
- [ ] Frontend: when mode = Precision, reuse upstream's point-prompt UI unchanged (same click-to-add-point interaction as SAM2 flow).
- [ ] Frontend: remove chunk-size controls. MobileSAM processes frame-by-frame; no chunking needed. For Selfie mode, no chunking needed either.
- [ ] Backend: remove job fields related to SAM2 model variant, chunk size, RTMPose/OpenPose overlay selection. Keep the masking pipeline parameters simple: `{mode, hidingStrategy, prompts?}`.

**Phase 4 exit criteria:** end-user can open https://localhost, upload a video, pick a mode, optionally place prompts, run a job, and see the masked result — without ever touching a GPU.

### Phase 5: Hardening & Release Prep (Days 5–7)

- [ ] Add a smoke test script `scripts/smoke_test.sh` that uploads a bundled 5-second test clip, runs a selfie-mode job, and asserts the output exists with the expected duration. Run in CI.
- [ ] Add a second smoke test for MobileSAM mode with a hardcoded prompt.
- [ ] Add `scripts/license_audit.sh` that walks all `requirements.txt`, `package.json`, and Dockerfile `apt-get install` lists, cross-references against a known-good Apache/MIT/BSD list, and flags anything AGPL or GPL.
- [ ] Write `docs/OUTPUT_MANIFEST.md`: every masked video should have a sidecar JSON with: model names, model checksums, masking config, git SHA of this repo, software version, timestamp. This is needed for research reproducibility.
- [ ] Implement the manifest generation in the worker's `_process_job` after upload.
- [ ] Write `README.md` with: what this is, what it's not (pointing to upstream for GPU), hardware requirements (minimum 4-core CPU, 8 GB RAM, 10 GB disk), setup steps, example use.
- [ ] Write `docs/DIFFERENCES.md` explaining how this differs from upstream MaskAnyone (shorter than you'd think — mostly "no GPU, simpler pipeline").

**Phase 5 exit criteria:** CI green; `bash setup.sh` works on Ubuntu 22.04, macOS (Docker Desktop), and Windows 11 (Docker Desktop + WSL2); smoke tests pass on all three.

## 7. Open Decisions (for the implementer to make)

1. **Video tracking strategy for EdgeTAM-image / MobileSAM.** Both primary and fallback segmenters are image-only in the current configuration — they need cross-frame tracking glue. Options: frame-by-frame with re-prompt every N frames, IoU-based mask propagation, lightweight tracker (ByteTrack). Start with IoU propagation; document flicker as a known limitation. A separate spike to adapt [facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM) to CPU would unlock EdgeTAM's built-in memory-attention tracking and obsolete this decision — but it's not Phase 1 scope.

2. **Whether to keep pgadmin.** Upstream ships it. For a lightweight release, dropping it saves ~200 MB image and a port. Recommend: drop by default, add back behind a compose profile if needed.

3. **Default masking strategy.** Upstream has many (`solid_fill`, `blurring`, `pixelation`, `contours`, `none`). For lite, start with blurring as default and offer pixelation as alternative. Others can come later.

4. **Frontend dev server vs. static build in production.** Upstream runs `yarn start` (CRA dev server) inside a container. For a release image, a static build served by nginx is smaller and faster. Recommend: keep upstream's dev-server pattern during development, add a "release" compose override that uses a static build.

5. **Whether to support RGB vs. grayscale masking output.** Upstream outputs RGB. No reason to change unless a user explicitly asks.

## 8. Testing Requirements

The implementer must test:

- **Cold start:** `docker compose down -v && docker compose up -d` succeeds in <2 minutes on a laptop.
- **Selfie mode on a 30-second clip** completes in <60 seconds on 8-core CPU.
- **MobileSAM mode on a 10-second clip** with 2 point prompts completes in <30 seconds on 8-core CPU.
- **Memory peak stays under 6 GiB** during any single job on a laptop-class machine.
- **Output manifest exists** for every completed job and contains all required fields.
- **License audit script returns zero AGPL/GPL dependencies.**

## 9. Deliberate Non-Decisions

The following are **explicitly left to upstream MaskAnyone** and should not be reimplemented here:

- Kinematics export (upstream's `result_mp_kinematics`)
- Blendshapes export
- Multi-object tracking across long videos with ID consistency (hard problem, not lite's target)
- Voice/audio masking beyond "preserve" or "strip"
- Server-mode with Keycloak
- CLI entrypoint (users interact via the web UI)

If requested later, document the request and keep the codebase focused.

## 10. Distribution Plan

- **GitHub repo:** `maskanyone-lite` under your org.
- **License:** Apache 2.0.
- **Container images:** Eventually publish to `ghcr.io/<org>/maskanyone-lite/*` once stable. Not in Phase 1–5.
- **Paper/citation:** If you plan to cite this in research, add a `CITATION.cff` file in Phase 5.
- **Issue templates:** Add `bug`, `feature`, `question` templates. Make it clear that "my GPU use case" should go upstream.

## 11. What to Ask the User

Before starting, the implementer should confirm:

1. Is the new repo's name `maskanyone-lite`, or something else?
2. Is Apache 2.0 the preferred license (vs. MIT or BSD-3)?
3. Is MobileSAM acceptable as the primary segmenter, or does the user want to also evaluate MediaPipe's Image Segmenter task as an alternative first?
4. Target platforms for Phase 5 smoke tests — Ubuntu / macOS / Windows all three, or a narrower initial set?
5. Does the user want a CLI mode, or web-UI only?

## 12. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| MobileSAM output quality insufficient for research de-identification | Medium | Prototype on 5 representative clips in Phase 2 before committing. If quality is poor, switch to full-frame SAM2-tiny on CPU with explicit "this is slow" UI warning. |
| Frame-by-frame segmentation flickers noticeably | High | Accept and document; add temporal mask smoothing as Phase 6 if users complain. |
| MediaPipe Selfie misses partial occlusions (person behind object) | High | Document limitation; recommend Precision mode for complex scenes. |
| CPU peak memory exceeds laptop RAM on long videos | Medium | Implement upstream's chunk-based streaming for MobileSAM video mode (adapt `_mask_streaming` from `sam2_pose_masker.py`). |
| AGPL creeping in via transitive dep | Low–Medium | `scripts/license_audit.sh` in CI catches this. |

## 13. Handoff Notes

When picking this up in a fresh session:

1. **Read this document fully before touching code.** Every decision above is already made.
2. **Start with Phase 1.** Don't skip ahead. The goal is to have a working CPU-only upstream-derived stack before introducing new segmenters.
3. **Keep the PRs small.** One phase per PR if the user is reviewing, or one coherent chunk per commit if not.
4. **Ask the user questions in §11 before writing any code.**
5. **When in doubt about whether a dep is AGPL, check.** Do not assume. Many ML libraries have surprising licenses.
6. **Upstream is at** `c:/Users/babaj/Documents/projects/masking/MaskAnyone` (branch `samhack`). Read any code you're copying, don't blindly copy.
