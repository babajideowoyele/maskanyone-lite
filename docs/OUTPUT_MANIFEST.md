# Output Manifest Schema

Every masked video shipped by maskanyone-lite carries a sidecar
`manifest.json` inside its download zip. The manifest records enough
provenance for a researcher to **cite the output, reproduce it later,
and audit the dependency license posture**.

## Why it exists

- **Reproducibility.** Pins input checksum, model IDs, model versions,
  and the git SHA of the code that produced the output. Five years from
  now you can read a manifest and know exactly what produced the file.
- **Citability.** Papers need this; sidecar JSON saves re-running a job
  just to fill in a methods section.
- **Audit.** `models.segmenter` names the actual model used. If you
  used a research-licensed weight, the manifest says so — useful if a
  downstream user cares about license chain.

## Schema (version 1)

```jsonc
{
  "schema_version": 1,
  "generated_at_utc": "2026-04-24T05:00:40Z",

  "input": {
    "original_filename": "ted_kid.mp4",   // what the user uploaded as
    "basename":         "7c6b….mp4",       // on-disk name while processing
    "sha256":           "fd79…f27f"        // canonical identity; survives renames
  },

  "output": {
    "basename":   "7c6b….mp4",
    "frames":     210,
    "mode":       "quick" | "precision",
    "strategy":   "blur" | "solid" | "pixelate" | "skeleton",
    "prompt_xy":  null | [x, y],            // set only for manual-prompt precision
    "downsample": 1.0,                      // 1.0 = native input res for segmenter
    "track_ids":  [1, 2],                   // one id per person detected (precision+detect path)

    // New in the detect-then-segment precision path:
    "detection_max_per_frame":  1,          // null if detection wasn't run
    "frames_with_no_detection": 0           // null if detection wasn't run
  },

  "runtime": {
    "duration_seconds": 19.9,
    "python":           "3.11.15",
    "numpy_version":    "1.26.4"
  },

  "software": {
    "repo":    "maskanyone-lite",
    "git_sha": "19de441aae…"                // "unknown" if env var not injected at run time
  },

  "models": {
    // Shape depends on mode:

    // mode = "quick":
    "segmenter":         "mediapipe.solutions.selfie_segmentation",
    "model_selection":   1,
    "mediapipe_version": "0.10.14"

    // mode = "precision" (either manual-prompt or detect-then-crop):
    "segmenter":              "yonigozlan/EdgeTAM-hf",
    "framework":              "transformers",
    "transformers_version":   "4.57.6",
    "torch_version":          "2.4.1+cpu"
  }
}
```

## Field stability

- `schema_version` is the contract. Bumping it signals a breaking change.
  Consumers should check this field before parsing.
- Fields with `null` values are present but inapplicable to the run (e.g.,
  `prompt_xy` on a quick-mode job).
- New optional fields (e.g., `detection_max_per_frame`) may be added at
  the same `schema_version` if they don't break existing consumers. Any
  removal or semantic change requires a version bump.

## How the git SHA is injected

The worker reads `MASKANYONE_LITE_GIT_SHA` from its environment. The
compose file propagates it from the host:

```yaml
worker:
  environment:
    - MASKANYONE_LITE_GIT_SHA=${MASKANYONE_LITE_GIT_SHA:-unknown}
```

Inject it at the host:

```bash
export MASKANYONE_LITE_GIT_SHA=$(git rev-parse HEAD)
docker compose up -d
```

If absent, the manifest records `"git_sha": "unknown"`. Ship with it set
for anything you plan to cite.

## What's **not** in the manifest (yet)

- Per-frame detection bboxes (only max count is recorded).
- Output mp4 codec / bitrate / resolution (stable per run but redundant with the file).
- License strings for each dependency (see `LICENSE` and the linked HF/GitHub repos).
- Cross-frame track identity — `track_ids` is derived from per-frame
  detection count, not a real ReID association. Explicit gap; a future
  schema_version=2 will carry stable IDs when we ship real tracking.

## Consumer hints

- To test-pin a paper result: include `input.sha256` and `software.git_sha`
  in your methods section. If someone clones the repo at that SHA and
  processes a video with that SHA256, they get a byte-identical output
  mp4 (modulo OpenCV build differences).
- To audit license posture: parse `models.segmenter`. All v1 values are
  Apache 2.0. Future Sapiens integration would surface a non-Apache
  model id; that's your trigger to require explicit opt-in.
