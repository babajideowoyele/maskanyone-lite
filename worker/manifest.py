"""Sidecar manifest writer for every masked output.

Captures enough provenance (input hash, model IDs + versions, git SHA, run
parameters) for a researcher to cite the output and reproduce it later.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Optional


SCHEMA_VERSION = 1


def _sha256(path: str, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _model_info(mode: str) -> dict:
    if mode == "quick":
        import mediapipe
        return {
            "segmenter": "mediapipe.solutions.selfie_segmentation",
            "model_selection": 1,
            "mediapipe_version": mediapipe.__version__,
        }
    if mode == "precision":
        import torch
        import transformers
        return {
            "segmenter": "yonigozlan/EdgeTAM-hf",
            "framework": "transformers",
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }
    return {"segmenter": "unknown"}


def write(
    manifest_path: str,
    *,
    input_path: str,
    output_path: str,
    mode: str,
    strategy: str,
    prompt_xy: Optional[tuple[int, int]],
    downsample: float,
    frames: int,
    duration_s: float,
    original_filename: Optional[str] = None,
    detection_max_per_frame: Optional[int] = None,
    frames_with_no_detection: Optional[int] = None,
) -> dict:
    import numpy
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {
            "original_filename": original_filename,
            "basename": os.path.basename(input_path),
            "sha256": _sha256(input_path),
        },
        "output": {
            "basename": os.path.basename(output_path),
            "frames": frames,
            "mode": mode,
            "strategy": strategy,
            "prompt_xy": list(prompt_xy) if prompt_xy else None,
            "downsample": downsample,
            "track_ids": (
                list(range(1, detection_max_per_frame + 1))
                if detection_max_per_frame is not None and detection_max_per_frame > 0
                else [1]
            ),
            "detection_max_per_frame": detection_max_per_frame,
            "frames_with_no_detection": frames_with_no_detection,
        },
        "runtime": {
            "duration_seconds": round(duration_s, 2),
            "python": sys.version.split()[0],
            "numpy_version": numpy.__version__,
        },
        "software": {
            "repo": "maskanyone-lite",
            "git_sha": os.environ.get("MASKANYONE_LITE_GIT_SHA", "unknown"),
        },
        "models": _model_info(mode),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
