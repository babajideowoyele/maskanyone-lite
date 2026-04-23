"""Benchmark MobileSAM on CPU.

Per-frame inference with a single center-point prompt re-applied each frame.
Matches bench_edgetam.py's protocol so results are directly comparable.

Usage:
    python bench_mobilesam.py <video_path> [num_frames]
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from bench_common import (
    FrameTiming,
    current_rss_mb,
    pick_center_point,
    print_result,
    sample_frames,
    stopwatch,
    summarize,
)


CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mobile_sam.pt")


def run(video_path: str, num_frames: int) -> dict:
    from mobile_sam import sam_model_registry, SamPredictor

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    torch.set_grad_enabled(False)

    print(f"loading MobileSAM from {CHECKPOINT} ...")
    t_load = time.perf_counter()
    sam = sam_model_registry["vit_t"](checkpoint=CHECKPOINT)
    sam.to("cpu").eval()
    predictor = SamPredictor(sam)
    load_s = time.perf_counter() - t_load
    print(f"model loaded in {load_s:.1f}s; torch threads={torch.get_num_threads()}")

    frames_bgr = sample_frames(video_path, num_frames)
    h, w = frames_bgr[0].shape[:2]
    px, py = pick_center_point(frames_bgr[0])
    print(f"video: {w}x{h}, {len(frames_bgr)} frames; prompt=({px},{py})")

    start_rss = current_rss_mb()
    print(f"start RSS: {start_rss:.0f} MB")

    per_frame: list[FrameTiming] = []
    wall_start = time.perf_counter()

    point_coords = np.array([[px, py]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    for idx, frame_bgr in enumerate(frames_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        with stopwatch() as sw:
            predictor.set_image(frame_rgb)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
            mask = masks[0].astype(np.uint8)
        rss_mb = current_rss_mb()
        area = int(mask.sum())
        per_frame.append(
            FrameTiming(frame_idx=idx, ms=sw.ms, mask_area_px=area, rss_mb=rss_mb)
        )
        print(f"frame {idx:3d}: {sw.ms:6.0f} ms, area={area:>8d}, rss={rss_mb:.0f} MB, score={scores[0]:.3f}")

    total_wall_s = time.perf_counter() - wall_start

    result = summarize(
        segmenter="MobileSAM (vit_t)",
        video=video_path,
        per_frame=per_frame,
        start_rss_mb=start_rss,
        total_wall_s=total_wall_s,
    )
    print_result(result)

    return {
        "segmenter": "MobileSAM",
        "checkpoint": "mobile_sam.pt",
        "backbone": "vit_t",
        "mode": "per-frame (no tracking)",
        "torch_version": torch.__version__,
        "dtype": "float32",
        "num_threads": torch.get_num_threads(),
        "video": os.path.basename(result.video),
        "video_size": f"{w}x{h}",
        "num_frames": result.num_frames,
        "load_s": round(load_s, 2),
        "first_frame_ms": round(result.first_frame_ms, 1),
        "mean_ms_excl_first": round(result.mean_ms_excl_first, 1),
        "median_ms_excl_first": round(result.median_ms_excl_first, 1),
        "p95_ms_excl_first": round(result.p95_ms_excl_first, 1),
        "total_wall_s": round(result.total_wall_s, 2),
        "start_rss_mb": round(start_rss, 0),
        "peak_rss_mb": round(result.peak_rss_mb, 0),
        "rss_growth_mb": round(result.rss_growth_mb, 0),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python bench_mobilesam.py <video_path> [num_frames]", file=sys.stderr)
        sys.exit(2)
    vp = sys.argv[1]
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    result = run(vp, nf)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_mobilesam.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nresult written to {out_path}")
