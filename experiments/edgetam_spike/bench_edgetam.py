"""Benchmark EdgeTAM (image model, yonigozlan/EdgeTAM-hf) on CPU.

Per-frame inference with a single center-point prompt re-applied each frame.
This mirrors how MobileSAM would be used without a tracker and gives a fair
apples-to-apples CPU cost comparison.

NOTE: The *video* variant of EdgeTAM (yonigozlan/edgetam-video-1), which has
built-in cross-frame tracking via Sam2VideoProcessor, is currently returning
HTTP 401 (private or ungated). This script deliberately uses the image
variant so the measurement is reproducible today. Retry the video variant
when the repo becomes public.

Usage:
    python bench_edgetam.py <video_path> [num_frames]
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


MODEL_ID = "yonigozlan/EdgeTAM-hf"


def run(video_path: str, num_frames: int) -> dict:
    from transformers import EdgeTamModel, Sam2Processor

    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    torch.set_grad_enabled(False)

    print(f"loading {MODEL_ID} ...")
    t_load = time.perf_counter()
    processor = Sam2Processor.from_pretrained(MODEL_ID)
    model = EdgeTamModel.from_pretrained(MODEL_ID).to("cpu").eval()
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

    for idx, frame_bgr in enumerate(frames_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image

        pil = Image.fromarray(frame_rgb)
        with stopwatch() as sw:
            inputs = processor(
                images=pil,
                input_points=[[[[px, py]]]],
                input_labels=[[[1]]],
                return_tensors="pt",
            )
            outputs = model(**inputs)
            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"]
            )[0]
            # masks shape: (1, num_masks, H, W) — take the first mask.
            mask = masks[0, 0].numpy().astype(np.uint8) if masks.ndim == 4 else masks[0].numpy().astype(np.uint8)
        rss_mb = current_rss_mb()
        area = int(mask.sum())
        per_frame.append(
            FrameTiming(frame_idx=idx, ms=sw.ms, mask_area_px=area, rss_mb=rss_mb)
        )
        print(f"frame {idx:3d}: {sw.ms:6.0f} ms, area={area:>8d}, rss={rss_mb:.0f} MB")

    total_wall_s = time.perf_counter() - wall_start

    result = summarize(
        segmenter=f"EdgeTAM-image ({MODEL_ID})",
        video=video_path,
        per_frame=per_frame,
        start_rss_mb=start_rss,
        total_wall_s=total_wall_s,
    )
    print_result(result)

    return {
        "segmenter": "EdgeTAM-image",
        "model_id": MODEL_ID,
        "mode": "per-frame (no tracking)",
        "transformers_version": __import__("transformers").__version__,
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
        print("usage: python bench_edgetam.py <video_path> [num_frames]", file=sys.stderr)
        sys.exit(2)
    vp = sys.argv[1]
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    result = run(vp, nf)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_edgetam.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nresult written to {out_path}")
