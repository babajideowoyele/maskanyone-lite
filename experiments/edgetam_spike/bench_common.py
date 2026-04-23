"""Shared benchmarking utilities for segmenter CPU spikes.

Both the EdgeTAM and MobileSAM benchmarks use the same sampling strategy,
timing logic, and measurement reporting, so results are directly comparable.
"""
from __future__ import annotations

import dataclasses
import gc
import os
import statistics
import time

import cv2
import numpy as np
import psutil


@dataclasses.dataclass
class FrameTiming:
    frame_idx: int
    ms: float
    mask_area_px: int
    rss_mb: float


@dataclasses.dataclass
class BenchResult:
    segmenter: str
    video: str
    num_frames: int
    first_frame_ms: float
    mean_ms_excl_first: float
    median_ms_excl_first: float
    p95_ms_excl_first: float
    total_wall_s: float
    peak_rss_mb: float
    rss_growth_mb: float
    per_frame: list[FrameTiming]


def sample_frames(video_path: str, num_frames: int) -> list[np.ndarray]:
    """Return the first `num_frames` BGR frames from the video.

    We use contiguous leading frames (not evenly spaced) because tracking
    quality is what we want to measure — a tracker fed skip-frames doesn't
    reflect real use.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < num_frames:
        raise RuntimeError(
            f"video has only {len(frames)} frames, requested {num_frames}"
        )
    return frames


def pick_center_point(frame: np.ndarray) -> tuple[int, int]:
    """Heuristic prompt location: center of the frame.

    For our test clips the subject dominates the center, so this is a
    reasonable stand-in for a real user click. Each benchmark can override.
    """
    h, w = frame.shape[:2]
    return (w // 2, h // 2)


def summarize(
    segmenter: str,
    video: str,
    per_frame: list[FrameTiming],
    start_rss_mb: float,
    total_wall_s: float,
) -> BenchResult:
    all_ms = [t.ms for t in per_frame]
    tail_ms = all_ms[1:] if len(all_ms) > 1 else all_ms
    peak_rss_mb = max(t.rss_mb for t in per_frame)
    return BenchResult(
        segmenter=segmenter,
        video=video,
        num_frames=len(per_frame),
        first_frame_ms=all_ms[0] if all_ms else 0.0,
        mean_ms_excl_first=statistics.mean(tail_ms) if tail_ms else 0.0,
        median_ms_excl_first=statistics.median(tail_ms) if tail_ms else 0.0,
        p95_ms_excl_first=(
            statistics.quantiles(tail_ms, n=20)[18]
            if len(tail_ms) >= 20
            else max(tail_ms) if tail_ms else 0.0
        ),
        total_wall_s=total_wall_s,
        peak_rss_mb=peak_rss_mb,
        rss_growth_mb=peak_rss_mb - start_rss_mb,
        per_frame=per_frame,
    )


def print_result(result: BenchResult) -> None:
    print()
    print(f"=== {result.segmenter} on {os.path.basename(result.video)} ===")
    print(f"frames                : {result.num_frames}")
    print(f"first-frame ms        : {result.first_frame_ms:.1f}")
    print(f"mean ms (ex. first)   : {result.mean_ms_excl_first:.1f}")
    print(f"median ms (ex. first) : {result.median_ms_excl_first:.1f}")
    print(f"p95 ms (ex. first)    : {result.p95_ms_excl_first:.1f}")
    print(f"total wall time       : {result.total_wall_s:.2f} s")
    print(f"peak RSS              : {result.peak_rss_mb:.0f} MB")
    print(f"RSS growth over run   : {result.rss_growth_mb:+.0f} MB")


def current_rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def stopwatch():
    """Context-managed ms timer. Usage: with stopwatch() as t: ...; t.ms"""

    class _Stopwatch:
        ms: float = 0.0
        _start: float = 0.0

        def __enter__(self):
            gc.collect()
            self._start = time.perf_counter()
            return self

        def __exit__(self, *_):
            self.ms = (time.perf_counter() - self._start) * 1000.0

    return _Stopwatch()
