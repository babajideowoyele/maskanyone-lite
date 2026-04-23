"""MediaPipe Selfie-segmentation masker.

Quick mode of maskanyone-lite: blur the person silhouette, keep the
background. No prompts, no tracking — just per-frame binary segmentation.

Usage:
    python masker.py <input.mp4> <output.mp4> [strategy]
    strategy: blur (default) | solid | pixelate
"""
from __future__ import annotations

import sys

import cv2
import mediapipe as mp
import numpy as np


def mask_video(input_path: str, output_path: str, strategy: str = "blur") -> int:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {input_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mask = selfie.process(rgb).segmentation_mask > 0.5
            mask3 = np.repeat(mask[..., None], 3, axis=2)
            if strategy == "blur":
                replacement = cv2.GaussianBlur(frame, (0, 0), sigmaX=25)
            elif strategy == "solid":
                replacement = np.zeros_like(frame)
            elif strategy == "pixelate":
                small = cv2.resize(frame, (max(1, w // 20), max(1, h // 20)), interpolation=cv2.INTER_LINEAR)
                replacement = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                raise ValueError(f"unknown strategy: {strategy}")
            out.write(np.where(mask3, replacement, frame))
            n += 1
    finally:
        cap.release()
        out.release()
        selfie.close()
    return n


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python masker.py <input.mp4> <output.mp4> [blur|solid|pixelate]", file=sys.stderr)
        sys.exit(2)
    strat = sys.argv[3] if len(sys.argv) > 3 else "blur"
    n = mask_video(sys.argv[1], sys.argv[2], strat)
    print(f"wrote {sys.argv[2]}: {n} frames, strategy={strat}")
