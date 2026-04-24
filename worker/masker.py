"""Video masker — quick (MediaPipe Selfie) and precision (EdgeTAM) modes.

Both modes produce a per-frame binary mask of the subject; `strategy`
(blur | solid | pixelate) controls how the mask region is rewritten.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

import manifest as _manifest


def _replace(frame: np.ndarray, mask3: np.ndarray, strategy: str, w: int, h: int) -> np.ndarray:
    if strategy == "blur":
        replacement = cv2.GaussianBlur(frame, (0, 0), sigmaX=25)
    elif strategy == "solid":
        replacement = np.zeros_like(frame)
    elif strategy == "pixelate":
        small = cv2.resize(frame, (max(1, w // 20), max(1, h // 20)), interpolation=cv2.INTER_LINEAR)
        replacement = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return np.where(mask3, replacement, frame)


def _mask_quick(frame_bgr: np.ndarray, segmenter) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return segmenter.process(rgb).segmentation_mask > 0.5


def _skeleton_frame(frame_bgr: np.ndarray, holistic_ctx) -> np.ndarray:
    """Black out the person silhouette and overlay pose/face/hand landmarks.

    Uses MediaPipe Holistic (single pass for segmentation + landmarks). The
    mask color is solid black; the landmarks are drawn with MediaPipe's
    default styles on top of the blacked-out region. Background is untouched.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_h = mp.solutions.holistic

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = holistic_ctx.process(rgb)
    out = frame_bgr.copy()

    if res.segmentation_mask is not None:
        mask = res.segmentation_mask > 0.5
        out[mask] = 0

    if res.pose_landmarks:
        mp_drawing.draw_landmarks(
            out,
            res.pose_landmarks,
            mp_h.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
        )
    if res.face_landmarks:
        mp_drawing.draw_landmarks(
            out,
            res.face_landmarks,
            mp_h.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
        )
    if res.left_hand_landmarks:
        mp_drawing.draw_landmarks(out, res.left_hand_landmarks, mp_h.HAND_CONNECTIONS)
    if res.right_hand_landmarks:
        mp_drawing.draw_landmarks(out, res.right_hand_landmarks, mp_h.HAND_CONNECTIONS)

    return out


# EdgeTAM is loaded lazily (heavy import + model download) and cached.
_edgetam: Optional[tuple] = None


def _get_edgetam():
    global _edgetam
    if _edgetam is None:
        import torch
        from PIL import Image
        from transformers import EdgeTamModel, Sam2Processor

        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
        torch.set_grad_enabled(False)
        processor = Sam2Processor.from_pretrained("yonigozlan/EdgeTAM-hf")
        model = EdgeTamModel.from_pretrained("yonigozlan/EdgeTAM-hf").to("cpu").eval()
        _edgetam = (model, processor, torch, Image)
    return _edgetam


def _mask_precision(frame_bgr: np.ndarray, prompt_xy: tuple[int, int]) -> np.ndarray:
    model, processor, torch, Image = _get_edgetam()
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(
        images=pil,
        input_points=[[[[prompt_xy[0], prompt_xy[1]]]]],
        input_labels=[[[1]]],
        return_tensors="pt",
    )
    outputs = model(**inputs)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    mask = masks[0, 0] if masks.ndim == 4 else masks[0]
    return mask.numpy().astype(bool)


def mask_video(
    input_path: str,
    output_path: str,
    strategy: str = "blur",
    mode: str = "quick",
    prompt_xy: Optional[tuple[int, int]] = None,
    downsample: float = 1.0,
    original_filename: Optional[str] = None,
) -> int:
    if not 0.1 <= downsample <= 1.0:
        raise ValueError(f"downsample must be in [0.1, 1.0], got {downsample}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {input_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Segmentation resolution; output stays at native resolution.
    sw, sh = max(1, int(w * downsample)), max(1, int(h * downsample))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Skeleton strategy bypasses the mask-then-replace flow: it runs
    # MediaPipe Holistic (one model for mask + pose + face + hands) per frame
    # and writes the frame with the silhouette blacked and landmarks drawn.
    if strategy == "skeleton":
        holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            enable_segmentation=True,
            refine_face_landmarks=False,
        )
        start = time.perf_counter()
        n = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(_skeleton_frame(frame, holistic))
                n += 1
        finally:
            cap.release()
            out.release()
            holistic.close()
        _manifest.write(
            output_path + ".manifest.json",
            input_path=input_path,
            output_path=output_path,
            mode=mode,
            strategy=strategy,
            prompt_xy=None,
            downsample=downsample,
            frames=n,
            duration_s=time.perf_counter() - start,
            original_filename=original_filename,
        )
        return n

    if mode == "quick":
        segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        mask_fn = lambda f: _mask_quick(f, segmenter)
        cleanup = lambda: segmenter.close()
    elif mode == "precision":
        # Scale prompt to the downsampled frame if we're resizing.
        raw = prompt_xy or (w // 2, h // 2)
        pxy_small = (int(raw[0] * downsample), int(raw[1] * downsample))
        mask_fn = lambda f: _mask_precision(f, pxy_small)
        cleanup = lambda: None
    else:
        cap.release(); out.release()
        raise ValueError(f"unknown mode: {mode}")

    start = time.perf_counter()
    n = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if downsample < 1.0:
                frame_small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
                mask_small = mask_fn(frame_small)
                mask = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                mask = mask_fn(frame)
            mask3 = np.repeat(mask[..., None], 3, axis=2)
            out.write(_replace(frame, mask3, strategy, w, h))
            n += 1
    finally:
        cap.release()
        out.release()
        cleanup()
    duration_s = time.perf_counter() - start
    _manifest.write(
        output_path + ".manifest.json",
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        strategy=strategy,
        prompt_xy=prompt_xy if mode == "precision" else None,
        downsample=downsample,
        frames=n,
        duration_s=duration_s,
        original_filename=original_filename,
    )
    return n


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python masker.py <in.mp4> <out.mp4> [strategy] [mode]", file=sys.stderr)
        sys.exit(2)
    strategy = sys.argv[3] if len(sys.argv) > 3 else "blur"
    mode = sys.argv[4] if len(sys.argv) > 4 else "quick"
    n = mask_video(sys.argv[1], sys.argv[2], strategy, mode)
    print(f"wrote {sys.argv[2]}: {n} frames, strategy={strategy}, mode={mode}")
