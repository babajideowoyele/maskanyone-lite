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


# Person detection via MediaPipe PoseLandmarker (Tasks API). The task file
# is pre-baked into the worker image at /worker_models/pose_landmarker_heavy.task.
# For native dev, we fall back to a user-overridable path.
#
# We do NOT cache the detector globally: detect_for_video() requires strictly
# monotonically increasing timestamps across the detector's lifetime, and a
# long-running worker processing multiple jobs would violate that. A fresh
# detector per mask_video() call costs ~50 ms on startup — negligible vs the
# per-frame inference budget.


def _pose_task_path() -> str:
    candidates = [
        os.environ.get("POSE_LANDMARKER_TASK", ""),
        "/worker_models/pose_landmarker_heavy.task",
        os.path.expanduser("~/.cache/maskanyone-lite/pose_landmarker_heavy.task"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "pose_landmarker_heavy.task not found. Set POSE_LANDMARKER_TASK env var "
        "or download from https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    )


def _make_pose_detector(num_poses: int = 2):
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    base_opts = mp_tasks.BaseOptions(model_asset_path=_pose_task_path())
    opts = vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def _detect_person_bboxes(
    frame_bgr: np.ndarray, timestamp_ms: int, detector, pad_ratio: float = 0.15
) -> list[tuple[int, int, int, int]]:
    """Return list of (x1, y1, x2, y2) pixel bboxes around each detected person."""
    import mediapipe as mp_lib

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp_lib.Image(image_format=mp_lib.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    h, w = frame_bgr.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for pose in result.pose_landmarks:
        xs = [lm.x for lm in pose]
        ys = [lm.y for lm in pose]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0.0, x1 - bw * pad_ratio)
        y1 = max(0.0, y1 - bh * pad_ratio)
        x2 = min(1.0, x2 + bw * pad_ratio)
        y2 = min(1.0, y2 + bh * pad_ratio)
        boxes.append((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
    return boxes


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

    # Precision mode with no explicit prompt → detect persons, crop, segment
    # per crop, composite. This is the path that makes heavier segmenters
    # (EdgeTAM today, Sapiens later) tractable on CPU and fixes the silent
    # multi-person failure.
    if mode == "precision" and prompt_xy is None:
        return _mask_video_detection_driven(
            cap, out, w, h, fps, strategy, downsample,
            input_path, output_path, original_filename,
        )

    if mode == "quick":
        segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        mask_fn = lambda f: _mask_quick(f, segmenter)
        cleanup = lambda: segmenter.close()
    elif mode == "precision":
        raw = prompt_xy  # guaranteed not None by the branch above
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


def _mask_video_detection_driven(
    cap, out_writer, w: int, h: int, fps: float,
    strategy: str, downsample: float,
    input_path: str, output_path: str, original_filename: Optional[str],
) -> int:
    """Precision mode, no manual prompt:
    per frame → detect people → crop each → EdgeTAM on crop → composite."""
    start = time.perf_counter()
    n = 0
    max_detections = 0
    frames_with_none = 0
    frame_detection_counts: list[int] = []
    detector = _make_pose_detector()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            ts_ms = int(n * (1000.0 / fps)) if fps else n
            boxes = _detect_person_bboxes(frame, ts_ms, detector)
            frame_detection_counts.append(len(boxes))

            if not boxes:
                frames_with_none += 1
                # No person detected — pass the frame through unmasked. Honest
                # default: don't silently produce a garbage mask on a bad frame.
                out_writer.write(frame)
                n += 1
                continue

            max_detections = max(max_detections, len(boxes))
            mask_full = np.zeros((h, w), dtype=bool)
            for (x1, y1, x2, y2) in boxes:
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue  # degenerate bbox
                crop = frame[y1:y2, x1:x2]
                if downsample < 1.0:
                    cw, ch = crop.shape[1], crop.shape[0]
                    sw_c = max(1, int(cw * downsample))
                    sh_c = max(1, int(ch * downsample))
                    crop_small = cv2.resize(crop, (sw_c, sh_c), interpolation=cv2.INTER_AREA)
                    mask_crop_small = _mask_precision(crop_small, (sw_c // 2, sh_c // 2))
                    mask_crop = cv2.resize(
                        mask_crop_small.astype(np.uint8), (cw, ch),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    mask_crop = _mask_precision(crop, ((x2 - x1) // 2, (y2 - y1) // 2))
                mask_full[y1:y2, x1:x2] |= mask_crop

            mask3 = np.repeat(mask_full[..., None], 3, axis=2)
            out_writer.write(_replace(frame, mask3, strategy, w, h))
            n += 1
    finally:
        cap.release()
        out_writer.release()
        try:
            detector.close()
        except Exception:
            pass

    duration_s = time.perf_counter() - start
    _manifest.write(
        output_path + ".manifest.json",
        input_path=input_path,
        output_path=output_path,
        mode="precision",
        strategy=strategy,
        prompt_xy=None,
        downsample=downsample,
        frames=n,
        duration_s=duration_s,
        original_filename=original_filename,
        detection_max_per_frame=max_detections,
        frames_with_no_detection=frames_with_none,
    )
    print(
        f"[masker] precision+detect: {n} frames, "
        f"max_persons={max_detections}, empty_frames={frames_with_none}, "
        f"{duration_s:.1f}s"
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
