"""Experimental source-hair transfer for face swapping.

The inswapper model only reconstructs the face oval — it carries no hair
information at all, so a swap keeps the *target's* hair.  This module adds an
approximate hair transfer on top: it segments hair out of the source image
once, then warps that cutout onto each frame using the similarity transform
implied by the target's 5-point keypoints.

Known limits (this is a 2D approximation, not a 3D head model):

* Works best near-frontal.  Past roughly 25-30 degrees of yaw the pasted hair
  reads as a flat sticker because a 2D warp cannot reveal the far side of the
  head or re-shade the silhouette.
* No occlusion handling — a hand or shoulder passing in front of the head is
  drawn over by the hair layer.
* The source's own head pose is baked into the cutout.  A near-frontal,
  evenly-lit source with hair fully inside the frame gives by far the best
  result.
"""

from __future__ import annotations

import os
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

import modules.globals
from modules.typing import Face, Frame

NAME = "DLC.HAIR-TRANSFER"

MODEL_FILENAME = "hair_segmenter.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "hair_segmenter/float32/latest/hair_segmenter.tflite"
)

# MediaPipe's hair_segmenter emits 2 categories: 0 = background, 1 = hair.
_HAIR_CATEGORY = 1

_SEGMENTER = None
_SEGMENTER_LOCK = threading.Lock()

# Cached source-side extraction, keyed by source path + mtime so an edited or
# swapped-out source image re-segments instead of serving a stale cutout.
_SOURCE_CACHE_KEY: Optional[tuple] = None
_SOURCE_HAIR_RGB: Optional[np.ndarray] = None
_SOURCE_HAIR_ALPHA: Optional[np.ndarray] = None
_SOURCE_KPS: Optional[np.ndarray] = None


def _models_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models")


def get_segmenter():
    """Lazily build the MediaPipe hair segmenter (thread-safe)."""
    global _SEGMENTER
    if _SEGMENTER is not None:
        return _SEGMENTER

    with _SEGMENTER_LOCK:
        if _SEGMENTER is not None:
            return _SEGMENTER

        model_path = os.path.join(_models_dir(), MODEL_FILENAME)
        if not os.path.exists(model_path):
            print(f"[{NAME}] {MODEL_FILENAME} not found in models/. "
                  f"Download it from {MODEL_URL}")
            return None

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError:
            print(f"[{NAME}] mediapipe is not installed — hair transfer disabled.")
            return None

        options = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
        )
        _SEGMENTER = vision.ImageSegmenter.create_from_options(options)
    return _SEGMENTER


def _segment_hair(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Return a uint8 hair mask (0/255) for a BGR image, or None on failure."""
    segmenter = get_segmenter()
    if segmenter is None:
        return None

    try:
        import mediapipe as mp
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = segmenter.segment(mp_image)
        if result.category_mask is None:
            return None
        category = result.category_mask.numpy_view()
        if category.ndim == 3:
            category = category[:, :, 0]
        mask = (category == _HAIR_CATEGORY).astype(np.uint8) * 255
        if mask.shape[:2] != image_bgr.shape[:2]:
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        return mask
    except Exception as exc:  # never break the swap pipeline
        print(f"[{NAME}] segmentation failed: {exc}")
        return None


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Drop specks, close pinholes, and feather the border into an alpha ramp."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep only the largest connected component — isolates the head's hair
    # from stray matches on clothing or background texture.
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    return cv2.GaussianBlur(mask, (0, 0), 4)


def prepare_source(source_path: str, source_face: Face) -> bool:
    """Segment and cache the source image's hair. Returns True when ready.

    Runs once per source image — the expensive segmentation never touches the
    per-frame path.
    """
    global _SOURCE_CACHE_KEY, _SOURCE_HAIR_RGB, _SOURCE_HAIR_ALPHA, _SOURCE_KPS

    if not source_path or source_face is None or source_face.kps is None:
        return False

    try:
        mtime = os.path.getmtime(source_path)
    except OSError:
        return False

    key = (source_path, mtime)
    if key == _SOURCE_CACHE_KEY and _SOURCE_HAIR_ALPHA is not None:
        return True

    from modules import imread_unicode
    source_img = imread_unicode(source_path)
    if source_img is None:
        return False

    mask = _segment_hair(source_img)
    if mask is None or int((mask > 127).sum()) < 500:
        # Too little hair found to be worth compositing (bald, hat, tight crop).
        _SOURCE_CACHE_KEY = key
        _SOURCE_HAIR_RGB = None
        _SOURCE_HAIR_ALPHA = None
        _SOURCE_KPS = None
        return False

    _SOURCE_HAIR_ALPHA = _clean_mask(mask)
    _SOURCE_HAIR_RGB = source_img
    _SOURCE_KPS = source_face.kps.astype(np.float32)
    _SOURCE_CACHE_KEY = key
    return True


def _similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
    """Least-squares similarity (rotate+scale+translate) mapping src → dst."""
    if src_pts is None or dst_pts is None:
        return None
    if len(src_pts) < 2 or len(dst_pts) < 2:
        return None
    matrix, _inliers = cv2.estimateAffinePartial2D(
        src_pts.reshape(-1, 1, 2).astype(np.float32),
        dst_pts.reshape(-1, 1, 2).astype(np.float32),
        method=cv2.LMEDS,
    )
    return matrix


# Nose-offset thresholds, measured along the eye-to-eye axis and normalised by
# eye span. Calibrated against real detections: near-frontal faces land around
# 0.04-0.20, clearly turned ones at 0.40+.
_YAW_FULL = 0.22   # at or below this the hair layer is drawn at full strength
_YAW_ZERO = 0.55   # at or above this it is faded out entirely


def _yaw_confidence(kps: np.ndarray) -> float:
    """Frontality score in [0, 1] from 5-point keypoint asymmetry.

    kps order is [right_eye, left_eye, nose, right_mouth, left_mouth].  On a
    frontal face the nose sits midway between the eyes; as the head turns it
    slides toward the near eye.  The offset is projected onto the eye-to-eye
    axis so in-plane head *tilt* (which the warp handles fine) is not mistaken
    for yaw.  1.0 means frontal, 0.0 means turned too far to composite.
    """
    try:
        right_eye, left_eye, nose = (np.asarray(kps[i], dtype=np.float32)
                                     for i in (0, 1, 2))
    except (IndexError, TypeError, ValueError):
        return 0.0

    eye_vec = left_eye - right_eye
    eye_span = float(np.linalg.norm(eye_vec))
    if eye_span < 1e-3:
        return 0.0

    eye_mid = (left_eye + right_eye) * 0.5
    axis = eye_vec / eye_span
    offset = abs(float(np.dot(nose - eye_mid, axis))) / eye_span

    if offset <= _YAW_FULL:
        return 1.0
    return float(np.clip((_YAW_ZERO - offset) / (_YAW_ZERO - _YAW_FULL), 0.0, 1.0))


def apply_hair(frame: Frame, target_face: Face) -> Frame:
    """Composite the cached source hair onto ``frame`` for ``target_face``.

    Returns the frame unchanged when the source has no usable hair, the
    transform cannot be solved, or the head is turned too far for a 2D warp to
    look right.
    """
    if _SOURCE_HAIR_ALPHA is None or _SOURCE_HAIR_RGB is None or _SOURCE_KPS is None:
        return frame
    if target_face is None or target_face.kps is None:
        return frame

    target_kps = target_face.kps.astype(np.float32)

    # Fade the layer out as the head turns instead of popping it off, so the
    # transition is not a visible flicker mid-motion.
    confidence = _yaw_confidence(target_kps)
    if confidence <= 0.05:
        return frame

    matrix = _similarity_transform(_SOURCE_KPS, target_kps)
    if matrix is None:
        return frame

    h, w = frame.shape[:2]
    warped_rgb = cv2.warpAffine(_SOURCE_HAIR_RGB, matrix, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped_alpha = cv2.warpAffine(_SOURCE_HAIR_ALPHA, matrix, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    strength = float(getattr(modules.globals, "hair_transfer_strength", 100.0)) / 100.0
    alpha = (warped_alpha.astype(np.float32) / 255.0) * strength * confidence
    if float(alpha.max()) <= 0.01:
        return frame

    alpha = alpha[:, :, None]
    blended = warped_rgb.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)
    return blended.astype(np.uint8)


def reset_source_cache() -> None:
    """Force the next prepare_source() call to re-segment."""
    global _SOURCE_CACHE_KEY, _SOURCE_HAIR_RGB, _SOURCE_HAIR_ALPHA, _SOURCE_KPS
    _SOURCE_CACHE_KEY = None
    _SOURCE_HAIR_RGB = None
    _SOURCE_HAIR_ALPHA = None
    _SOURCE_KPS = None
