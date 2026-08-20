"""Eye detail preservation for face swapping.

inswapper rebuilds the whole face from a 128x128 tensor, and the eyes pay for
it twice: the iris is reconstructed from the source identity, so the *target's*
gaze and eye colour are gone, and whatever is left has been upscaled from a
handful of pixels, so it smears the moment the head moves.

This module puts the real eyes back.  It cuts the eye regions out of the
untouched frame and composites them over the finished swap, which restores all
three things at once — the actual gaze direction, the actual eye colour, and
full sensor resolution instead of an upscale.

Optionally the iris can then be re-tinted toward the source identity's eye
colour, so a blue-eyed source stays blue-eyed while the movement and sharpness
still come from the live frame.
"""

from __future__ import annotations

import os
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np

import modules.globals
from modules.typing import Face, Frame

# insightface 2d106 landmark groups.  Same indices the repo's other eye code
# uses; verified against live detections rather than assumed.
RIGHT_EYE = slice(33, 42)
LEFT_EYE = slice(87, 96)

_SOURCE_IRIS_LOCK = threading.Lock()
_SOURCE_IRIS_KEY: Optional[tuple] = None
_SOURCE_IRIS_LAB: Optional[np.ndarray] = None


def _eye_polygons(face: Face) -> Optional[List[np.ndarray]]:
    """Return [right_eye, left_eye] point arrays, or None when unusable."""
    landmarks = getattr(face, "landmark_2d_106", None)
    if landmarks is None or not isinstance(landmarks, np.ndarray):
        return None
    if landmarks.shape[0] < 106:
        return None

    polygons = [
        landmarks[RIGHT_EYE].astype(np.float32),
        landmarks[LEFT_EYE].astype(np.float32),
    ]
    if not all(np.all(np.isfinite(p)) for p in polygons):
        return None
    return polygons


def _expand(points: np.ndarray, scale: float) -> np.ndarray:
    """Grow a polygon about its own centroid, keeping it simple."""
    center = points.mean(axis=0)
    return center + (points - center) * scale


def _feather_kernel(polygons: List[np.ndarray]) -> int:
    """Odd blur kernel sized from a single eye, not from both eyes' extent.

    Deriving it from the combined bounding box makes the kernel roughly as wide
    as the gap between the eyes, which blurs the alpha so far that the centre
    never reaches full opacity and only a fraction of the real eye comes
    through.  A single eye's height is the dimension that matters, and staying
    under half of it leaves a solid core.
    """
    heights = [float(p[:, 1].max() - p[:, 1].min()) for p in polygons]
    eye_h = max(1.0, sum(heights) / len(heights))
    k = int(eye_h * 0.35)
    return max(3, min(k | 1, 31))


def create_eyes_mask(
    face: Face, frame: Frame
) -> Tuple[np.ndarray, Optional[np.ndarray], tuple, Optional[List[np.ndarray]]]:
    """Build the eye-region mask and cut the real eyes out of ``frame``.

    ``frame`` must be the pre-swap frame — that is the whole point.
    Returns (mask, cutout, box, polygons); cutout is None when unavailable.
    """
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    empty: Tuple[np.ndarray, Optional[np.ndarray], tuple, Optional[List[np.ndarray]]] = (
        mask, None, (0, 0, 0, 0), None
    )

    if face is None or frame is None or frame.size == 0:
        return empty

    polygons = _eye_polygons(face)
    if polygons is None:
        return empty

    try:
        size = float(getattr(modules.globals, "eyes_mask_size", 0.0))
        s = max(0.0, min(1.0, size / 100.0))
        # 1.15x at the low end already clears the lash line; 2.3x reaches the
        # socket without spilling onto the brow.
        scale = 1.15 + s * 1.15
        expanded = [_expand(p, scale) for p in polygons]

        all_points = np.vstack(expanded)
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)

        pad_x = (max_x - min_x) * 0.12
        pad_y = (max_y - min_y) * 0.22  # more vertical room: lids move
        frame_h, frame_w = frame.shape[:2]
        min_x = int(max(0, min_x - pad_x))
        min_y = int(max(0, min_y - pad_y))
        max_x = int(min(frame_w, max_x + pad_x))
        max_y = int(min(frame_h, max_y + pad_y))

        if max_x <= min_x or max_y <= min_y:
            return empty

        roi_mask = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
        for poly in expanded:
            shifted = (poly - [min_x, min_y]).astype(np.int32)
            cv2.fillConvexPoly(roi_mask, cv2.convexHull(shifted), 255)

        k = _feather_kernel(expanded)
        roi_mask = cv2.GaussianBlur(roi_mask, (k, k), 0)

        mask[min_y:max_y, min_x:max_x] = roi_mask
        cutout = frame[min_y:max_y, min_x:max_x].copy()
        return mask, cutout, (min_x, min_y, max_x, max_y), expanded

    except Exception as exc:  # never break the swap
        print(f"[eye_detail] mask build failed: {exc}")
        return empty


def _iris_pixels(bgr: np.ndarray) -> np.ndarray:
    """Boolean mask of iris-ish pixels: the darker half of the eye patch.

    The sclera is the bright part of any eye patch regardless of skin tone or
    exposure, so a within-patch luminance split isolates iris plus pupil
    without needing a threshold tuned per person.
    """
    if bgr.size == 0:
        return np.zeros(bgr.shape[:2], dtype=bool)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cutoff = np.percentile(gray, 42)
    return gray <= cutoff


def prepare_source_iris(source_path: str, source_face: Face) -> bool:
    """Cache the source identity's mean iris colour (LAB). Once per source."""
    global _SOURCE_IRIS_KEY, _SOURCE_IRIS_LAB

    if not source_path or source_face is None:
        return False
    try:
        key = (source_path, os.path.getmtime(source_path))
    except OSError:
        return False

    with _SOURCE_IRIS_LOCK:
        if key == _SOURCE_IRIS_KEY and _SOURCE_IRIS_LAB is not None:
            return True

        from modules import imread_unicode
        image = imread_unicode(source_path)
        if image is None:
            return False

        polygons = _eye_polygons(source_face)
        if polygons is None:
            _SOURCE_IRIS_KEY, _SOURCE_IRIS_LAB = key, None
            return False

        pts = np.vstack(polygons)
        min_x, min_y = np.maximum(np.min(pts, axis=0).astype(int), 0)
        max_x, max_y = np.min(
            [np.max(pts, axis=0).astype(int), [image.shape[1], image.shape[0]]], axis=0
        )
        if max_x <= min_x or max_y <= min_y:
            _SOURCE_IRIS_KEY, _SOURCE_IRIS_LAB = key, None
            return False

        patch = image[min_y:max_y, min_x:max_x]
        iris = _iris_pixels(patch)
        if int(iris.sum()) < 12:
            _SOURCE_IRIS_KEY, _SOURCE_IRIS_LAB = key, None
            return False

        lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
        _SOURCE_IRIS_LAB = lab[iris].mean(axis=0).astype(np.float32)
        _SOURCE_IRIS_KEY = key
        return True


def _retint_iris(cutout: np.ndarray, strength: float) -> np.ndarray:
    """Shift the cutout's iris chroma toward the cached source iris colour.

    Only a and b (chroma) move; L is left alone so the eye keeps its own
    highlights, shadow and wetness — swapping luminance is what makes tinted
    eyes look like flat contact lenses.
    """
    if _SOURCE_IRIS_LAB is None or strength <= 0.0 or cutout.size == 0:
        return cutout

    iris = _iris_pixels(cutout)
    if int(iris.sum()) < 12:
        return cutout

    lab = cv2.cvtColor(cutout, cv2.COLOR_BGR2LAB).astype(np.float32)
    current = lab[iris].mean(axis=0)
    delta_a = (_SOURCE_IRIS_LAB[1] - current[1]) * strength
    delta_b = (_SOURCE_IRIS_LAB[2] - current[2]) * strength

    lab[iris, 1] = np.clip(lab[iris, 1] + delta_a, 0, 255)
    lab[iris, 2] = np.clip(lab[iris, 2] + delta_b, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_eyes_area(
    frame: np.ndarray,
    cutout: np.ndarray,
    box: tuple,
    face_mask: np.ndarray,
    polygons: List[np.ndarray],
) -> np.ndarray:
    """Composite the real eyes from ``cutout`` over the swapped ``frame``."""
    if frame is None or cutout is None or not polygons:
        return frame
    if cutout.size == 0 or face_mask is None or face_mask.size == 0:
        return frame

    try:
        min_x, min_y, max_x, max_y = map(int, box)
        frame_h, frame_w = frame.shape[:2]
        min_x, max_x = max(0, min_x), min(frame_w, max_x)
        min_y, max_y = max(0, min_y), min(frame_h, max_y)
        box_w, box_h = max_x - min_x, max_y - min_y
        if box_w <= 0 or box_h <= 0:
            return frame

        roi = frame[min_y:max_y, min_x:max_x]
        if roi.size == 0:
            return frame

        patch = cutout
        if patch.shape[:2] != roi.shape[:2]:
            patch = cv2.resize(patch, (box_w, box_h), interpolation=cv2.INTER_LINEAR)

        strength = float(getattr(modules.globals, "eye_color_lock", 0.0)) / 100.0
        if strength > 0.0:
            patch = _retint_iris(patch, strength)

        # Build the alpha from the eye polygons, then gate it by the face mask
        # so nothing lands outside the swapped face.
        alpha_roi = np.zeros((box_h, box_w), dtype=np.uint8)
        for poly in polygons:
            shifted = (poly - [min_x, min_y]).astype(np.int32)
            cv2.fillConvexPoly(alpha_roi, cv2.convexHull(shifted), 255)
        k = _feather_kernel(polygons)
        alpha_roi = cv2.GaussianBlur(alpha_roi, (k, k), 0)

        face_roi = face_mask[min_y:max_y, min_x:max_x].astype(np.float32) / 255.0
        alpha = (alpha_roi.astype(np.float32) / 255.0) * face_roi
        alpha = alpha[:, :, None]

        blended = patch.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)
        frame[min_y:max_y, min_x:max_x] = blended.astype(np.uint8)
        return frame

    except Exception as exc:
        print(f"[eye_detail] composite failed: {exc}")
        return frame


def reset_source_cache() -> None:
    global _SOURCE_IRIS_KEY, _SOURCE_IRIS_LAB
    _SOURCE_IRIS_KEY = None
    _SOURCE_IRIS_LAB = None
