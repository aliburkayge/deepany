"""Experimental avatar body overlay for the live webcam preview.

The face swapper only ever touches the face oval. This module lets a user
pick a separate "avatar" photo and warps that avatar's body/clothing onto
their own live body, using MediaPipe Pose's 33-point skeleton to drive a
handful of rigid per-segment 2D warps (torso, upper/lower arms, upper/lower
legs). It renders *before* the face swap in the pipeline, so the swapped
face and hair always draw on top of it and the face stays the live user's.

Known limits (this is a rigid 2D approximation, not cloth simulation or a 3D
body model):

* No cloth physics — clothing does not drape, wrinkle, or flow with motion.
  Each segment is a flat texture warped by its two/four defining joints.
* Visible seams at joints (shoulder, elbow, hip, knee), worst at sharp bend
  angles or when the length-ratio clamp caps a stretch.
* Works best near-frontal with moderate poses. Side profiles, self-occlusion
  (an arm crossing the torso), and fast motion degrade it significantly —
  pose landmarks jitter/lag under motion blur and a 2D warp cannot reveal a
  body part the avatar photo never showed.
* The avatar's own pose/proportions are baked into each cutout; a clear,
  evenly-lit, near-frontal full-body avatar photo gives the best result.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import modules.globals
from modules.typing import Frame

NAME = "DLC.AVATAR-BODY"

POSE_MODEL_FILENAME = "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
SEG_MODEL_FILENAME = "selfie_multiclass_256x256.tflite"
SEG_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)

# selfie_multiclass_256x256 category indices (published MediaPipe label map):
# 0 background, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others.
# "Body" for our purposes is body-skin + clothes — everything but hair/face.
_BODY_SKIN_CATEGORY = 2
_CLOTHES_CATEGORY = 4

# MediaPipe Pose 33-point landmark indices we use.
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW, _R_ELBOW = 13, 14
_L_WRIST, _R_WRIST = 15, 16
_L_HIP, _R_HIP = 23, 24
_L_KNEE, _R_KNEE = 25, 26
_L_ANKLE, _R_ANKLE = 27, 28

# Segment name -> ("quad", [4 landmark indices]) or ("strip", [2 landmark
# indices]) defining the bone the strip follows.
SEGMENT_DEFS: Dict[str, Tuple[str, List[int]]] = {
    "torso": ("quad", [_L_SHOULDER, _R_SHOULDER, _R_HIP, _L_HIP]),
    "upper_arm_l": ("strip", [_L_SHOULDER, _L_ELBOW]),
    "lower_arm_l": ("strip", [_L_ELBOW, _L_WRIST]),
    "upper_arm_r": ("strip", [_R_SHOULDER, _R_ELBOW]),
    "lower_arm_r": ("strip", [_R_ELBOW, _R_WRIST]),
    "upper_leg_l": ("strip", [_L_HIP, _L_KNEE]),
    "lower_leg_l": ("strip", [_L_KNEE, _L_ANKLE]),
    "upper_leg_r": ("strip", [_R_HIP, _R_KNEE]),
    "lower_leg_r": ("strip", [_R_KNEE, _R_ANKLE]),
}

# Draw order: torso first so limb segments paint over its edges at joints.
Z_ORDER: List[str] = [
    "torso",
    "upper_arm_l", "upper_arm_r",
    "lower_arm_l", "lower_arm_r",
    "upper_leg_l", "upper_leg_r",
    "lower_leg_l", "lower_leg_r",
]

# A strip's half-width, as a fraction of its own bone length.
_STRIP_HALF_WIDTH_RATIO = 0.20

# Landmarks below this visibility/presence score are treated as absent —
# the segment that depends on them is skipped for that frame rather than
# warped from a guess.
_VISIBILITY_THRESHOLD = 0.5

# A live/avatar bone-length ratio outside this band is clamped to the
# nearest edge before warping, so a mismatched-proportion avatar stretches
# to a bounded, if imperfect, size instead of growing without limit.
_LENGTH_RATIO_MIN = 0.5
_LENGTH_RATIO_MAX = 1.8

# Minimum number of usable segments (torso + at least one limb pair) for an
# avatar photo to be considered worth compositing at all.
_MIN_USABLE_SEGMENTS = 3

_LOCK = threading.Lock()
_POSE_LANDMARKER_IMAGE = None  # IMAGE mode, used once on the avatar photo
_POSE_LANDMARKER_VIDEO = None  # VIDEO mode, used per live frame
_VIDEO_TIMESTAMP_MS = 0
_SEGMENTER = None


class _Segment:
    __slots__ = ("kind", "rgb", "alpha", "pts", "bone_length",
                 "half_width_a", "half_width_b")

    def __init__(self, kind: str, rgb: np.ndarray, alpha: np.ndarray,
                 pts: np.ndarray, bone_length: float,
                 half_width_a: float = 0.0, half_width_b: float = 0.0):
        self.kind = kind          # "quad" or "strip"
        self.rgb = rgb            # cropped avatar-space RGB patch
        self.alpha = alpha        # matching alpha mask, same shape
        self.pts = pts            # defining points *within the crop*, float32
        # "strip": bone_length is the a->b bone length; half_width_a/b unused.
        # "quad" (torso): bone_length is the shoulder-center->hip-center
        # spine length, half_width_a is half the shoulder width, half_width_b
        # is half the hip width — enough to rebuild a clamped, well-behaved
        # quad at apply time instead of trusting raw (possibly degenerate)
        # live landmark positions.
        self.bone_length = bone_length
        self.half_width_a = half_width_a
        self.half_width_b = half_width_b


# Cached avatar-side extraction, keyed by path + mtime.
_SOURCE_CACHE_KEY: Optional[tuple] = None
_SOURCE_SEGMENTS: Dict[str, _Segment] = {}


def _models_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "models")


def _mp_python_vision():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    return mp_python, vision


def get_pose_landmarker(video_mode: bool):
    """Lazily build (and cache) a MediaPipe PoseLandmarker.

    ``video_mode=False`` returns a shared IMAGE-mode instance for one-off use
    on the static avatar photo. ``video_mode=True`` returns a shared
    VIDEO-mode instance that must be fed monotonically increasing timestamps
    (handled internally by :func:`detect_live_pose`) so MediaPipe's own
    ROI-based tracker can smooth landmarks across frames.
    """
    global _POSE_LANDMARKER_IMAGE, _POSE_LANDMARKER_VIDEO

    cached = _POSE_LANDMARKER_VIDEO if video_mode else _POSE_LANDMARKER_IMAGE
    if cached is not None:
        return cached

    with _LOCK:
        cached = _POSE_LANDMARKER_VIDEO if video_mode else _POSE_LANDMARKER_IMAGE
        if cached is not None:
            return cached

        model_path = os.path.join(_models_dir(), POSE_MODEL_FILENAME)
        if not os.path.exists(model_path):
            print(f"[{NAME}] {POSE_MODEL_FILENAME} not found in models/. "
                  f"Download it from {POSE_MODEL_URL}")
            return None

        try:
            mp_python, vision = _mp_python_vision()
        except ImportError:
            print(f"[{NAME}] mediapipe is not installed — avatar body disabled.")
            return None

        running_mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        try:
            options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=running_mode,
                num_poses=1,
            )
            landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:  # never break the swap pipeline
            print(f"[{NAME}] failed to load pose landmarker: {exc}")
            return None

        if video_mode:
            _POSE_LANDMARKER_VIDEO = landmarker
        else:
            _POSE_LANDMARKER_IMAGE = landmarker
        return landmarker


def get_body_segmenter():
    """Lazily build the MediaPipe selfie_multiclass body/clothes segmenter."""
    global _SEGMENTER
    if _SEGMENTER is not None:
        return _SEGMENTER

    with _LOCK:
        if _SEGMENTER is not None:
            return _SEGMENTER

        model_path = os.path.join(_models_dir(), SEG_MODEL_FILENAME)
        if not os.path.exists(model_path):
            print(f"[{NAME}] {SEG_MODEL_FILENAME} not found in models/. "
                  f"Download it from {SEG_MODEL_URL}")
            return None

        try:
            mp_python, vision = _mp_python_vision()
        except ImportError:
            print(f"[{NAME}] mediapipe is not installed — avatar body disabled.")
            return None

        try:
            options = vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                output_category_mask=True,
            )
            _SEGMENTER = vision.ImageSegmenter.create_from_options(options)
        except Exception as exc:  # never break the swap pipeline
            print(f"[{NAME}] failed to load body segmenter: {exc}")
            return None
        return _SEGMENTER


def _segment_body(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Return a uint8 body+clothes mask (0/255) for a BGR image, or None."""
    segmenter = get_body_segmenter()
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
        mask = np.isin(category, (_BODY_SKIN_CATEGORY, _CLOTHES_CATEGORY))
        mask = (mask.astype(np.uint8)) * 255
        if mask.shape[:2] != image_bgr.shape[:2]:
            mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        return mask
    except Exception as exc:  # never break the swap pipeline
        print(f"[{NAME}] body segmentation failed: {exc}")
        return None


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Drop specks, close pinholes, and feather the border into an alpha ramp."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(mask, (0, 0), 4)


def _landmark_xy(landmarks, idx: int, w: int, h: int) -> Optional[np.ndarray]:
    """Pixel-space (x, y) for landmark ``idx``, or None if too uncertain."""
    if landmarks is None or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    visibility = getattr(lm, "visibility", 1.0)
    presence = getattr(lm, "presence", 1.0)
    if visibility is not None and visibility < _VISIBILITY_THRESHOLD:
        return None
    if presence is not None and presence < _VISIBILITY_THRESHOLD:
        return None
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _strip_quad(p_a: np.ndarray, p_b: np.ndarray, half_width: float) -> np.ndarray:
    """4-point quad around bone a->b, offset perpendicular by ``half_width``."""
    direction = p_b - p_a
    length = float(np.linalg.norm(direction))
    if length < 1e-3:
        perp = np.array([1.0, 0.0], dtype=np.float32)
    else:
        perp = np.array([-direction[1], direction[0]], dtype=np.float32) / length
    offset = perp * half_width
    return np.array([p_a + offset, p_b + offset, p_b - offset, p_a - offset],
                     dtype=np.float32)


def _clamp_length_ratio(live_len: float, avatar_len: float) -> float:
    """Effective scale to apply to a stretch, clamped to a sane band."""
    if avatar_len < 1e-3:
        return 1.0
    ratio = live_len / avatar_len
    return float(np.clip(ratio, _LENGTH_RATIO_MIN, _LENGTH_RATIO_MAX))


def _bbox_from_pts(pts: np.ndarray, w: int, h: int, pad: int = 4) -> Tuple[int, int, int, int]:
    x0 = max(0, int(np.floor(pts[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - pad)
    x1 = min(w, int(np.ceil(pts[:, 0].max())) + pad)
    y1 = min(h, int(np.ceil(pts[:, 1].max())) + pad)
    return x0, y0, x1, y1


def prepare_source(avatar_path: str) -> bool:
    """Extract and cache per-segment textures from the avatar image once.

    Returns True when at least a usable subset of segments (torso plus one
    limb pair) could be built. The expensive pose + segmentation inference
    never touches the per-frame path.
    """
    global _SOURCE_CACHE_KEY, _SOURCE_SEGMENTS

    if not avatar_path:
        return False

    try:
        mtime = os.path.getmtime(avatar_path)
    except OSError:
        return False

    key = (avatar_path, mtime)
    if key == _SOURCE_CACHE_KEY and _SOURCE_SEGMENTS:
        return True

    from modules import imread_unicode
    avatar_img = imread_unicode(avatar_path)
    if avatar_img is None:
        return False

    landmarker = get_pose_landmarker(video_mode=False)
    if landmarker is None:
        return False

    try:
        import mediapipe as mp
        rgb = cv2.cvtColor(avatar_img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
    except Exception as exc:  # never break the swap pipeline
        print(f"[{NAME}] avatar pose detection failed: {exc}")
        return False

    if not result.pose_landmarks:
        _SOURCE_CACHE_KEY = key
        _SOURCE_SEGMENTS = {}
        print(f"[{NAME}] no person/pose detected in the avatar image.")
        return False

    landmarks = result.pose_landmarks[0]
    h, w = avatar_img.shape[:2]

    body_mask = _segment_body(avatar_img)
    if body_mask is None or int((body_mask > 127).sum()) < 500:
        _SOURCE_CACHE_KEY = key
        _SOURCE_SEGMENTS = {}
        print(f"[{NAME}] no usable body/clothing region found.")
        return False
    alpha_full = _clean_mask(body_mask)

    segments: Dict[str, _Segment] = {}
    for name, (kind, idxs) in SEGMENT_DEFS.items():
        pts_world = [_landmark_xy(landmarks, i, w, h) for i in idxs]
        if any(p is None for p in pts_world):
            continue

        half_width_a = 0.0
        half_width_b = 0.0
        if kind == "quad":
            # Order is [L-shoulder, R-shoulder, R-hip, L-hip]. Store the
            # spine length (shoulder-center -> hip-center) plus half the
            # shoulder/hip widths separately, instead of just the raw quad,
            # so apply_body() can rebuild a clamped destination quad rather
            # than trusting four independent live points that can go
            # degenerate (e.g. hips foreshortened/occluded while sitting).
            shoulder_l, shoulder_r, hip_r, hip_l = pts_world
            shoulder_center = (shoulder_l + shoulder_r) / 2.0
            hip_center = (hip_l + hip_r) / 2.0
            bone_length = float(np.linalg.norm(hip_center - shoulder_center))
            if bone_length < 1e-3:
                continue
            half_width_a = float(np.linalg.norm(shoulder_r - shoulder_l)) / 2.0
            half_width_b = float(np.linalg.norm(hip_l - hip_r)) / 2.0
            if half_width_a < 1e-3 or half_width_b < 1e-3:
                continue
            quad = np.stack(pts_world, axis=0)
        else:
            p_a, p_b = pts_world
            bone_length = float(np.linalg.norm(p_b - p_a))
            if bone_length < 1e-3:
                continue
            half_width = bone_length * _STRIP_HALF_WIDTH_RATIO
            quad = _strip_quad(p_a, p_b, half_width)

        x0, y0, x1, y1 = _bbox_from_pts(quad, w, h)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue

        crop_rgb = avatar_img[y0:y1, x0:x1].copy()
        crop_alpha = alpha_full[y0:y1, x0:x1].copy()
        local_pts = (quad - np.array([x0, y0], dtype=np.float32))
        segments[name] = _Segment(kind, crop_rgb, crop_alpha, local_pts,
                                   bone_length, half_width_a, half_width_b)

    if "torso" not in segments:
        _SOURCE_CACHE_KEY = key
        _SOURCE_SEGMENTS = {}
        print(f"[{NAME}] torso not found in avatar image — cannot compose an overlay.")
        return False

    if len(segments) < _MIN_USABLE_SEGMENTS:
        _SOURCE_CACHE_KEY = key
        _SOURCE_SEGMENTS = {}
        print(f"[{NAME}] not enough visible body segments in the avatar image.")
        return False

    _SOURCE_SEGMENTS = segments
    _SOURCE_CACHE_KEY = key
    return True


def detect_live_pose(frame: Frame):
    """Run VIDEO-mode pose detection on a live frame.

    Returns the MediaPipe pose_landmarks list for the first detected person,
    or None on failure/no person — never raises.
    """
    global _VIDEO_TIMESTAMP_MS

    landmarker = get_pose_landmarker(video_mode=True)
    if landmarker is None:
        return None

    try:
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        _VIDEO_TIMESTAMP_MS = max(_VIDEO_TIMESTAMP_MS + 1, int(time.time() * 1000))
        result = landmarker.detect_for_video(mp_image, _VIDEO_TIMESTAMP_MS)
    except Exception as exc:  # never break the swap pipeline
        print(f"[{NAME}] live pose detection failed: {exc}")
        return None

    if not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]


def apply_body(frame: Frame, live_landmarks) -> Frame:
    """Composite the cached avatar body segments onto ``frame``.

    Each segment is warped and blended independently; a segment whose live
    landmarks are missing/low-confidence is simply skipped for this frame.
    Returns ``frame`` unchanged if no avatar source is cached.
    """
    if not _SOURCE_SEGMENTS or live_landmarks is None:
        return frame

    h, w = frame.shape[:2]
    strength = float(getattr(modules.globals, "avatar_body_strength", 100.0)) / 100.0
    if strength <= 0.0:
        return frame

    out = frame
    for name in Z_ORDER:
        seg = _SOURCE_SEGMENTS.get(name)
        if seg is None:
            continue
        kind, idxs = SEGMENT_DEFS[name]

        try:
            live_pts = [_landmark_xy(live_landmarks, i, w, h) for i in idxs]
            if any(p is None for p in live_pts):
                continue

            if kind == "quad":
                # Rebuild the torso quad from clamped spine/width ratios
                # instead of trusting the four raw live points directly.
                # A pose far from the avatar's own (e.g. sitting vs a
                # standing avatar photo) can foreshorten/occlude hips enough
                # that MediaPipe still reports "confident" but geometrically
                # implausible points; feeding those straight into
                # getPerspectiveTransform can blow the warp up to cover the
                # whole frame. Clamping keeps the quad bounded even then.
                shoulder_l, shoulder_r, hip_r, hip_l = live_pts
                shoulder_center = (shoulder_l + shoulder_r) / 2.0
                hip_center = (hip_l + hip_r) / 2.0
                spine_vec = hip_center - shoulder_center
                spine_len = float(np.linalg.norm(spine_vec))
                if spine_len < 1e-3:
                    continue
                spine_scale = _clamp_length_ratio(spine_len, seg.bone_length)
                hip_center = shoulder_center + (spine_vec / spine_len) * (
                    seg.bone_length * spine_scale
                )

                shoulder_vec = shoulder_l - shoulder_r
                shoulder_width = float(np.linalg.norm(shoulder_vec))
                if shoulder_width < 1e-3:
                    continue
                side = shoulder_vec / shoulder_width
                shoulder_scale = _clamp_length_ratio(
                    shoulder_width, seg.half_width_a * 2.0
                )
                eff_shoulder_half = seg.half_width_a * shoulder_scale

                hip_width = float(np.linalg.norm(hip_l - hip_r))
                hip_scale = _clamp_length_ratio(
                    hip_width, seg.half_width_b * 2.0
                ) if hip_width >= 1e-3 else shoulder_scale
                eff_hip_half = seg.half_width_b * hip_scale

                dst_quad = np.array([
                    shoulder_center + side * eff_shoulder_half,   # L-shoulder
                    shoulder_center - side * eff_shoulder_half,   # R-shoulder
                    hip_center - side * eff_hip_half,              # R-hip
                    hip_center + side * eff_hip_half,              # L-hip
                ], dtype=np.float32)

                # Degenerate/near-zero-area quad (e.g. shoulders and hips
                # nearly collapsed onto each other) — skip rather than warp
                # a sliver into a huge destination.
                area = cv2.contourArea(dst_quad)
                if area < (eff_shoulder_half + eff_hip_half) * seg.bone_length * spine_scale * 0.05:
                    continue
            else:
                p_a, p_b = live_pts
                live_bone_len = float(np.linalg.norm(p_b - p_a))
                if live_bone_len < 1e-3:
                    continue
                scale = _clamp_length_ratio(live_bone_len, seg.bone_length)
                effective_len = seg.bone_length * scale
                direction = (p_b - p_a)
                norm = float(np.linalg.norm(direction))
                if norm < 1e-3:
                    continue
                unit = direction / norm
                p_b_clamped = p_a + unit * effective_len
                half_width = effective_len * _STRIP_HALF_WIDTH_RATIO
                dst_quad = _strip_quad(p_a, p_b_clamped, half_width)

            matrix = cv2.getPerspectiveTransform(
                seg.pts.astype(np.float32), dst_quad.astype(np.float32)
            )
            warped_rgb = cv2.warpPerspective(
                seg.rgb, matrix, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            warped_alpha = cv2.warpPerspective(
                seg.alpha, matrix, (w, h), flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )

            alpha = (warped_alpha.astype(np.float32) / 255.0) * strength
            if float(alpha.max()) <= 0.01:
                continue
            alpha = alpha[:, :, None]
            out = (warped_rgb.astype(np.float32) * alpha
                   + out.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
        except Exception as exc:  # never break the swap pipeline
            print(f"[{NAME}] segment '{name}' warp failed: {exc}")
            continue

    return out


def reset_source_cache() -> None:
    """Force the next prepare_source() call to re-extract from the avatar image."""
    global _SOURCE_CACHE_KEY, _SOURCE_SEGMENTS
    _SOURCE_CACHE_KEY = None
    _SOURCE_SEGMENTS = {}
